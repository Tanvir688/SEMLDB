"""
2DFET backend model for SEMLDB.

NumPy-only inference core for the 2DFET Two-Tower FiLM surrogate
(ballistic NEGF-Poisson training data). Reproduces the PyTorch model's
forward pass exactly (validated to ~1e-6), so no torch runtime is needed
for this device.

Device parameters (as exposed to the API / frontend):
    tox        gate oxide thickness [nm]   (trained domain: 1 - 3 nm)
    Lg         gate length [nm]            (trained domain: 6 - 30 nm)
    eps_ox     oxide dielectric constant   (trained domain: 4 - 25)
    material   MoS2 / MoSe2 / WS2 / WSe2   -> meff 0.60 / 0.50 / 0.46 / 0.44 [m0]
    transport  fixed to ballistic          -> D = 0 [eV^2]
    V_th       threshold voltage [V]       (reference 0.20, range 0.10 - 0.35)

Voltage sweeps:
    nFET Vg/Vd use the existing positive-bias convention.
    WSe2 pFET Vg/Vd use negative physical terminal voltages at the API.
    The pFET boundary adapter mirrors those voltages into the unchanged
    positive electron-equivalent model/database domain.

Outputs (dataset units):
    Id [A/m] (numerically equal to uA/um), shape [len(Vg), len(Vd)]
    Qg [C/m], shape [len(Vg), len(Vd)]
"""
import math
import os
import pickle

import numpy as np
import torch

from .. import MODELS

_MODEL_DIR = os.path.dirname(__file__)
_PTH_PATH = os.path.join(_MODEL_DIR, "TwoDFET.pth")
_SCALERS_PATH = os.path.join(_MODEL_DIR, "TwoDFET.pkl")

_LAYERNORM_EPS = 1e-5


class _PickleState:
    """State-only replacement for training classes referenced by TwoDFET.pkl."""


class _ScalerUnpickler(pickle.Unpickler):
    """Load only the known scaler artifact types without training-code imports."""

    _STATE_CLASSES = {
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("two_tower_dataset", "TargetTransformer"),
        ("two_tower_dataset", "TargetTransformConfig"),
    }
    _NUMPY_GLOBALS = {
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
    }

    def find_class(self, module, name):
        if (module, name) in self._STATE_CLASSES:
            return _PickleState
        if (module, name) in self._NUMPY_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            "Unsupported global in 2DFET scaler artifact: %s.%s" % (module, name)
        )


def _load_scaler_bundle(path):
    with open(path, "rb") as handle:
        bundle = _ScalerUnpickler(handle).load()

    required = {
        "device_scaler", "bias_scaler", "target_scaler",
        "device_feature_fields", "bias_feature_fields", "target_fields",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError("2DFET scaler artifact is missing: %s" % sorted(missing))
    return bundle

# Vectorised exact error function (stdlib math.erf) -> matches torch's exact GELU.
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _linear(x, w, b):
    # torch Linear stores weight as (out, in); y = x @ w.T + b
    return x @ w.T + b


def _gelu(x):
    # Exact GELU (nn.GELU default, approximate='none').
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _layer_norm(x, eps, w=None, b=None):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased (population) variance, like torch
    y = (x - mean) / np.sqrt(var + eps)
    if w is not None:
        y = y * w + b
    return y


class Surrogate:
    """Two-Tower FiLM surrogate loaded from the trained PyTorch checkpoint."""

    def __init__(self, pth_path=_PTH_PATH, scalers_path=_SCALERS_PATH):
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)

        self.w = {}
        for key, tensor in state.items():
            prefix = "backbone."
            clean_key = key[len(prefix):] if key.startswith(prefix) else key
            self.w[clean_key] = tensor.detach().cpu().numpy().astype(np.float64)

        config = checkpoint.get("config", {})
        fusion_type = config.get("fusion_type", "film")
        if fusion_type != "film":
            raise ValueError("2DFET checkpoint must use FiLM fusion, got %r" % fusion_type)

        scaler_bundle = _load_scaler_bundle(scalers_path)
        self.scaler = {
            name: (
                np.asarray(scaler_bundle[name].mean_, dtype=np.float64),
                np.asarray(scaler_bundle[name].scale_, dtype=np.float64),
            )
            for name in ("device_scaler", "bias_scaler", "target_scaler")
        }
        self.eps = _LAYERNORM_EPS
        self.embed = int(config.get("embed_size", 16))
        self.device_fields = list(scaler_bundle["device_feature_fields"])
        self.bias_fields = list(scaler_bundle["bias_feature_fields"])
        self.target_fields = list(scaler_bundle["target_fields"])

        checkpoint_targets = list(checkpoint.get("target_fields", self.target_fields))
        if checkpoint_targets != self.target_fields:
            raise ValueError(
                "2DFET checkpoint/scaler target mismatch: %r != %r"
                % (checkpoint_targets, self.target_fields)
            )

    def _std_transform(self, x, name):
        mean, scale = self.scaler[name]
        return (x - mean) / scale

    def _std_inverse(self, x, name):
        mean, scale = self.scaler[name]
        return x * scale + mean

    def _device_tower(self, xd):
        # xd: (N, 5) scaled device features. Per-feature tokenisation + mean pool.
        w = self.w
        emb = w["device_type_embeddings.weight"]  # (5, embed)
        n = xd.shape[0]
        tokens = np.empty((n, xd.shape[1], self.embed), dtype=np.float64)
        for i in range(xd.shape[1]):
            scalar = xd[:, i:i + 1]
            type_emb = np.broadcast_to(emb[i], (n, self.embed))
            t = np.concatenate([scalar, type_emb], axis=1)
            t = _linear(t, w["device_shared_mlp.0.weight"], w["device_shared_mlp.0.bias"])
            t = _layer_norm(t, self.eps, w["device_shared_mlp.1.weight"], w["device_shared_mlp.1.bias"])
            t = _gelu(t)
            t = _linear(t, w["device_shared_mlp.3.weight"], w["device_shared_mlp.3.bias"])
            tokens[:, i, :] = t
        h_p = tokens.mean(axis=1)
        h_p = _layer_norm(h_p, self.eps)  # F.layer_norm, no affine
        return h_p

    def _bias_tower(self, xb):
        w = self.w
        h = _linear(xb, w["bias_mlp.0.weight"], w["bias_mlp.0.bias"])
        h = _layer_norm(h, self.eps, w["bias_mlp.1.weight"], w["bias_mlp.1.bias"])
        h = _gelu(h)
        h = _linear(h, w["bias_mlp.3.weight"], w["bias_mlp.3.bias"])
        h = _layer_norm(h, self.eps, w["bias_mlp.4.weight"], w["bias_mlp.4.bias"])
        h = _gelu(h)
        h = _linear(h, w["bias_mlp.6.weight"], w["bias_mlp.6.bias"])
        return h

    def _forward_scaled(self, xd, xb):
        w = self.w
        h_p = self._device_tower(xd)
        h_v = self._bias_tower(xb)
        # FiLM fusion
        film = _linear(h_p, w["film_projection.weight"], w["film_projection.bias"])
        gamma, beta = film[:, :self.embed], film[:, self.embed:]
        h = _layer_norm(h_v * gamma + beta, self.eps)
        # Output head
        h = _linear(h, w["output_head.0.weight"], w["output_head.0.bias"])
        h = _layer_norm(h, self.eps, w["output_head.1.weight"], w["output_head.1.bias"])
        h = _gelu(h)
        out = _linear(h, w["output_head.3.weight"], w["output_head.3.bias"])
        return out

    def predict_points(self, x_device_raw, x_bias_raw):
        """Physical [Id, Q] for raw (tox,Lg,eps_ox,meff,D) [SI] & (Vg,Vd) rows."""
        x_device_raw = np.asarray(x_device_raw, dtype=np.float64).reshape(-1, len(self.device_fields))
        x_bias_raw = np.asarray(x_bias_raw, dtype=np.float64).reshape(-1, len(self.bias_fields))
        xd = self._std_transform(x_device_raw, "device_scaler")
        xb = self._std_transform(x_bias_raw, "bias_scaler")
        scaled = self._forward_scaled(xd, xb)
        y_trans = self._std_inverse(scaled, "target_scaler")  # log10(Id), Q
        y = y_trans.copy()
        id_idx = self.target_fields.index("Id")
        y[:, id_idx] = np.power(10.0, y_trans[:, id_idx])  # undo log10
        return y

    def predict_grid(self, tox, Lg, eps_ox, meff, D, Vg, Vd):
        """I-V over a Vg x Vd grid for one device. Returns (Id, Q), each (len(Vg), len(Vd))."""
        Vg = np.asarray(Vg, dtype=np.float64).ravel()
        Vd = np.asarray(Vd, dtype=np.float64).ravel()
        VG, VD = np.meshgrid(Vg, Vd, indexing="ij")
        n = VG.size
        device_row = np.array([tox, Lg, eps_ox, meff, D], dtype=np.float64)
        x_device = np.broadcast_to(device_row, (n, 5))
        x_bias = np.column_stack([VG.ravel(), VD.ravel()])
        y = self.predict_points(x_device, x_bias)
        Id = y[:, self.target_fields.index("Id")].reshape(VG.shape)
        Q = y[:, self.target_fields.index("Q")].reshape(VG.shape)
        return Id, Q


_SURROGATE = None


def _get_surrogate():
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = Surrogate()
    return _SURROGATE


def convert_str_to_float(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = convert_str_to_float(value)
            elif isinstance(value, str):
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
    return data


def parse_voltage_input(v_input):
    """Parse voltage input into numpy array."""
    if v_input is None:
        raise ValueError("Voltage input cannot be None")

    if isinstance(v_input, (int, float)):
        return np.array([float(v_input)])

    if isinstance(v_input, (list, tuple)):
        return np.array(v_input, dtype=float)

    if isinstance(v_input, np.ndarray):
        return v_input.astype(float)

    if isinstance(v_input, dict):
        if {'start', 'end', 'step'}.issubset(v_input.keys()):
            return np.linspace(v_input['start'], v_input['end'], int(v_input['step']))
        raise ValueError("Dict format must contain 'start', 'end', 'step' keys")

    raise ValueError("Unsupported voltage input type: %s" % type(v_input))


# Channel material dropdown: the frontend sends the option index. Keep this
# order aligned with the frontend labels. WSe2 is exposed as a physically
# signed pFET while its stored/trained representation remains the unchanged
# positive electron-equivalent reference.
MATERIALS = ['MoS2', 'MoSe2', 'WS2', 'WSe2']
MATERIAL_MEFF = [0.60, 0.50, 0.46, 0.44]  # [m0], aligned with MATERIALS
PFET_MATERIALS = {'WSe2'}

def _resolve_option(value, options, values, name):
    """Map a dropdown selection (index or option name) to its physical value."""
    if isinstance(value, str) and value in options:
        idx = options.index(value)
    else:
        idx = int(float(value))
    if not 0 <= idx < len(options):
        raise ValueError("Unknown %s selection: %s" % (name, value))
    return values[idx]


def _resolve_meff(parameters):
    """Map the 'material' selection (index or name) to meff; accept raw meff too."""
    if parameters.get('meff') is not None:
        return float(parameters['meff'])
    material = parameters.get('material')
    if material is None:
        raise ValueError("Missing device parameter: require 'material' (or 'meff').")
    return _resolve_option(material, MATERIALS, MATERIAL_MEFF, 'material')


def _material_from_meff(meff):
    """Return the canonical material name for one of the trained masses."""
    for material, trained_meff in zip(MATERIALS, MATERIAL_MEFF):
        if np.isclose(float(meff), trained_meff, rtol=0.0, atol=1e-12):
            return material
    raise ValueError("Unsupported 2DFET effective mass: %s" % meff)


def _material_metadata(meff):
    """Return canonical material identity and external terminal polarity."""
    material = _material_from_meff(meff)
    polarity = -1.0 if material in PFET_MATERIALS else 1.0
    return material, polarity


def _validate_external_pfet_biases(vth, Vg=None, Vd=None):
    """Reject legacy positive pFET biases at the new signed API boundary."""
    if vth >= 0.0:
        raise ValueError("WSe2 pFET requires a negative V_th.")
    if Vg is not None and np.any(np.asarray(Vg, dtype=float) > 1e-12):
        raise ValueError("WSe2 pFET requires non-positive Vg values.")
    if Vd is not None and np.any(np.asarray(Vd, dtype=float) > 1e-12):
        raise ValueError("WSe2 pFET requires non-positive Vd values.")


def _externalize_database_grid(Vg, Vd, Id, Qg, polarity):
    """Convert an electron-equivalent database grid to the external convention.

    For a pFET, reverse both positive stored axes before negating them so the
    returned negative Vg/Vd axes remain strictly increasing. Reverse the two
    corresponding data dimensions and negate Id/Qg at the same boundary.
    The stored charge contains an arbitrary off-state offset; reference each
    pFET charge curve to its electron-equivalent Vg=0 row before negation.
    This guarantees non-positive physical pFET charge without changing Cg.
    """
    if polarity > 0.0:
        return list(Vg), list(Vd), list(Id), list(Qg)

    vg = -np.asarray(Vg, dtype=float)[::-1]
    vd = -np.asarray(Vd, dtype=float)[::-1]
    current = -np.asarray(Id, dtype=float)[::-1, ::-1]
    charge_model = np.asarray(Qg, dtype=float)
    charge_model = charge_model - charge_model[0:1, :]
    charge = -charge_model[::-1, ::-1]
    return vg.tolist(), vd.tolist(), current.tolist(), charge.tolist()


def _resolve_D(parameters):
    """The current database and surrogate contain ballistic data only."""
    return 0.0


# Absolute threshold-voltage knob, SiFET-style: the stored/trained sweep is
# -0.15..0.6 V but the UI window is 0..0.5 V, leaving 0.15 V of lower margin
# and 0.1 V of upper margin for the rigid work-function shift (exact physics:
# Vg and Vfb enter the NEGF-Poisson equations only as Vg - Vfb).
VTH_REF = 0.20
VTH_MIN, VTH_MAX = 0.10, 0.35
VG_STEP = 0.0125
# Display window rows in the stored 61-point Vg grid (-0.15..0.6 V)
WIN_START, WIN_END = 12, 53  # rows 12..52 -> Vg 0.0..0.5 V (41 points)


def run_simulation(parameters):
    """Run a 2DFET simulation with the NumPy Two-Tower FiLM surrogate.

    Expects tox and Lg in nm (converted to m internally for the surrogate).
    'material' selects the channel effective mass for MoS2, MoSe2, WS2, or
    WSe2. WSe2 accepts negative physical Vg/Vd/V_th values and mirrors them
    into the same positive electron-equivalent equations used for training.
    Transport is fixed to the ballistic limit (D = 0).
    V_th [V] (reference 0.20) rigidly shifts the transfer characteristics
    along Vg: the model is evaluated at Vg - (V_th - 0.20).
    """
    parameters = convert_str_to_float(parameters)
    tox = parameters.get('tox')
    Lg = parameters.get('Lg')
    eps_ox = parameters.get('eps_ox')
    meff = _resolve_meff(parameters)
    D = _resolve_D(parameters)
    material, polarity = _material_metadata(meff)
    default_vth = polarity * VTH_REF
    vth_external = float(parameters.get('V_th', default_vth))

    if tox is None or Lg is None or eps_ox is None:
        raise ValueError("Missing device parameters: require tox, Lg, eps_ox, material.")

    Vg_array = parse_voltage_input(parameters.get('Vg'))
    Vd_array = parse_voltage_input(parameters.get('Vd'))
    if polarity < 0.0:
        _validate_external_pfet_biases(vth_external, Vg_array, Vd_array)

    # Model coordinates remain exactly as trained. Multiplication by -1 maps
    # signed WSe2 terminal values to the positive electron-equivalent domain.
    vth_model = polarity * vth_external
    dvth = vth_model - VTH_REF
    Vg_model = polarity * Vg_array
    Vd_model = polarity * Vd_array

    model = _get_surrogate()
    Id, Q = model.predict_grid(
        tox=float(tox) * 1e-9,   # nm -> m
        Lg=float(Lg) * 1e-9,     # nm -> m
        eps_ox=float(eps_ox),
        meff=float(meff),
        D=float(D),
        Vg=Vg_model - dvth,
        Vd=Vd_model,
    )

    if polarity < 0.0:
        # Use the physical Vg=0 state as the pFET charge reference. The
        # equivalent model coordinate includes the same threshold shift as the
        # requested sweep. Subtracting this per-Vd constant preserves Cg.
        _, Q_reference = model.predict_grid(
            tox=float(tox) * 1e-9,
            Lg=float(Lg) * 1e-9,
            eps_ox=float(eps_ox),
            meff=float(meff),
            D=float(D),
            Vg=np.array([-dvth], dtype=float),
            Vd=Vd_model,
        )
        Q = Q - Q_reference

    # Restore the physical terminal convention at the API boundary. Because
    # both the pFET voltage and response signs are mirrored, dId/dVg and
    # dQg/dVg remain positive without taking absolute values.
    Id = polarity * Id
    Q = polarity * Q

    device_params = dict(parameters)
    device_params['material'] = material
    device_params['meff'] = meff  # resolved from 'material'
    device_params['D'] = D        # fixed ballistic value
    device_params['V_th'] = vth_external
    device_params['polarity'] = int(polarity)
    device_params['device_type'] = 'pFET' if polarity < 0.0 else 'nFET'
    device_params['bias_convention'] = 'physical_signed'

    return {
        'simulation_data': {
            'Vg': Vg_array.tolist(),
            'Vd': Vd_array.tolist(),
            'Id': Id.tolist(),   # [len(Vg), len(Vd)], A/m (= uA/um)
            'Qg': Q.tolist(),    # [len(Vg), len(Vd)], C/m
        },
        'device_params': device_params,
    }


def _format_database_document(device):
    """Return the public database response shape for one MongoDB document."""
    if not device:
        return None

    simulation_data = device.get('simulation_data', {})
    if not simulation_data:
        return None

    return {
        'device': device.get('device'),
        'device_params': device.get('device_params', {}),
        'simulation_data': simulation_data,
    }


def _get_material_scoped_simulation_data(db_helper, parameters):
    """Find a 2DFET record without crossing material or transport axes.

    This device-local query preserves SEMLDB's low-coupling contract: the
    shared DBHelper remains unchanged, while nearest-geometry fallback is
    restricted to records with the requested effective mass and scattering
    parameter.
    """
    exact_query = {'device': '2DFET'}
    exact_query.update({
        'device_params.%s' % key: value
        for key, value in parameters.items()
    })

    device = db_helper.collection.find_one(exact_query)
    complete_data = _format_database_document(device)
    if complete_data:
        return complete_data, True, None, parameters

    scoped_query = {
        'device': '2DFET',
        'device_params.meff': parameters['meff'],
        'device_params.D': parameters['D'],
    }
    available_params = list(db_helper.collection.find(
        scoped_query,
        {'_id': 1, 'device_params': 1},
    ))
    if not available_params:
        return None, False, None, None

    nearest_id, nearest_params, distance = db_helper.find_nearest_parameters(
        parameters,
        available_params,
    )
    if nearest_id is None:
        return None, False, None, None

    device = db_helper.collection.find_one({'_id': nearest_id})
    complete_data = _format_database_document(device)
    if not complete_data:
        return None, False, None, None

    return complete_data, False, distance, nearest_params


def get_simulation_data(db_helper, parameters):
    """Fetch pre-computed 2DFET simulation data from the database (SiFET-style).

    'material' is translated to meff, and D is fixed to zero for ballistic
    transport, matching the current database.
    V_th is not a database axis: the stored grid is wider (-0.15..0.6 V) than
    the fixed 0..0.5 V electron-equivalent display window. WSe2 accepts a
    negative physical threshold, converts its magnitude for window selection,
    then returns signed, increasing negative axes and signed Id/Qg.
    """
    parameters = convert_str_to_float(parameters)
    meff = _resolve_meff(parameters)
    material, polarity = _material_metadata(meff)
    default_vth = polarity * VTH_REF
    vth_external = float(parameters.get('V_th', default_vth))
    if polarity < 0.0:
        _validate_external_pfet_biases(vth_external)
    vth_model = polarity * vth_external
    vth_model = min(max(vth_model, VTH_MIN), VTH_MAX)
    vth_external = polarity * vth_model
    vth_shift = vth_model - VTH_REF
    D = _resolve_D(parameters)

    db_query_params = {k: v for k, v in parameters.items()
                       if k not in ('V_th', 'material', 'meff', 'transport', 'D')}
    db_query_params['meff'] = meff
    db_query_params['D'] = D

    complete_data, exact_match, distance, matched_params = \
        _get_material_scoped_simulation_data(db_helper, db_query_params)

    if not complete_data:
        return None, False, None, None

    sd = complete_data.get('simulation_data', {})
    vg_values = sd.get('Vg', [])
    vd_values = sd.get('Vd', [])
    id_data = sd.get('Id', [])
    qg_data = sd.get('Qg', [])

    # Slide the fixed display window over the stored grid
    index_shift = -int(round(vth_shift / VG_STEP))
    total_points = len(vg_values)
    start_idx = max(0, min(WIN_START + index_shift, total_points))
    end_idx = max(0, min(WIN_END + index_shift, total_points))

    selected_vg = vg_values[start_idx:end_idx]
    selected_id = id_data[start_idx:end_idx]
    selected_qg = qg_data[start_idx:end_idx]

    # Relabel the window back to the fixed 0..0.5 V axis
    shifted_vg = [round(vg + vth_shift, 4) + 0.0 for vg in selected_vg]  # +0.0 normalizes -0.0

    response_vg, response_vd, response_id, response_qg = \
        _externalize_database_grid(
            shifted_vg,
            vd_values,
            selected_id,
            selected_qg,
            polarity,
        )

    simulation_data = {
        'Vg': response_vg,
        'Vd': response_vd,
        'Id': response_id,
        'Qg': response_qg,
        'nVg': len(response_vg),
        'nVd': len(response_vd),
    }

    device_params = dict(complete_data.get('device_params', {}))
    device_params['V_th'] = vth_external
    device_params['material'] = material
    device_params['meff'] = meff
    device_params['D'] = D
    device_params['polarity'] = int(polarity)
    device_params['device_type'] = 'pFET' if polarity < 0.0 else 'nFET'
    device_params['bias_convention'] = 'physical_signed'
    adjusted_data = {
        'simulation_data': simulation_data,
        'device_params': device_params,
    }

    return adjusted_data, exact_match, distance, matched_params


@MODELS.register("2DFET")
class TwoDFET:
    simulation_func = staticmethod(run_simulation)
    device_params = ['tox', 'Lg', 'eps_ox', 'material', 'V_th']
    voltage_params = ['Vg', 'Vd']
    postprocess = staticmethod(get_simulation_data)


if __name__ == "__main__":
    parameters = {
        'tox': 2.0,
        'Lg': 10.0,
        'eps_ox': 20.0,
        'meff': 0.5,
        'D': 0.0,
        'Vg': {'start': -0.15, 'end': 0.6, 'step': 61},
        'Vd': {'start': 0.001, 'end': 0.501, 'step': 41},
    }
    result = run_simulation(parameters)
    Id = np.array(result['simulation_data']['Id'])
    print("Id grid", Id.shape, "min", Id.min(), "max", Id.max())
