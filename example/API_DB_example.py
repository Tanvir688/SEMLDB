import requests
import json
import numpy as np


server_url = 'https://semldb.rc.ufl.edu:443/'
get_sim_url = f'{server_url}/simulation_data'

### Device parameters
params = {
            'device': "CNTFET",
            'tox': 2.0,
            'Lg': 12,
            'eps_ox': 25,
            'V_th': 0.358,
            'sca_flag': 1,
        }


response = requests.get(
        get_sim_url,
        params=params
    )
    
if response.status_code == 200:
    complete_data = response.json().get('data', {})
    simulation_data = complete_data.get('simulation_data', {})

    is_exact_match = response.json().get('exact_match', None)
    if not is_exact_match:
        print("Failed to fetch exact same device in DB")
        print("Use the nearest data point based on device: ", response.json().get('matched_parameters'))
    Vg_data = np.array(simulation_data.get('Vg', []))
    Vd_data = np.array(simulation_data.get('Vd', []))
    Id_data = np.array(simulation_data.get('Id', []))
    
    print(f"shape of data responsed by API: Vg{Vg_data.shape}, Vd{Vd_data.shape}, Id{Id_data.shape}")
else:
    print(f'Batch API request failed: Status code {response.status_code}')
