import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from lifelines import CoxPHFitter


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

# Load the data
data = pd.read_csv('heart_valve.csv')

# Select only the first observation for each patient
first_obs = data.groupby('num').first().reset_index()

# Preprocess the data
first_obs['time'] = first_obs['fuyrs'].astype(float)  # Use follow-up years as time
first_obs['status'] = first_obs['status'].astype(int)
first_obs['lvmi'] = first_obs['lvmi'].astype(float)

# Select preoperative variables and covariates
preop_vars = ['age', 'sex', 'bsa', 'lvh', 'prenyha', 'redo', 'size', 'con.cabg', 
              'creat', 'dm', 'acei', 'lv', 'emergenc', 'hc', 'sten.reg.mix']

# Prepare the dataset for Cox model
cox_data = first_obs[['num', 'time', 'status', 'lvmi'] + preop_vars]

# Fit the Cox model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='status',
        formula='+'.join(preop_vars))

# Save the fitted model for plotting in plot.py
with open('fitted_cox_model.pkl', 'wb') as f:
    pickle.dump(cph, f)


results = cph.summary
results = results[['coef', 'exp(coef)', 'se(coef)', 'p', 'coef lower 95%', 'coef upper 95%']]
results = json.loads(cph.summary.to_json())

out_dir = args.out_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)
with open(f'{out_dir}/final_info.json', "w") as f:
    json.dump(results, f)
cph.summary.to_csv(f'{out_dir}/all_results.csv')