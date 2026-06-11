import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from metrics import calculate_event_metrics

import CLUEstering as clue

# Load optimal parameters
print("Loading best parameters from 'best_params.json'")
try:
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
except FileNotFoundError:
    print("Error: 'best_params.json' not found. Please run train_CLUE.py first.")
    exit()

dc = best_params['dc']
rhoc = best_params['rhoc']

# Load parquet files on DIAS
BASE_DATA_DIR = "/home/xzcappon/phd/projects/maskformer/colliderml/data/ttbar_p0"
file_pattern = os.path.join(BASE_DATA_DIR, "*", "calo_hits", "*.parquet")
all_files = sorted(glob.glob(file_pattern))
num_files = len(all_files)

# worker function for parallel processing
def process_single_event(file_path):
    try:
        event = pd.read_parquet(file_path).iloc[0]

        # Format spatial data for CLUE
        clue_input_df = pd.DataFrame({
            'x0': event['x'],
            'x1': event['y'],
            'x2': event['z'],  
            'weight': event['total_energy'] 
        })

        # Run CLUE with the optimised parameters
        clust = clue.clusterer(dc, rhoc) 
        clust.read_data(clue_input_df)
        clust.run_clue()

        # calculate metrics
        purity, efficiency = calculate_event_metrics(event, clust.output_df)

        return purity, efficiency
    except Exception:
        # catch curropted files
        return 0.0,0.0 

if __name__ == "__main__":
    print(f"\nStarting evaluation on {num_files} calo files using 16 cores...")
    event_purities, event_efficiencies = [], []
    completed = 0

    # run in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = executor.map(process_single_event, all_files, chunksize=1000)
        
        for p, e in results:
            if p > 0 or e > 0:
                event_purities.append(p)
                event_efficiencies.append(e)
            completed += 1
            if completed % 1000 == 0:
                print(f"Processed {completed}/{num_files} files...")

    # save raw metrics
    results_df = pd.DataFrame({
        'Purity': event_purities,
        'Efficiency': event_efficiencies
    })
    results_df.to_csv('validation_histograms.csv', index=False)
    print("Saved raw metrics to 'validation_histograms.csv'")

    # Baseline results
    final_purity = np.mean(event_purities)
    final_efficiency = np.mean(event_efficiencies)
    final_f1 = 2 * (final_purity * final_efficiency) / (final_purity + final_efficiency)

    print("\n--- Final Validation Results ---")
    print(f"Final Average Purity:     {final_purity * 100:.2f}%")
    print(f"Final Average Efficiency: {final_efficiency * 100:.2f}%")
