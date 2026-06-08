import pandas as pd
import numpy as np
import json
import CLUEstering as clue

from metrics import calculate_event_metrics

# Load validation data
PATH_TO_CALO_DATA = "/Users/finn/Documents/Code/Large_Datasets/ColliderML_R0/calo_hits/events0-999.parquet"
calo_data = pd.read_parquet(PATH_TO_CALO_DATA)

# Skip the first 100 events used for traning
TRAIN_EVENTS = 100
val_data = calo_data.iloc[TRAIN_EVENTS:]
num_val_events = len(val_data)

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
print(f"Using dc={dc:.4f}, rhoc={rhoc:.4f}")

# Run evaluation
print(f"\nStarting evaluation on {num_val_events} unseen validation events...")

event_purities = []
event_efficiencies = []

for i in range(num_val_events):
    event = val_data.iloc[i]
    
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
    
    # Calculate metrics
    purity, efficiency = calculate_event_metrics(event, clust.output_df)
    
    # Append results (ignoring complete algorithm failures to not skew the mean)
    if purity > 0 or efficiency > 0:
        event_purities.append(purity)
        event_efficiencies.append(efficiency)

    # Progress tracker
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{num_val_events} events...")

# Baseline results
final_purity = np.mean(event_purities)
final_efficiency = np.mean(event_efficiencies)
final_f1 = 2 * (final_purity * final_efficiency) / (final_purity + final_efficiency)

print("\n--- Final Validation Results ---")
print(f"Tested on: {num_val_events} events")
print(f"Parameters: dc = {dc:.4f}, rhoc = {rhoc:.4f}")
print(f"Final Average Purity:     {final_purity * 100:.2f}%")
print(f"Final Average Efficiency: {final_efficiency * 100:.2f}%")
print(f"Final F1 Score:           {final_f1:.4f}")
