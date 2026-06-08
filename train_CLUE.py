import pandas as pd
import numpy as np
import optuna
import json
import CLUEstering as clue

from metrics import calculate_event_metrics

# Load traning data
PATH_TO_CALO_DATA = "/Users/finn/Documents/Code/Large_Datasets/ColliderML_R0/calo_hits/events0-999.parquet"
calo_data = pd.read_parquet(PATH_TO_CALO_DATA)

TRAIN_EVENTS = 100
train_data = calo_data.iloc[:TRAIN_EVENTS]

def objective(trial):
    """Optuna objective function for hyperparameter tuning"""

    # Define hyperparameters to tune
    dc = trial.suggest_float('dc', 20.0, 70.0)
    rhoc = trial.suggest_float('rhoc', 0.005, 0.1, log=True) # log scale for higher sensitivity

    event_purities = []
    event_efficiencies = []

    # Loop over events to collect metrics
    for i in range(TRAIN_EVENTS):
        event = calo_data.iloc[i]
        
        # Format spatial data for CLUE
        clue_input_df = pd.DataFrame({
            'x0': event['x'],
            'x1': event['y'],
            'x2': event['z'],  
            'weight': event['total_energy'] 
        })
        
        # Run CLUE
        clust = clue.clusterer(dc, rhoc) 
        clust.read_data(clue_input_df)
        clust.run_clue()
        
        # Calculate proper metrics
        purity, efficiency = calculate_event_metrics(event, clust.output_df)
        
        event_purities.append(purity)
        event_efficiencies.append(efficiency)

    # Results for this trial
    avg_purity = np.mean(event_purities)
    avg_efficiency = np.mean(event_efficiencies)
    
    # Edge case: If the algorithm fails return 0
    if avg_purity == 0 or avg_efficiency == 0:
        return 0.0
    
    # Maximise the harmonic mean (F1-score) of Purity and Efficiency
    f1_score = 2 * (avg_purity * avg_efficiency) / (avg_purity + avg_efficiency)
    
    trial.set_user_attr("Purity", avg_purity)
    trial.set_user_attr("Efficiency", avg_efficiency)
    
    return float(f1_score)


if __name__ == "__main__":
    print(f"Starting optuna optimisation on {TRAIN_EVENTS} traning events...")

    # Need highest possible score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("\n--- Traning Results ---")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    print(f"Resulting Average Purity:     {study.best_trial.user_attrs['Purity'] * 100:.2f}%")
    print(f"Resulting Average Efficiency: {study.best_trial.user_attrs['Efficiency'] * 100:.2f}%")

    # Save the parameters to a JSON file
    with open('best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    print("\nSaved optimal parameters to 'best_params.json'")