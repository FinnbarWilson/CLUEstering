import pandas as pd

def calculate_event_metrics(event, results_df):
    """
    Calculates cluster purity and efficiency for a single high-energy physics event.
    
    Args:
        event (pd.Series): The raw truth data for a single event.
        results_df (pd.DataFrame): The output DataFrame from the CLUEstering algorithm.
        
    Returns:
        tuple: (average_purity, average_efficiency) as floats.
    """

    # Dataframe combining hit data, truth data, and CLUEs predictions
    hits_df = pd.DataFrame({
        'weight': event['total_energy'],
        'contrib_particle_ids': event['contrib_particle_ids'],
        'contrib_energies': event['contrib_energies'],
        'cluster_id': results_df['cluster_ids']
    })

    # Edge case: if CLUE found zero clusters
    clustered_hits = hits_df[hits_df['cluster_id'] != -1].copy()
    if clustered_hits.empty:
        return 0, 0

    # Explode truth data of contributing particles into individual rows per hit
    exploded_hits = clustered_hits.explode(['contrib_particle_ids', 'contrib_energies']) # type: ignore
    exploded_hits['contrib_energies'] = exploded_hits['contrib_energies'].astype(float)

    # Find main particle type per cluster
    # Sum true energy grouped by cluster and particle ID
    cluster_particle_energy = pd.DataFrame(exploded_hits.groupby(['cluster_id', 'contrib_particle_ids'], as_index=False)['contrib_energies'].sum())

    # Sort by energy and drop duplicates to keep only the highest energy particle per cluster
    dominant_particles = cluster_particle_energy.sort_values('contrib_energies', ascending=False).drop_duplicates('cluster_id')
    dominant_particles = dominant_particles.rename(columns={
        'contrib_particle_ids': 'dom_particle_id', 
        'contrib_energies': 'dom_energy_in_cluster'
    })

    # Calculate purity
    cluster_total_energy = pd.DataFrame(clustered_hits.groupby('cluster_id', as_index=False)['weight'].sum())
    purity_df = pd.merge(dominant_particles, cluster_total_energy, on='cluster_id')
    purity_df['purity'] = purity_df['dom_energy_in_cluster'] / purity_df['weight']
        
    # Calculate efficiency
    # need the true total deposited energy for the dominant particles across the whole event (including noise hits (-1))
    all_hits_exploded = hits_df.explode(['contrib_particle_ids', 'contrib_energies'])
    all_hits_exploded['contrib_energies'] = all_hits_exploded['contrib_energies'].astype(float)
        
    particle_total_energy = pd.DataFrame(all_hits_exploded.groupby('contrib_particle_ids', as_index=False)['contrib_energies'].sum())
    particle_total_energy = particle_total_energy.rename(columns={
        'contrib_particle_ids': 'dom_particle_id', 
        'contrib_energies': 'true_total_energy'
    })

    # Merge and calculate efficiency
    metrics_df = pd.merge(purity_df, particle_total_energy, on='dom_particle_id')
    metrics_df['efficiency'] = metrics_df['dom_energy_in_cluster'] / metrics_df['true_total_energy']
        
    # Return the average metrics for this event
    return metrics_df['purity'].mean(), metrics_df['efficiency'].mean()