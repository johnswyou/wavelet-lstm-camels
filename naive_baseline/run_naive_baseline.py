import os
import pandas as pd
import numpy as np
import pickle

def compute_nse(obs, sim):
    """
    Compute the Nash-Sutcliffe Efficiency (NSE) coefficient.
    
    Parameters
    ----------
    obs : array-like
        Observed values
    sim : array-like
        Simulated (or forecasted) values
    
    Returns
    -------
    float
        NSE value
    """
    obs = np.array(obs, dtype=float)
    sim = np.array(sim, dtype=float)
    
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        # If observed values are constant, NSE is not well-defined.
        # If sim matches obs exactly, we can return 1.0 as an ideal match.
        # Otherwise, we might return -inf or 0.0. Here we choose -inf to indicate poor fit.
        if np.allclose(obs, sim):
            return 1.0
        else:
            return float('-inf')
    return 1 - (numerator / denominator)

def evaluate_nse_for_file(filepath, leadtime, num_test_days=None):
    """
    Evaluate NSE for a given file and leadtime using a naive forecast.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    leadtime : int
        Forecast horizon (in days) for the naive forecast.
    num_test_days : int or None
        Number of days at the end of the dataset to use for evaluation. If None, use the entire dataset (minus leadtime).
    
    Returns
    -------
    float
        NSE for the given file and leadtime.
    """
    df = pd.read_csv(filepath)
    # Ensure Q column exists
    if 'Q' not in df.columns:
        raise ValueError(f"File {filepath} does not contain 'Q' column.")
    
    Q = df['Q'].values
    total_length = len(Q)
    
    if total_length <= leadtime:
        # Not enough data to do a forecast at this lead time
        return float('nan')
    
    obs_full = Q[leadtime:]
    sim_full = Q[:-leadtime]
    
    # If num_test_days is specified, use only the last num_test_days entries
    if num_test_days is not None:
        if num_test_days > len(obs_full):
            print(f"Warning: num_test_days={num_test_days} is larger than available points {len(obs_full)}. Using entire set.")
        else:
            obs_full = obs_full[-num_test_days:]
            sim_full = sim_full[-num_test_days:]
    
    # Compute NSE
    nse = compute_nse(obs_full, sim_full)
    return nse

def main(data_dir='data', num_test_days=None, output_file='results.pkl'):
    """
    Main function to process all CSV files in the specified directory and produce the desired dictionary.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing CSV files.
    num_test_days : int or None
        Number of trailing rows to use for evaluation.
    output_file : str
        Name of the pickle file to save results to.
    
    Returns
    -------
    dict
        Nested dictionary of NSE values as described.
    """
    # Define the lead times we want to evaluate
    leadtimes = [1, 3, 5, 7, 14]
    
    # Initialize the output dictionary
    results = {f'leadtime_{lt}': {} for lt in leadtimes}
    
    # Iterate over all files in data_dir
    for filename in os.listdir(data_dir):
        if filename.endswith('_camels.csv') and len(filename.split('_')[0]) == 8:
            filepath = os.path.join(data_dir, filename)
            station_id = filename.split('_')[0]  # Extract the 8-digit station ID
            
            for lt in leadtimes:
                nse_value = evaluate_nse_for_file(filepath, lt, num_test_days=num_test_days)
                results[f'leadtime_{lt}'][station_id] = nse_value
    
    # Save results to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == '__main__':
    # Example usage:
    # To use the entire dataset:
    # results = main(data_dir='data', num_test_days=None, output_file='results.pkl')
    # To use a subset of the data:
    # results = main(data_dir='data', num_test_days=100, output_file='results.pkl')
    
    results = main(data_dir='data', num_test_days=None, output_file='results.pkl')
    print("Results saved to results.pkl.")
