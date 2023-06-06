import os
import argparse
from utils import *
from lstm import *
from keras import backend as K

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(
        description='Grid search for LSTM hyperparameters')

    parser_.add_argument('--csv_file', dest='csv_file', type=str)

    args_ = parser_.parse_args()
    print(args_)
    kwag_ = vars(args_)
    csv_file_ = kwag_['csv_file']

    # Remove extra characters
    if csv_file_.endswith('\r'):
        csv_file_ = csv_file_[:-1]

    # Define the grid search parameters
    possible_dropout_rates = [0, 0.5]
    possible_lstm_units = [32, 64, 128, 256]
    possible_lookbacks = [30, 60, 120, 270]

    csv_filepath = os.path.join("data", csv_file_)
    save_path = "/home/jswyou/scratch/results/vanilla_lstm"
    save_path = os.path.join(save_path, get_csv_filename_without_extension(csv_filepath))
    create_directory_if_not_exists(save_path)

    forecast_horizons = [1]

    for forecast_horizon in forecast_horizons:
        for dropout_rate in possible_dropout_rates:
            for lstm_units in possible_lstm_units:
                for lookback in possible_lookbacks:
                    vanilla_lstm_forecast_grid_search(
                        csv_filepath, 
                        forecast_horizon, 
                        save_path, 
                        dropout_rate=dropout_rate, 
                        lstm_units=lstm_units, 
                        lookback=lookback)
                    K.clear_session()
