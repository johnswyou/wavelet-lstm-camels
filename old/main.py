import os
import argparse
from utils import *
from lstm import *
from keras import backend as K

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(
        description='Brute force wavelet search with LSTM')
    
    parser_.add_argument('--csv_file', dest='csv_file', type = str)

    args_ = parser_.parse_args()
    print(args_)
    kwag_ = vars(args_)
    csv_file_ = kwag_['csv_file']

    # For some reason, when reading a line from canopex_ids.txt using sed in linux, there is \r added to the end
    if csv_file_.endswith('\r'):
        csv_file_ = csv_file_[:-1]

    # -----------------------
    # Define fixed parameters
    # -----------------------

    # 33 filters, all less than or equal to length of 14
    possible_filters = ["bl7", "coif1", "coif2", "db1", "db2", "db3", "db4", "db5", 
    "db6", "db7", "fk4", "fk6", "fk8", "fk14", "han2_3", "han3_3",
    "han4_5", "han5_5", "mb4_2", "mb8_2", "mb8_3", "mb8_4", "mb10_3", "mb12_3",
    "mb14_3", "sym4", "sym5", "sym6", "sym7", "la8", 
    "la10", "la12", "la14"]

    csv_filepath = os.path.join("data", csv_file_)
    save_path = "/home/jswyou/scratch/results/wavelet"
    save_path = os.path.join(save_path, get_csv_filename_without_extension(csv_filepath))
    create_directory_if_not_exists(save_path)
    
    forecast_horizons = [1]

    # -----------------
    # Commence training
    # -----------------

    for forecast_horizon in forecast_horizons:

        for wavelet_filter in possible_filters:

            wavelet_lstm_forecast(csv_filepath, forecast_horizon, wavelet_filter, save_path)
            K.clear_session()
