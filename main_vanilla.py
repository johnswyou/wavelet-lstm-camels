import os
import argparse
from utils import *
from lstm import *
from keras import backend as K

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(
        description='Vanilla LSTM')
    
    parser_.add_argument('--leadtime', dest='leadtime', type = int)

    args_ = parser_.parse_args()
    print(args_)
    kwag_ = vars(args_)
    leadtime_ = kwag_['leadtime']

    with open('final_filenames.txt', 'r') as file:
        csv_filenames = [line.rstrip() for line in file.readlines()]

    # -----------------
    # Commence training
    # -----------------

    basin_counter = 0

    for csv_file_ in csv_filenames:

        basin_counter += 1
        print(f'Starting basin: {basin_counter}')

        csv_filepath = os.path.join("data", csv_file_)
        save_path = "/home/jswyou/scratch/results/vanilla"
        save_path = os.path.join(save_path, get_csv_filename_without_extension(csv_filepath))
        create_directory_if_not_exists(save_path)

        vanilla_lstm_forecast(csv_filepath, leadtime_, save_path)
        K.clear_session()
