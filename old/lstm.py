import joblib
import os
import numpy as np
import pandas as pd
import time
from utils import *
from metrics import *
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

def create_dataset(X, y, lookback=270, forecast_horizon=1):
    Xs, ys = [], []
    for i in range(len(X) - lookback - forecast_horizon + 1):
        v = X.iloc[i:(i + lookback)].values
        Xs.append(v)
        ys.append(y.iloc[i + lookback + forecast_horizon - 1])
    return np.array(Xs), np.array(ys)

# *************
# Wavelet LSTM
# *************

pandas2ri.activate()
r = robjects.r
r.source('modwt_ea_processor.R')
r_modwt_ea_processor = robjects.globalenv['modwt_ea_processor']

def wavelet_lstm_forecast(csv_filepath, forecast_horizon, wavelet_filter, save_path):

    # NOTE: csv_filepath should be the full or relative path to the CSV file

    # Fixed parameters
    target_var = "streamflow"
    lookback = 270
    filename_suffix = f'{get_csv_filename_without_extension(csv_filepath)}_LT_{forecast_horizon}_{wavelet_filter}'

    # Perform EA IVS
    print("Performing MODWT and EA IVS ...")
    start_time = time.time()  # save the current time
    r_train_val_test = r_modwt_ea_processor(csv_filepath, wavelet_filter, forecast_horizon)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference
    print("Done MODWT and EA IVS.")

    joblib.dump(elapsed_time, os.path.join(save_path, f'MODWT_EA_time_{filename_suffix}.pkl'))

    # Read CSV file, add MODWT coefficients, performs IVS, remove boundary coefficients, remove date column
    with robjects.default_converter + pandas2ri.converter:
        train = robjects.conversion.get_conversion().rpy2py(r_train_val_test[0])
    with robjects.default_converter + pandas2ri.converter:
        val = robjects.conversion.get_conversion().rpy2py(r_train_val_test[1])
    with robjects.default_converter + pandas2ri.converter:
        test = robjects.conversion.get_conversion().rpy2py(r_train_val_test[2])

    train_target, val_target, test_target = train[target_var], val[target_var], test[target_var]
    train, val, test = train.drop([target_var], axis=1), val.drop([target_var], axis=1), test.drop([target_var], axis=1)

    assert list(train) == list(val) == list(test)

    print(f'Selected inputs: {list(train)}')

    # Save selected input variable names
    joblib.dump(list(train), os.path.join(save_path, f'selected_inputs_{filename_suffix}.pkl'))

    # Standardize the features using the training set
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(save_path, f'scaler_{filename_suffix}.pkl'))

    # Create the LSTM datasets
    X_train, y_train = create_dataset(pd.DataFrame(train_scaled), train_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_val, y_val = create_dataset(pd.DataFrame(val_scaled), val_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_test, y_test = create_dataset(pd.DataFrame(test_scaled), test_target, lookback=lookback, forecast_horizon=forecast_horizon)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    # Compile and fit the LSTM model
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print('Training LSTM ...')
    start_time = time.time()  # save the current time
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2, shuffle=False)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference    
    print('Done training LSTM.')

    joblib.dump(elapsed_time, os.path.join(save_path, f'LSTM_time_{filename_suffix}.pkl'))

    # Save the model
    model.save(os.path.join(save_path, f'lstm_model_{filename_suffix}.h5'))

    # Make predictions
    val_predictions = model.predict(X_val).squeeze()
    test_predictions = model.predict(X_test).squeeze()

    # Compute the evaluation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    val_mae = mean_absolute_error(y_val, val_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    val_mape = mean_absolute_percentage_error(y_val, val_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)

    val_mase = mean_absolute_scaled_error(y_val, val_predictions)
    test_mase = mean_absolute_scaled_error(y_test, test_predictions)

    val_nse = nash_sutcliffe_efficiency(y_val, val_predictions)
    test_nse = nash_sutcliffe_efficiency(y_test, test_predictions)

    val_kge = kling_gupta_efficiency(y_val, val_predictions)
    test_kge = kling_gupta_efficiency(y_test, test_predictions)

    metrics = {
        'validation': {
            'RMSE': val_rmse,
            'MAE': val_mae,
            'MAPE': val_mape,
            'MASE': val_mase,
            'NSE': val_nse,
            'KGE': val_kge,
        },
        'test': {
            'RMSE': test_rmse,
            'MAE': test_mae,
            'MAPE': test_mape,
            'MASE': test_mase,
            'NSE': test_nse,
            'KGE': test_kge,
        }
    }

    # Save the predictions and metrics
    np.save(os.path.join(save_path, f'val_predictions_{filename_suffix}.npy'), val_predictions)
    np.save(os.path.join(save_path, f'test_predictions_{filename_suffix}.npy'), test_predictions)
    joblib.dump(metrics, os.path.join(save_path, f'metrics_{filename_suffix}.pkl'))

    # return model, val_predictions, test_predictions, metrics

# *************
# Vanilla LSTM
# *************

r_ea_processor = robjects.globalenv['ea_processor']

def vanilla_lstm_forecast(csv_filepath, forecast_horizon, save_path):

    # NOTE: csv_filepath should be the full or relative path to the CSV file

    # Fixed parameters
    target_var = "streamflow"
    lookback = 270
    filename_suffix = f'{get_csv_filename_without_extension(csv_filepath)}_LT_{forecast_horizon}'

    # Perform EA IVS
    print("Performing EA IVS ...")
    start_time = time.time()  # save the current time
    r_train_val_test = r_ea_processor(csv_filepath, forecast_horizon)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference
    print("Done EA IVS.")

    joblib.dump(elapsed_time, os.path.join(save_path, f'EA_time_{filename_suffix}.pkl'))

    # Read CSV file, performs IVS, remove boundary coefficients, remove date column
    with robjects.default_converter + pandas2ri.converter:
        train = robjects.conversion.get_conversion().rpy2py(r_train_val_test[0])
    with robjects.default_converter + pandas2ri.converter:
        val = robjects.conversion.get_conversion().rpy2py(r_train_val_test[1])
    with robjects.default_converter + pandas2ri.converter:
        test = robjects.conversion.get_conversion().rpy2py(r_train_val_test[2])

    train_target, val_target, test_target = train[target_var], val[target_var], test[target_var]
    train, val, test = train.drop([target_var], axis=1), val.drop([target_var], axis=1), test.drop([target_var], axis=1)

    assert list(train) == list(val) == list(test)

    print(f'Selected inputs: {list(train)}')

    # Save selected input variable names
    joblib.dump(list(train), os.path.join(save_path, f'selected_inputs_{filename_suffix}.pkl'))

    # Standardize the features using the training set
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(save_path, f'scaler_{filename_suffix}.pkl'))

    # Create the LSTM datasets
    X_train, y_train = create_dataset(pd.DataFrame(train_scaled), train_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_val, y_val = create_dataset(pd.DataFrame(val_scaled), val_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_test, y_test = create_dataset(pd.DataFrame(test_scaled), test_target, lookback=lookback, forecast_horizon=forecast_horizon)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    # Compile and fit the LSTM model
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print('Training LSTM ...')
    start_time = time.time()  # save the current time
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2, shuffle=False)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference    
    print('Done training LSTM.')

    joblib.dump(elapsed_time, os.path.join(save_path, f'LSTM_time_{filename_suffix}.pkl'))

    # Save the model
    model.save(os.path.join(save_path, f'lstm_model_{filename_suffix}.h5'))

    # Make predictions
    val_predictions = model.predict(X_val).squeeze()
    test_predictions = model.predict(X_test).squeeze()

    # Compute the evaluation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    val_mae = mean_absolute_error(y_val, val_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    val_mape = mean_absolute_percentage_error(y_val, val_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)

    val_mase = mean_absolute_scaled_error(y_val, val_predictions)
    test_mase = mean_absolute_scaled_error(y_test, test_predictions)

    val_nse = nash_sutcliffe_efficiency(y_val, val_predictions)
    test_nse = nash_sutcliffe_efficiency(y_test, test_predictions)

    val_kge = kling_gupta_efficiency(y_val, val_predictions)
    test_kge = kling_gupta_efficiency(y_test, test_predictions)

    metrics = {
        'validation': {
            'RMSE': val_rmse,
            'MAE': val_mae,
            'MAPE': val_mape,
            'MASE': val_mase,
            'NSE': val_nse,
            'KGE': val_kge,
        },
        'test': {
            'RMSE': test_rmse,
            'MAE': test_mae,
            'MAPE': test_mape,
            'MASE': test_mase,
            'NSE': test_nse,
            'KGE': test_kge,
        }
    }

    # Save the predictions and metrics
    np.save(os.path.join(save_path, f'val_predictions_{filename_suffix}.npy'), val_predictions)
    np.save(os.path.join(save_path, f'test_predictions_{filename_suffix}.npy'), test_predictions)
    joblib.dump(metrics, os.path.join(save_path, f'metrics_{filename_suffix}.pkl'))

# *****************************
# Vanilla LSTM With Grid Search
# *****************************

def vanilla_lstm_forecast_grid_search(csv_filepath, forecast_horizon, save_path, dropout_rate=0.4, lstm_units=256, lookback=270):
    
    # NOTE: csv_filepath should be the full or relative path to the CSV file

    # Fixed parameters
    target_var = "streamflow"
    filename_suffix = f'{get_csv_filename_without_extension(csv_filepath)}_LT_{forecast_horizon}_dropout_{dropout_rate}_units_{lstm_units}_lookback_{lookback}'

    # Perform EA IVS
    print("Performing EA IVS ...")
    start_time = time.time()  # save the current time
    r_train_val_test = r_ea_processor(csv_filepath, forecast_horizon)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference
    print("Done EA IVS.")

    joblib.dump(elapsed_time, os.path.join(save_path, f'EA_time_{filename_suffix}.pkl'))

    # Read CSV file, performs IVS, remove boundary coefficients, remove date column
    with robjects.default_converter + pandas2ri.converter:
        train = robjects.conversion.get_conversion().rpy2py(r_train_val_test[0])
    with robjects.default_converter + pandas2ri.converter:
        val = robjects.conversion.get_conversion().rpy2py(r_train_val_test[1])
    with robjects.default_converter + pandas2ri.converter:
        test = robjects.conversion.get_conversion().rpy2py(r_train_val_test[2])

    train_target, val_target, test_target = train[target_var], val[target_var], test[target_var]
    train, val, test = train.drop([target_var], axis=1), val.drop([target_var], axis=1), test.drop([target_var], axis=1)

    assert list(train) == list(val) == list(test)

    print(f'Selected inputs: {list(train)}')

    # Save selected input variable names
    joblib.dump(list(train), os.path.join(save_path, f'selected_inputs_{filename_suffix}.pkl'))

    # Standardize the features using the training set
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Save the scaler
    joblib.dump(scaler, os.path.join(save_path, f'scaler_{filename_suffix}.pkl'))

    # Create the LSTM datasets
    X_train, y_train = create_dataset(pd.DataFrame(train_scaled), train_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_val, y_val = create_dataset(pd.DataFrame(val_scaled), val_target, lookback=lookback, forecast_horizon=forecast_horizon)
    X_test, y_test = create_dataset(pd.DataFrame(test_scaled), test_target, lookback=lookback, forecast_horizon=forecast_horizon)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # Compile and fit the LSTM model
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print('Training LSTM ...')
    start_time = time.time()  # save the current time
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2, shuffle=False)
    end_time = time.time()
    elapsed_time = end_time - start_time  # calculate the difference    
    print('Done training LSTM.')

    joblib.dump(elapsed_time, os.path.join(save_path, f'LSTM_time_{filename_suffix}.pkl'))

    # Save the model
    model.save(os.path.join(save_path, f'lstm_model_{filename_suffix}.h5'))

    # Make predictions
    val_predictions = model.predict(X_val).squeeze()
    test_predictions = model.predict(X_test).squeeze()

    # Compute the evaluation metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    val_mae = mean_absolute_error(y_val, val_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    val_mape = mean_absolute_percentage_error(y_val, val_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)

    val_mase = mean_absolute_scaled_error(y_val, val_predictions)
    test_mase = mean_absolute_scaled_error(y_test, test_predictions)

    val_nse = nash_sutcliffe_efficiency(y_val, val_predictions)
    test_nse = nash_sutcliffe_efficiency(y_test, test_predictions)

    val_kge = kling_gupta_efficiency(y_val, val_predictions)
    test_kge = kling_gupta_efficiency(y_test, test_predictions)

    metrics = {
        'validation': {
            'RMSE': val_rmse,
            'MAE': val_mae,
            'MAPE': val_mape,
            'MASE': val_mase,
            'NSE': val_nse,
            'KGE': val_kge,
        },
        'test': {
            'RMSE': test_rmse,
            'MAE': test_mae,
            'MAPE': test_mape,
            'MASE': test_mase,
            'NSE': test_nse,
            'KGE': test_kge,
        }
    }

    # Save the predictions and metrics
    np.save(os.path.join(save_path, f'val_predictions_{filename_suffix}.npy'), val_predictions)
    np.save(os.path.join(save_path, f'test_predictions_{filename_suffix}.npy'), test_predictions)
    joblib.dump(metrics, os.path.join(save_path, f'metrics_{filename_suffix}.pkl'))
