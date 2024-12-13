import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        file = pickle.load(f)

    return file


def get_csv_filename_without_extension(file_path):
    """
    Given a file path, returns the base file name without the '.csv' extension.

    Args:
        file_path (str): The full path to the file.

    Returns:
        str: The base file name without the '.csv' extension.

    Raises:
        ValueError: If the file path does not point to a CSV file.
    """
    file_name = os.path.basename(file_path)
    if file_name.endswith('.csv'):
        return file_name[:-4]  # drop the last four characters, which are ".csv"
    else:
        raise ValueError('File path does not point to a CSV file.')


def create_directory_if_not_exists(directory_path):
    """
    Creates a directory at the given path if it does not already exist.

    Args:
        directory_path (str): The full path to the directory to be created.

    Prints:
        str: A message indicating whether the directory was created or if it already existed.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory {directory_path} created successfully")
    else:
        print(f"Directory {directory_path} already exists")


def check_csv_order_and_continuity(folder_path):
    """
    Checks each CSV file in the specified folder to ensure that:
    1. The rows are ordered in ascending order by 'date'.
    2. There are no missing days between the first and last dates.

    Parameters:
    - folder_path (str): Path to the folder containing the CSV files.

    Returns:
    - bool: True if all files pass the checks, False otherwise.
    """
    # Define the file pattern
    pattern = os.path.join(folder_path, '*_camels.csv')
    files = glob.glob(pattern)
    
    if not files:
        print("No files matching the pattern were found.")
        return False

    all_passed = True  # Flag to track overall status

    for file in files:
        try:
            # Read the CSV file, parse 'date' column as datetime
            df = pd.read_csv(file, parse_dates=['date'])
            
            # Check if 'date' column is sorted in ascending order
            if not df['date'].is_monotonic_increasing:
                print(f"❌ {os.path.basename(file)} is not ordered by ascending dates.")
                all_passed = False
                continue  # Skip further checks for this file
            
            # Generate a date range from the first to the last date
            expected_dates = pd.date_range(start=df['date'].iloc[0],
                                           end=df['date'].iloc[-1],
                                           freq='D')
            
            # Check if the number of dates matches
            if len(expected_dates) != len(df):
                # Identify missing dates
                missing_dates = expected_dates.difference(df['date'])
                missing_dates_str = ', '.join(missing_dates.strftime('%Y-%m-%d').tolist())
                print(f"❌ {os.path.basename(file)} has missing dates: {missing_dates_str}")
                all_passed = False
                continue  # Skip to next file
            
            # If both checks pass
            print(f"✅ {os.path.basename(file)} is properly ordered with no missing dates.")
        
        except Exception as e:
            print(f"⚠️  An error occurred while processing {os.path.basename(file)}: {e}")
            all_passed = False

    if all_passed:
        print("\nAll files are ordered correctly with no missing dates.")
    else:
        print("\nSome files failed the checks. Please review the messages above.")

    return all_passed

# Example usage:
# folder = "/path/to/your/csv/folder"
# check_csv_order_and_continuity(folder)


def plot_wavelet_stacks(df, original_feature, max_level, plot_first_n=None):
    """
    Plots a vertical stack of time series:
    1. The original feature on top
    2. The scaling coefficients (V) at max_level
    3. The detail (wavelet) coefficients (W) from max_level down to 1

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original feature and its wavelet decomposition columns.
        Expected columns:
            - original_feature
            - original_feature_V{max_level}
            - original_feature_W{level} for levels in [1, ..., max_level]
    original_feature : str
        The name of the original feature column.
    max_level : int
        The maximum decomposition level.
    plot_first_n : int or None
        If provided, only plot the first n values for each series. Otherwise, plot all.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    # Construct the column names
    scaling_col = f"{original_feature}_V{max_level}"
    wavelet_cols = [f"{original_feature}_W{i}" for i in range(max_level, 0, -1)]

    # Determine the slice if plot_first_n is given
    if plot_first_n is not None:
        df_slice = df.iloc[:plot_first_n]
    else:
        df_slice = df

    # Prepare the figure and axes
    n_plots = max_level + 2  # original + scaling + wavelets
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2*n_plots), sharex=True)
    
    # Plot the original feature
    axes[0].plot(df_slice.index, df_slice[original_feature], color='black')
    axes[0].set_title(f"Original: {original_feature}")
    axes[0].grid(True)
    
    # Plot the scaling coefficients at max level
    axes[1].plot(df_slice.index, df_slice[scaling_col], color='blue')
    axes[1].set_title(f"Scaling Coefficients: {scaling_col}")
    axes[1].grid(True)
    
    # Plot the wavelet detail coefficients from top (max_level) to bottom (1)
    for i, w_col in enumerate(wavelet_cols, start=2):
        axes[i].plot(df_slice.index, df_slice[w_col], color='red')
        axes[i].set_title(f"Wavelet Detail Coefficients: {w_col}")
        axes[i].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    return fig, axes
