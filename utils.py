import os
import glob
import pandas as pd

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
