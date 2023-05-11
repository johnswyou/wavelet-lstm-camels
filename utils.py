import os

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
        os.mkdir(directory_path)
        print(f"Directory {directory_path} created successfully")
    else:
        print(f"Directory {directory_path} already exists")