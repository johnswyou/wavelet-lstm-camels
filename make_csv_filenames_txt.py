import os

def write_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)

# Example usage:
my_list = os.listdir("data")
write_list_to_file(my_list, 'csv_filenames.txt')
