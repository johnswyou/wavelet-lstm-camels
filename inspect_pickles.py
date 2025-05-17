import pickle
import pandas as pd # Often used with scaling/wavelets
# Add any other imports that might be necessary if you encounter ModuleNotFoundError

def inspect_pickle(file_path):
    print(f"Inspecting {file_path}...")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"  Type of loaded data: {type(data)}")

        if isinstance(data, dict):
            print(f"  It's a dictionary with {len(data)} keys.")
            print(f"  Keys: {list(data.keys())}")
            # Inspect the first item
            if data:
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                print(f"  Value for key '{first_key}':")
                print(f"    Type: {type(first_value)}")
                if hasattr(first_value, 'shape'):
                    print(f"    Shape: {first_value.shape}")
                elif isinstance(first_value, (list, tuple, dict)):
                        print(f"    Length/Keys: {len(first_value) if not isinstance(first_value, dict) else list(first_value.keys())}")
                else:
                    # Try to print a small part of it
                    print(f"    Value (partial): {str(first_value)[:100]}")

        elif isinstance(data, list):
            print(f"  It's a list with {len(data)} elements.")
            if data:
                first_element = data[0]
                print(f"  First element:")
                print(f"    Type: {type(first_element)}")
                if hasattr(first_element, 'shape'):
                    print(f"    Shape: {first_element.shape}")
                # Add more detailed inspection for list elements if needed

        else:
            print(f"  Data (partial): {str(data)[:200]}")

    except ModuleNotFoundError as e:
        print(f"  Error: A module was not found. You might need to install it or ensure it's in your PYTHONPATH.")
        print(f"  Details: {e}")
    except Exception as e:
        print(f"  Error loading or inspecting {file_path}: {e}")
    print("-" * 30)

inspect_pickle("filters/wavelet_dict.pkl")
inspect_pickle("filters/scaling_dict.pkl")