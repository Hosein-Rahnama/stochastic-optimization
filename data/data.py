# This script calculates some properties of the data set.

import sys


# Read the data set file in svm light format and extract some information.
def svm_light_info(file_name):
    num_samples = 0
    max_squared_norm = 0.0
    max_feature_index = 0
    total_none_zero = 0

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            num_samples += 1

            # Skip label and get the domain point.
            x = line.split()[1:] 
            squared_norm = 0.0
            for pair in x:
                if ":" not in pair:
                    continue
                index, value = pair.split(":")
                index = int(index)
                value = float(value)
                squared_norm += value ** 2
                max_feature_index = max(max_feature_index, index)
                total_none_zero += 1

            max_squared_norm = max(max_squared_norm, squared_norm)

    headers = ["number-of-samples", "maximum-feature-index", "number-of-non-zero-enteries", "maximum-squared-norm"]
    values = [num_samples, max_feature_index, total_none_zero, max_squared_norm]
    return dict(zip(headers, values))

# These functions help format a table consisting of data set information.
def format_cell(cell, width):
    if isinstance(cell, float):
        return f"{cell:>{width}.2f}"
    else:
        return f"{str(cell):>{width}}"
    
def format_row(row, widths):
    return " ".join(format_cell(cell, w) for cell, w in zip(row, widths))


if __name__ == "__main__":
    # Get the data set name from command line.
    DAT_DIR = "../data"
    DATA_SET = sys.argv[1]

    # Process the data set files and extract information.
    train_data = f"{DAT_DIR}/{DATA_SET}/train.dat"
    train_info = svm_light_info(train_data)

    test_data = f"{DAT_DIR}/{DATA_SET}/test.dat"
    test_info = svm_light_info(test_data)

    dim = max(train_info["maximum-feature-index"], test_info["maximum-feature-index"])
    max_squared_norm = max(train_info["maximum-squared-norm"], test_info["maximum-squared-norm"])
    num_all_samples = train_info["number-of-samples"] + test_info["number-of-samples"]
    data_set_size = num_all_samples * dim
    num_none_zero_enteries = train_info["number-of-non-zero-enteries"] + test_info["number-of-non-zero-enteries"]
    sparsity = (1 - (num_none_zero_enteries / data_set_size)) * 100

    # Save the extracted information.
    headers = ["training-samples", "testing-samples", "all-samples", "features", "sparsity", "maximum-squared-norm"]
    values = [train_info["number-of-samples"], test_info["number-of-samples"], num_all_samples, dim, sparsity, max_squared_norm]

    column_widths = [len(h) for h in headers]
    headers = format_row(headers, column_widths)
    values = format_row(values, column_widths)

    with open(f"{DAT_DIR}/{DATA_SET}/info.txt", "w") as f:
        f.write(headers + "\n")
        f.write(values + "\n")
