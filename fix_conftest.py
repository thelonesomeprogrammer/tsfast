import os

with open("benchmarks/conftest.py", "r") as f:
    text = f.read()

# I am instructed by memory to add the new features to the fixture lists in benchmarks/conftest.py (like full_30_features)
# I added percentage_of_reoccurring_datapoints_to_all_datapoints, percentage_of_reoccurring_values_to_all_values, ratio_value_number_to_time_series_length

text = text.replace('"length", "variance_larger_than_standard_deviation"', '"length", "variance_larger_than_standard_deviation", "percentage_of_reoccurring_datapoints_to_all_datapoints", "percentage_of_reoccurring_values_to_all_values", "ratio_value_number_to_time_series_length"')
# and update name to full_33_features ? The user said to append it to full_30_features.

with open("benchmarks/conftest.py", "w") as f:
    f.write(text)
