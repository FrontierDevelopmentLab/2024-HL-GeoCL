from io import StringIO

import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    A class to handle the processing of CSV data from Google Cloud Storage.

    Attributes:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        directory (str): The directory path within the bucket where CSV files are located.
        client (google.cloud.storage.Client): Google Cloud Storage client instance.
        column_names (list): List of column names from the merged DataFrame.
    """

    def __init__(self, bucket_name, directory):
        """
        Initializes the DataProcessor with the specified bucket name and directory.

        Args:
            bucket_name (str): The name of the Google Cloud Storage bucket.
            directory (str): The directory path within the bucket where CSV files are located.
        """
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.directory = directory

    def read_and_merge_csvs(self):
        """
        Reads and merges CSV files from the specified Google Cloud Storage bucket.

        Returns:
            pd.DataFrame: A DataFrame containing the merged data from all CSV files.
        """
        bucket = self.client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.directory)
        csv_files = [blob.name for blob in blobs if blob.name.endswith(".csv")]

        dfs = []
        for file in csv_files:
            blob = bucket.blob(file)
            content = blob.download_as_text()
            df = pd.read_csv(StringIO(content), low_memory=False)
            dfs.append(df)

        merged_df = pd.concat(dfs)
        self.column_names = merged_df.columns.tolist()
        return merged_df

    def create_splits(self, data, test_intervals, val_years_months):
        """
        Creates training, validation, and test sets from the given data based on specified intervals,
        ensuring that test and validation sets do not overlap.
        """
        initial_timestamp_col = self.column_names[0]
        data[initial_timestamp_col] = pd.to_datetime(
            data[initial_timestamp_col], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
        )

        # Initialize boolean Series for test and validation indices
        test_indices = pd.Series([False] * len(data), index=data.index)
        val_indices = pd.Series([False] * len(data), index=data.index)

        # Create test set mask
        for start, end in test_intervals:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            test_indices = test_indices | (
                (data[initial_timestamp_col] >= start_date)
                & (data[initial_timestamp_col] <= end_date)
            )

        test_set = data[test_indices]
        print("Test set length:", len(test_set))

        # Create validation set mask
        for key, value in val_years_months.items():
            if isinstance(value, tuple):
                start_month, end_month = value
                val_indices = val_indices | (
                    (data[initial_timestamp_col].dt.year == key)
                    & (data[initial_timestamp_col].dt.month >= start_month)
                    & (data[initial_timestamp_col].dt.month <= end_month)
                )
            else:
                val_indices = val_indices | (data[initial_timestamp_col].dt.year == key)

        # Ensure validation set does not overlap with test set
        val_indices = val_indices & ~test_indices
        val_set = data[val_indices]
        print("Validation set length:", len(val_set))

        # Create train set
        train_set = data[~test_indices & ~val_indices]
        print("Training set length:", len(train_set))

        return train_set, val_set, test_set

    def save_splits(self, train_set, val_set, test_set, output_directory):
        """
        Uploads the training, validation, and test sets to CSV files in the specified GCS directory.

        Args:
            train_set (pd.DataFrame): The training data to be saved.
            val_set (pd.DataFrame): The validation data to be saved.
            test_set (pd.DataFrame): The test data to be saved.
            output_directory (str): The GCS directory where the CSV files will be saved.
        """
        bucket = self.client.get_bucket(self.bucket_name)

        # Define the paths for the files in the bucket
        train_path = f"{output_directory}/train_set.csv"
        val_path = f"{output_directory}/val_set.csv"
        test_path = f"{output_directory}/test_set.csv"

        # Convert DataFrames to CSV strings
        train_csv = train_set.to_csv(index=False)
        val_csv = val_set.to_csv(index=False)
        test_csv = test_set.to_csv(index=False)

        # Upload the CSV strings to GCS
        bucket.blob(train_path).upload_from_string(train_csv, content_type="text/csv")
        bucket.blob(val_path).upload_from_string(val_csv, content_type="text/csv")
        bucket.blob(test_path).upload_from_string(test_csv, content_type="text/csv")


# Specifying the GCP bucket and folders
bucket_name = "geocloak2024"
directory = "formatted_data/sheath_train/"
output_directory = "formatted_data/sheath_splits"
test_intervals = [
    ("2011-08-04", "2011-08-08"),
    ("2012-07-21", "2012-07-25"),
    ("2013-03-13", "2013-03-17"),
]
val_years_months = {2011: (10, 12), 2012: (4, 6), 2013: (10, 12)}
# val_years_months = {2011: (10, 12), 2012: (4, 6), 2013: None}  # Example of specifying whole year 2013

# Initialize the DataProcessor
processor = DataProcessor(bucket_name, directory)

# Read and merge CSV files from the bucket
data = processor.read_and_merge_csvs()

# Create training, validation, and test splits
train_set, val_set, test_set = processor.create_splits(
    data, test_intervals, val_years_months
)

# Save the splits to the local directory
processor.save_splits(train_set, val_set, test_set, output_directory)
