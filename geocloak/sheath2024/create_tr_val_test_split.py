import os
import pandas as pd

class DataProcessor:
    """
    A class to handle the processing of CSV data from a local directory.

    Attributes:
        directory (str): The directory path where CSV files are located.
        column_names (list): List of column names from the merged DataFrame.
    """

    def __init__(self, directory):
        """
        Initializes the DataProcessor with the specified directory.

        Args:
            directory (str): The directory path where CSV files are located.
        """
        self.directory = directory

    def read_and_merge_csvs(self):
        """
        Reads and merges CSV files from the specified directory.

        Returns:
            pd.DataFrame: A DataFrame containing the merged data from all CSV files.
        """
        csv_files = [f for f in os.listdir(self.directory) if f.endswith(".csv")]

        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_csv(file_path, low_memory=False)
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
        Saves the training, validation, and test sets to CSV files in the specified directory.

        Args:
            train_set (pd.DataFrame): The training data to be saved.
            val_set (pd.DataFrame): The validation data to be saved.
            test_set (pd.DataFrame): The test data to be saved.
            output_directory (str): The directory where the CSV files will be saved.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define the paths for the files in the directory
        train_path = os.path.join(output_directory, "train_set.csv")
        val_path = os.path.join(output_directory, "val_set.csv")
        test_path = os.path.join(output_directory, "test_set.csv")

        # Save DataFrames to CSV files
        train_set.to_csv(train_path, index=False)
        val_set.to_csv(val_path, index=False)
        test_set.to_csv(test_path, index=False)


def main():
    directory = "/home/chetrajpandey/data/formatted_data/sheath_train/"
    output_directory = "/home/chetrajpandey/data/formatted_data/sheath_splits"
    test_intervals = [
        ("2011-08-04", "2011-08-08"),
        ("2012-07-21", "2012-07-25"),
        ("2013-03-13", "2013-03-17"),
    ]
    val_years_months = {2011: (10, 12), 2012: (4, 6), 2013: (10, 12)}
    # val_years_months = {2011: (10, 12), 2012: (4, 6), 2013: None}  # Example of specifying whole year 2013

    # Initialize the DataProcessor
    processor = DataProcessor(directory)

    # Read and merge CSV files from the directory
    data = processor.read_and_merge_csvs()

    # Create training, validation, and test splits
    train_set, val_set, test_set = processor.create_splits(
        data, test_intervals, val_years_months
    )

    # Save the splits to the local directory
    processor.save_splits(train_set, val_set, test_set, output_directory)


if __name__ == "__main__":
    main()
