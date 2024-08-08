# InfluxDB Data Processing Script

## Overview

This script is designed to interact with InfluxDB to fetch, process, and manage data related to space weather measurements from instruments such as ACE and DSCOVR. The script is structured to handle data extraction, directory management, data processing, and analysis with the ultimate goal of producing feature vectors and cleaned datasets.

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- influxdb_client
- shutil
- os
- datetime

## Functions

### `get_earliest_data_point(influxdb_client, bucket, measurement)`

Retrieves the earliest data point from a specified bucket and measurement in InfluxDB. This is useful for determining the range of data available and ensuring that requests for data fall within the available range.

**Parameters:**
- `influxdb_client`: Configured client instance for accessing InfluxDB.
- `bucket`: The name of the bucket.
- `measurement`: The specific measurement within the bucket.

**Returns:** The timestamp of the earliest data point.

### `format_datetime_for_filename(dt_str)`

Converts an ISO 8601 datetime string to a format suitable for filenames, eliminating characters that are not valid in file paths.

**Parameters:**
- `dt_str`: Datetime string in ISO 8601 format.

**Returns:** Formatted datetime string suitable for filenames.

### `create_directory_structure()`

Sets up the required directory structure for saving dataframes and feature vectors into categorized folders. This structure includes main directories for each instrument and subdirectories for various data types.

### `fetch_data(bucket, measurement, start, stop)`

Queries InfluxDB to fetch data within a specified time range for a given bucket and measurement. This function pivots the data for better handling in pandas.

### `process_dataframe(df, output_path)`

Processes a dataframe by cleaning, restructuring, and finally saving it as a pickle file at the specified path.

### `load_and_display_pkl(file_path)`

Loads and displays the content of a pickle file, which is particularly useful for verification and quick checks.

### `calculate_and_merge_dipole(df)`

Calculates the dipole tilt and merges it with the input dataframe, enhancing the dataset with additional geophysical parameters.

### `process_instrument_feature_vector(df)`

Processes the final instrument data to compute various feature vectors, preparing the dataset for further analysis or machine learning applications.

### `complete_data_process(influxdb_client, instrument, start, stop)`

Coordinates the overall data processing steps from fetching data, processing it, and handling file operations based on the given time range and instrument. This function is central to ensuring data integrity and readiness for analysis.

## Earliest Available Data

- **ACE Bucket**: Earliest record timestamp for `ace_data`: 2001-08-07 00:00:00+00:00
- **DSCOVR Bucket**: Earliest record timestamp for `dscovr_data`: 2016-07-26 00:00:00+00:00
- **Indices Bucket**: Earliest record timestamp for `solar_indices`: 2001-08-10 00:00:00+00:00

## Usage

To use this script, ensure that all dependencies are installed and the InfluxDB client is correctly configured. The functions can be called within a Python environment, scripts, or Jupyter notebooks to fetch, process, and analyze data according to the needs of your project.
