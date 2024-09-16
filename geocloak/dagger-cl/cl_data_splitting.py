import os
import pandas as pd

def filter_data(df, start_date, end_date):
    """
    Filter the DataFrame for rows with 'Time' within the specified date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Time' column to filter.
    start_date : str
        Start date for the filter in 'YYYY-MM-DD' format.
    end_date : str
        End date for the filter in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with 'Time' within the specified date range.
    """
    mask = (df['Time'] >= start_date) & (df['Time'] <= end_date)
    return df.loc[mask]

def save_sample(df, file_path):
    """
    Save the DataFrame to a CSV file after sorting it by the 'Time' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    file_path : str
        Path where the CSV file will be saved.
    """
    df_sorted = df.sort_values(by='Time')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_sorted.to_csv(file_path, index=False)

def create_samples(df, output_folder, filename, periods, random_seeds, sampling_fractions):
    """
    Create and save samples of the DataFrame based on specified time periods, sampling fractions, and random seeds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to sample from.
    output_folder : str
        Folder path where the sampled CSV files will be saved.
    filename : str
        Base filename for the saved CSV files.
    periods : list of tuple
        List of (start_date, end_date) tuples defining the time periods.
    random_seeds : list of int
        List of random seeds for reproducibility of sampling.
    sampling_fractions : list of float
        List of fractions for sampling the previous sample before concatenating with new data.
    """
    samples = []
    for i, (start_date, end_date) in enumerate(periods):
        df_filtered = filter_data(df, start_date, end_date)
        
        if i > 0:
            previous_sample = samples[-1].sample(frac=sampling_fractions[i-1], random_state=random_seeds[i-1])
            sample = pd.concat([previous_sample, df_filtered])
        else:
            sample = df_filtered
        
        file_path = os.path.join(output_folder, f'{filename}_{i+1}.csv')
        save_sample(sample, file_path)
        samples.append(sample)

def main(input_csv, output_folder, filename, periods, random_seeds, sampling_fractions):
    """
    Load the data from a CSV file and create samples based on specified parameters.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file containing the data.
    output_folder : str
        Folder path where the sampled CSV files will be saved.
    filename : str
        Base filename for the saved CSV files.
    periods : list of tuple
        List of (start_date, end_date) tuples defining the time periods.
    random_seeds : list of int
        List of random seeds for reproducibility of sampling.
    sampling_fractions : list of float
        List of fractions for sampling the previous sample before concatenating with new data.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv, parse_dates=['Time'])
    
    # Create and save samples
    create_samples(df, output_folder, filename, periods, random_seeds, sampling_fractions)

if __name__ == '__main__':
    random_seeds = [10, 20]
    sampling_fractions = [0.4, 0.4]
    
    periods_ace = [
        ('2001-01-01', '2019-12-31'),
        ('2020-01-01', '2021-12-31'),
        ('2022-01-01', '2023-12-31')
    ]
    
    periods_dscovr = [
        ('2016-01-01', '2019-12-31'),
        ('2020-01-01', '2021-12-31'),
        ('2022-01-01', '2023-12-31')
    ]
    
    main(
        input_csv='/home/jupyter/ace_processed/ace_mapping_train.csv',
        output_folder='/home/chetrajpandey/data',
        filename='ace_mapping_sample',
        periods=periods_ace,
        random_seeds=random_seeds,
        sampling_fractions=sampling_fractions
    )
    
    main(
        input_csv='/home/jupyter/dscovr_processed/dscovr_mapping_train.csv',
        output_folder='/home/chetrajpandey/data',
        filename='dscovr_mapping_sample',
        periods=periods_dscovr,
        random_seeds=random_seeds,
        sampling_fractions=sampling_fractions
    )
