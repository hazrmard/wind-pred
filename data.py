import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.special import kl_div


def read_file(dir_path='./', processed=False) -> pd.DataFrame:
    """
    Read wind data from a file and return a pandas DataFrame.

    Parameters:
    dir_path (str, optional): The directory path where the file is located. Defaults to './'.
    processed (bool, optional): Flag indicating whether the file has already been processed. Defaults to False.

    Returns:
    pd.DataFrame: A DataFrame containing the wind data.

    """
    if os.path.exists(dir_path+'ERDCSlow.hdf5') and processed:
        df = pd.read_hdf(dir_path+'ERDCSlow.hdf5', key='table', index_col='Time')
    else:
        df = pd.read_csv(dir_path + 'ERDCSlow.csv',
                        index_col='Time', parse_dates=['Time'])
        cols = []
        for col in df.columns:
            c = pd.to_numeric(df[col], errors='coerce', downcast='float')
            cols.append(c)
        df = pd.concat(cols, axis=1)

        df = df[['Speed', 'Direction', 'Temp']]
        df = df.dropna(axis=0, how='any')
        df.to_hdf(dir_path + '/ERDCSlow.hdf5', key='table')
    df['dx'] = np.cos(df.Direction)
    df['dy'] = np.sin(df.Direction)
    return df



def find_sequences(df, interval=pd.Timedelta(minutes=1), agg_fn='mean'):
    """
    Find sequences of non-null values in the given DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - interval (Timedelta, optional): The time interval for resampling. Defaults to 1 minute.
    - agg_fn (str or float, optional): The aggregation function to use. Can be 'mean', 'max', or a quantile value. Defaults to 'mean'.

    Returns:
    - list: A list of sequences, where each sequence is a DataFrame with non-null values and a frequency index equal to the specified interval.
    """
    resampler = df.resample(interval)
    if agg_fn=='mean':
        dfm = resampler.mean()
    elif agg_fn=='max':
        dfm = resampler.max()
    elif isinstance(agg_fn, float):
        dfm = resampler.quantile(agg_fn)
    i, ii = 0, dfm.index[0]
    j, jj = 1, dfm.index[1]
    seqs = []
    while j < len(dfm):
        while i < len(dfm):
            if not np.isnan(dfm.Speed.iloc[i]):
                break
            i += 1
        j = i+1
        while j <= len(dfm):
            if j==len(dfm) or np.isnan(dfm.Speed.iloc[j]):
                seqs.append(dfm.iloc[i:j])
                i = j+1
                break
            j += 1

    for s in seqs:
        s.index.freq = interval

    return [s for s in seqs if len(s)>=4]


def plot_seqs(seqs: List[pd.DataFrame]):
    plt.figure(figsize=(12,6))
    for s in seqs:
        plt.plot(s.Speed)
    plt.tight_layout()



def probabilistic_distance(a: np.ndarray, b: np.ndarray, measure='wasserstein'):
    """
    Compute the probabilistic distance between two arrays representing empirical
    samples from a probability distribution.

    Parameters:
    - a (np.ndarray): The first array.
    - b (np.ndarray): The second array.

    Returns:
    - float: The Wasserstein distance between the histograms of the two arrays.
    """
    if measure=='wasserstein':
        return wasserstein_distance(a, b)
    elif measure=='kl':
        # Compute the histograms of the arrays (probability densities)
        hist_a, bins_a = np.histogram(a, bins='auto', density=True)
        hist_b, bins_b = np.histogram(b, bins=bins_a, density=True)
        return kl_div(hist_a, hist_b).sum()   