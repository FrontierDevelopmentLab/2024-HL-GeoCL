import numpy as np


class AIA_MinMaxScaler:
    def __init__(self, minvalue=None, ptpvalue=None):
        self.minvalue = minvalue
        self.ptpvalue = ptpvalue

    def transform(self, data):
        assert (self.minvalue is not None) and (self.ptpvalue is not None)
        return (data - self.minvalue) / self.ptpvalue

    def fit(self, data):
        # Don't be dumb and give a 1D vector. Make it (1,N). Last channel is individual features.
        axis = range(len(data.shape))[:-1]
        self.minvalue = np.nanmin(data, axis=axis)
        self.ptpvalue = np.ptp(data, axis=axis)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        assert (self.minvalue is not None) and (self.ptpvalue is not None)
        return data * self.ptpvalue + self.minvalue


class AIA_StandardScaler:
    def __init__(self, mean=None, stddev=None):
        self.mean = mean
        self.stddev = stddev

    def transform(self, data):
        assert (self.mean is not None) and (self.stddev is not None)
        return (data - self.mean) / self.stddev

    def fit(self, data):
        # Don't be dumb and give a 1D vector. Make it (1,N). Last channel is individual features.
        axis = range(len(data.shape))[:-1]
        self.mean = np.nanmean(data, axis=axis)
        self.stddev = np.nanstd(data, axis=axis)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        assert (self.mean is not None) and (self.stddev is not None)
        return data * self.stddev + self.mean
