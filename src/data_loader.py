import os
import math
import glob
import numpy as np
from src.folderconstants import *
from torch.utils.data import Dataset, DataLoader


file_prefixes = {
	'SMD': ['machine-1-1_', 'machine-2-1_', 'machine-3-2_', 'machine-3-7_'],
	'SMAP_new': ['A-4_', 'T-1_'],
	'MSL_new': 'C-2_',
	'UCR': '136_',
}


class MyDataset(Dataset):
    def __init__(self, dataset, window_size, step_size, modelname, flag='train', feats=-1, less=False, enc=False, k=-1, shuffle=False, forecasting=False):
        """
        Initializes the data loader with the specified parameters.
        Args:
            dataset (str): The name of the dataset to load.
            window_size (int): The size of the window for creating data segments.
            step_size (int): The step size for moving the window.
            modelname (str): The name of the model to be used.
            forecasting (bool, optional): A flag indicating whether to use forecasting instead of reconstruction. Defaults to False.
            flag (str, optional): The type of data to load (e.g., 'train', 'test'). Defaults to 'train'.
            feats (int, optional): The number of features in the dataset. Defaults to None (then automatically == nb of input features).
            less (bool, optional): A flag indicating whether to load a smaller subset (10k timestamps) of the data. Defaults to False.
            enc (bool, optional): A flag indicating whether to use encoding of timestamp. Defaults to False.
            k (int, optional): Whether to do a 5-fold cross validation, k indicates which fold to use for validation. Must be > 0 and defaults to -1.
            shuffle (bool, optional): A flag indicating whether to shuffle the train and validation data. Defaults to False.
        Methods:
            __load_data__(type): Loads the data based on the specified type.
            __make_windows__(data): Creates windows of data segments based on the window size.
        """

        self.data_name = dataset
        self.modelname = modelname

        self.window_size = window_size
        self.step_size = step_size
        self.feats = feats
        self.enc = enc
        self.enc_feats = self.__get_enc_feats__()
        self.shuffle = shuffle
        
        self.forecasting = forecasting
        self.flag = flag
        self.less = less
        assert k <= 5
        self.k = k - 1 if flag in ['train', 'valid'] else -1  # k is 0-indexed for correct data selection

        if self.data_name in ['ATLAS_DQM_TS']:
            self.__load_ATLAS_DQM_TS_data__(type=flag)
        else:
            self.__load_data__(type=flag)
        # self.feats = self.data.shape[1] if feats <= 0 else feats
        # if self.enc:
        #     self.feats -= self.enc_feats
        self.complete_data = self.data
        if self.window_size > 0:
            self.__make_windows__(self.data)

    def __load_data__(self, type='train'):
        folder = os.path.join(output_folder, self.data_name)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')

        file = 'train' if type == 'valid' else type
        labelfile = 'labels'
        kfold = False

        if self.less and self.data_name in file_prefixes.keys():
                if isinstance(file_prefixes[self.data_name], list):
                    paths = []; labelfile_complete = []
                    for prefix in file_prefixes[self.data_name]:
                        file_complete = prefix + file
                        labelfile_complete.append(prefix + labelfile)
                        paths.append(glob.glob(os.path.join(folder, f'*{file_complete}*.npy'))[0])
                else:
                    file = file_prefixes[self.data_name] + file
                    labelfile = file_prefixes[self.data_name] + labelfile
                    paths = glob.glob(os.path.join(folder, f'*{file}*.npy'))
        else:
            paths = glob.glob(os.path.join(folder, f'*{file}*.npy'))

        paths = sorted(paths)  # sort paths to ensure correct order, otherwise labels & test files are mismatched
        if self.k >= 0 and len(paths) > 1:
            if self.k >= len(paths):
                self.k = 0
            n = max(len(paths) // 5, 1)
            if type == 'train':
                paths = paths[:self.k*n] + paths[(self.k+1)*n:]
            elif type == 'valid':
                paths = paths[self.k*n:(self.k+1)*n] 
            kfold = True
        data = np.concatenate([np.load(p) for p in paths])
        ts_lengths = [np.load(p).shape[0] for p in paths]

        if type == 'test':
            if self.less and self.data_name in file_prefixes.keys() and isinstance(file_prefixes[self.data_name], list):
                l_paths = []
                for l in labelfile_complete:
                    l_paths.append(glob.glob(os.path.join(folder, f'*{l}*.npy'))[0])
            else:
                l_paths = glob.glob(os.path.join(folder, f'*{labelfile}*.npy'))
            if self.less and self.data_name in file_prefixes.keys() and isinstance(file_prefixes[self.data_name], list):
                l_paths = []
                for l in labelfile_complete:
                    l_paths.append(glob.glob(os.path.join(folder, f'*{l}*.npy'))[0])
            else:
                l_paths = glob.glob(os.path.join(folder, f'*{labelfile}*.npy'))
            labels = np.concatenate([np.load(p) for p in l_paths])

        if self.feats > 0:
            if self.feats > data.shape[1]:
                self.feats = data.shape[1]
            if self.enc:
                max_feats = self.feats + self.enc_feats
            else:
                max_feats = self.feats
            data = data[:, :max_feats]
            if type == 'test' and labels.ndim > 1:
                labels = labels[:, :max_feats]
        else:
            self.feats = data.shape[1] - self.enc_feats

        if self.less and self.data_name not in file_prefixes.keys():
            data = data[:10000]
            data = data[:10000]
            ts_lengths = [data.shape[0]]
            if type == 'test':  
                labels = labels[:10000]
        
        # 5-fold cross validation
        if not kfold and self.k >= 0 and len(paths) == 1:
            n = data.shape[0] // 5
            if type == 'train':
                data = np.concatenate([data[:self.k*n], data[(self.k+1)*n:]])
                ts_lengths = [data.shape[0]]
            elif type == 'valid':
                data = data[self.k*n:(self.k+1)*n]
                ts_lengths = [data.shape[0]]

        self.data = data
        self.ts_lengths = ts_lengths
        if type == 'test':
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            self.labels = labels

    def __load_ATLAS_DQM_TS_data__(self, type='train'):
        # specific loading function for ATLAS_DQM_TS dataset (not public)
        folder = os.path.join('processed', self.data_name)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')

        file = 'train' if type == 'valid' else type
        labelfile = 'labels'

        if type == 'test':
            # 'ATLAS_DQM_TS': 'cosmicCalo_',  # 'hardProbes_', 'hvononNominal_', 'pumpNoise_'
            run = 'hardProbes_'
            file = run + file
            labelfile = run + labelfile

        paths = glob.glob(os.path.join(folder, f'*{file}*.npy'))
        paths = sorted(paths)  # sort paths to ensure correct order, otherwise labels & test files are mismatched
        if self.less and type == 'train':
            paths = paths[:20]
        data = np.concatenate([np.load(p) for p in paths])
        ts_lengths = [np.load(p).shape[0] for p in paths]
        
        if self.feats > 0:
            if self.feats > data.shape[1]:
                self.feats = data.shape[1]
            if self.enc:
                max_feats = self.feats + self.enc_feats
            else:
                max_feats = self.feats
            data = data[:, :max_feats]
        else:
            self.feats = data.shape[1] - self.enc_feats

        if type == 'test':
            l_paths = glob.glob(os.path.join(folder, f'*{labelfile}*.npy'))
            labels = np.concatenate([np.load(p) for p in l_paths])

        self.data = data
        self.ts_lengths = ts_lengths
        if type == 'test':
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            self.labels = labels


    def __make_windows__(self, data):
        """
        Converts the input time series data into overlapping (except if window_size == step_size) windows.
        Parameters:
        data (np.array): The input time series data.
        Creates:
        tuple: A tuple containing:
            - windows (np.array): The tensor containing the windows of the time series data.
            - ideal_lengths (list): A list containing the lengths of the padded time series.
        """

        ideal_lengths = []
        if ('iTransformer' in self.modelname or self.modelname in ['LSTM_AE', 'Transformer', 'AnomalyTransformer']) and not self.forecasting: 
            windows = np.empty((0, self.window_size, data.shape[1]))
            start = 0
            for l in self.ts_lengths:
                # get number of complete windows with window size window_size and step size step_size + 1 incomplete
                nb_windows = math.ceil((l - self.window_size) / self.step_size) + 1  
                if nb_windows <= 0: nb_windows = 1            
                # if not enough data for one window, create one window with padding, ideal_len: length of padded time series        	   
                ideal_len = (nb_windows - 1) * self.step_size + self.window_size    
                ideal_lengths.append(ideal_len)
                # separate the individual time series before slicing it into windows to avoid overlap
                ts = data[start:start+l]  
                if ideal_len > l: # pad with last element to have a multiple of window_size
                    ts = np.concatenate((ts, ts[-1:].repeat(ideal_len - l, 0)), axis=0)  
                new_window = np.stack([ts[i*self.step_size:i*self.step_size+self.window_size] for i in range(nb_windows)])
                windows = np.concatenate((windows, new_window), axis=0)
                start += l
            self.data = windows
            self.ideal_lengths = ideal_lengths
        else:  # alternative version where first few windows repeat the first element, then always have 1 new element per window
            windows = []
            self.ideal_lengths = self.ts_lengths
            for i, g in enumerate(data): 
                if i >= self.window_size: w = data[i-self.window_size:i]
                else: w = np.concatenate([data[:1].repeat(self.window_size-i, 0), data[0:i]])
                windows.append(w if self.modelname in ['TranAD', 'iTransformer', 'Transformer', 'AnomalyTransformer', 'LSTM_AE'] else w.reshape(-1))
            windows = np.stack(windows)
            self.data = windows
            if self.modelname not in ['TranAD','iTransformer', 'Transformer', 'LSTM_AE', 'None', 'USAD', 'AnomalyTransformer']:
                self.feats = windows.shape[1]

    def __len__(self):
        if self.forecasting:
            return len(self.data) - 1
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.forecasting:
            sample_y = self.data[idx+1]
            return sample, sample_y
        return sample
    
    def get_ts_lengths(self):
        return self.ts_lengths
    
    def get_ideal_lengths(self):
        return self.ideal_lengths
    
    def __get_enc_feats__(self):
        # number of additional time encoder covariates, usually 4
        if self.enc:
            return 4
        else:
            return 0
    
    def get_labels(self):
        assert self.flag == 'test'
        # if labels are 1D, repeat them for each feature to have 2D labels
        if self.feats != self.labels.shape[1]:
            self.labels = np.repeat(self.labels, self.feats, axis=1)
        if self.forecasting:
            # self.labels = self.labels[self.window_size:] # remove first window_size labels
            self.labels = self.labels[:-1] # remove last label because of forecasting
        return self.labels
    
    def get_complete_data(self):
        # for plots or if we want to use unsliced data
        return self.complete_data[:, self.enc_feats:]
    
    def get_complete_data_wpadding(self):
        # for plots or if we want to use unsliced data with padding
        if self.modelname in ['TranAD', 'iTransformer', 'Transformer', 'LSTM_AE']:
            return self.data[:, :, self.enc_feats:].reshape(-1, self.feats)
        else:
            return self.data[:, self.enc_feats:].reshape(-1, self.feats)


if __name__ == '__main__':
    dataset = 'GECCO'
    fc = True
    # Create dataset
    train = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='train', feats=30, less=False, enc=False, k=2, forecasting=fc)
    valid = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='valid', feats=30, less=False, enc=False, k=2, forecasting=fc)
    test = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='test', feats=30, less=False, enc=False, k=-1, forecasting=fc)
    print(train.__len__(), train.data.shape, train.complete_data.shape)
    print(valid.__len__(), valid.data.shape)
    print(train.get_ts_lengths(), valid.get_ts_lengths())
    print(test.__len__(), test.data.shape)
    labels = test.get_labels()
    print(labels.shape)

    # Create data loader
    data_loader_train = DataLoader(train, batch_size=24, shuffle=True)
    data_loader_test = DataLoader(test, batch_size=24, shuffle=True)

    # # # Iterate through the data loader
    # for batch in data_loader_train:
    #     print(batch.shape)
    # for batch in data_loader_test:
    #     print(batch.shape)