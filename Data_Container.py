import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader



class DataInput(object):
    def __init__(self, M_adj:int, data_dir:str, norm_opt:bool):
        self.M_sta = M_adj
        self.data_dir = data_dir
        self.norm_opt = norm_opt

    def load_data(self):
        print('Loading data...')
        npz_data = np.load(self.data_dir)
        print('Available keys:', list(npz_data.keys()))

        dataset = dict()
        # taxi demand
        dataset['taxi'] = self.minmax_normalize(npz_data['taxi']) if self.norm_opt else npz_data['taxi']
        # sta_adj
        if self.M_sta >= 1:
            dataset['neighbor_adj'] = npz_data['neighbor_adj']
        if self.M_sta >= 2:
            dataset['trans_adj'] = npz_data['trans_adj']
        if self.M_sta >= 3:
            dataset['semantic_adj'] = npz_data['semantic_adj']
        return dataset

    def minmax_normalize(self, x:np.array):
        self._max, self._min = x.max(), x.min()
        print('min:', self._min, 'max:', self._max)
        x = (x - self._min) / (self._max - self._min)
        x = 2 * x - 1
        return x

    def minmax_denormalize(self, x:np.array):
        x = (x + 1)/2
        x = (self._max - self._min) * x + self._min
        return x

    def std_normalize(self, x:np.array):
        self._mean, self._std = x.mean(), x.std()
        print('mean:', round(self._mean, 4), 'std:', round(self._std, 4))
        x = (x - self._mean)/self._std
        return x

    def std_denormalize(self, x:np.array):
        x = x * self._std + self._mean
        return x


class TaxiDataset(Dataset):
    '''
        inputs: history obs: short-term seq | daily seq | weekly seq (B, seq, N, C)
        output: y_t+1 target (B, N, C)
        mode: one in [train, validate, test]
        mode_len: {train, validate, test}
    '''
    def __init__(self, device:str, inputs:dict, output:np.array, mode:str, mode_len:dict, start_idx:int):
        self.device = device
        self.mode = mode
        self.mode_len = mode_len
        self.start_idx = start_idx  # train_start idx
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.output[item]

    def prepare_xy(self, inputs:dict, output:np.array):
        if self.mode == 'train':
            pass
        elif self.mode == 'validate':
            self.start_idx += self.mode_len['train']
        else:       # test
            self.start_idx += self.mode_len['train'] + self.mode_len['validate']

        obs = []
        for kw in ['weekly', 'daily', 'serial']:
            if len(inputs[kw].shape) != 2:      # dim=2 for empty seq
                obs.append(inputs[kw])
        x_seq = np.concatenate(obs, axis=1)   # concatenate timeslices to one seq
        x = dict()
        x['x_seq'] = torch.from_numpy(x_seq[self.start_idx : (self.start_idx + self.mode_len[self.mode])]).float().to(self.device)
        y = torch.from_numpy(output[self.start_idx : self.start_idx + self.mode_len[self.mode]]).float().to(self.device)
        return x, y



class DataGenerator(object):
    def __init__(self, dt:int, obs_len:tuple, train_test_dates:list, val_ratio:float, year=2017):
        self.day_timesteps = 24//dt
        self.serial_len, self.daily_len, self.weekly_len = obs_len
        self.train_test_dates = train_test_dates        # [train_start, train_end, test_start, test_end]
        self.val_ratio = val_ratio
        self.start_idx, self.mode_len = self.date2len(year=year)

    def date2len(self, year:int):
        date_range = pd.date_range(str(year)+'0101', str(year)+'1231').strftime('%Y%m%d').tolist()
        train_s_idx, train_e_idx = date_range.index(str(year)+self.train_test_dates[0]),\
                                   date_range.index(str(year)+self.train_test_dates[1])
        train_len = (train_e_idx + 1 - train_s_idx) * self.day_timesteps
        validate_len = int(train_len * self.val_ratio)
        train_len -= validate_len
        test_s_idx, test_e_idx = date_range.index(str(year)+self.train_test_dates[2]),\
                                 date_range.index(str(year)+self.train_test_dates[3])
        test_len = (test_e_idx + 1 - test_s_idx) * self.day_timesteps
        return train_s_idx, {'train':train_len, 'validate':validate_len, 'test':test_len}

    def get_data_loader(self, data:dict, batch_size:int, device:str):
        feat_dict = dict()
        feat_dict['serial'], feat_dict['daily'], feat_dict['weekly'], output = self.get_feats(data['taxi'])

        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = TaxiDataset(device=device, inputs=feat_dict, output=output,
                                  mode=mode, mode_len=self.mode_len, start_idx=self.start_idx)
            data_loader[mode] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return data_loader

    def get_feats(self, data:np.array):
        serial, daily, weekly, y = [], [], [], []
        start_idx = max(self.serial_len, self.daily_len*self.day_timesteps, self.weekly_len*self.day_timesteps * 7)
        for i in range(start_idx, data.shape[0]):
            serial.append(data[i-self.serial_len : i])
            daily.append(self.get_periodic_skip_seq(data, i, 'daily'))
            weekly.append(self.get_periodic_skip_seq(data, i, 'weekly'))
            y.append(data[i])
        return np.array(serial), np.array(daily), np.array(weekly), np.array(y)

    def get_periodic_skip_seq(self, data:np.array, idx:int, p:str):
        p_seq = list()
        if p == 'daily':
            p_steps = self.daily_len * self.day_timesteps
            for d in range(1, self.daily_len+1):
                p_seq.append(data[idx - p_steps*d])
        else:   # weekly
            p_steps = self.weekly_len * self.day_timesteps * 7
            for w in range(1, self.weekly_len+1):
                p_seq.append(data[idx - p_steps*w])
        p_seq = p_seq[::-1]     # inverse order
        return np.array(p_seq)


