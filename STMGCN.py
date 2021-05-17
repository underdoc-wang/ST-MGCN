from torch import nn
import torch
from GCN import GCN



class CG_LSTM(nn.Module):
    def __init__(self, seq_len:int, n_nodes:int, input_dim:int,
                 lstm_hidden_dim: int, lstm_num_layers: int,
                 K:int, gconv_use_bias:bool, gconv_activation=nn.ReLU):
        super().__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.gconv_temporal_feats = GCN(K=K, input_dim=seq_len, hidden_dim=seq_len,
                                        bias=gconv_use_bias, activation=gconv_activation)
        self.fc = nn.Linear(in_features=seq_len, out_features=seq_len, bias=True)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers, batch_first=True)

    def forward(self, adj:torch.Tensor, obs_seq:torch.Tensor, hidden:tuple):
        '''
        Context Gated LSTM:
            1. temporal obs_seq as feature for region, convolve neighbors on adj
            2. global pool -> FC -> FC -> temporal obs weights
            3. re-weighted obs_seq -> global-shared LSTM
        :param adj: support adj matrices for collecting neighbors - torch.Tensor (K, n_nodes, n_nodes)
        :param obs_seq: observation sequence - torch.Tensor (batch_size, seq_len, n_nodes, n_feats)
        :param hidden: tuple of hidden states (h, c) - torch.Tensor (n_layers, batch_size*n_nodes, hidden_dim) x2
        :return:
        '''
        batch_size = obs_seq.shape[0]
        x_seq = obs_seq.sum(dim=-1)     # sum up feature dimension: default 1

        # channel-wise attention on timestep
        x_seq = x_seq.permute(0, 2, 1)
        x_seq_gconv = self.gconv_temporal_feats(A=adj, x=x_seq)
        x_hat = torch.add(x_seq, x_seq_gconv)       # eq. 6
        z_t = x_hat.sum(dim=1)/x_hat.shape[1]       # eq. 7
        s = torch.sigmoid(self.fc(torch.relu(self.fc(z_t))))    # eq. 8
        obs_seq_reweighted = torch.einsum('btnf,bt->btnf', [obs_seq, s])      # eq. 9

        # global-shared LSTM
        shared_seq = obs_seq_reweighted.permute(0, 2, 1, 3).reshape(batch_size*self.n_nodes, self.seq_len, self.input_dim)
        x, hidden = self.lstm(shared_seq, hidden)

        output = x[:, -1, :].reshape(batch_size, self.n_nodes, self.lstm_hidden_dim)
        return output, hidden

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim),
                  weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim))
        return hidden



class ST_MGCN(nn.Module):
    def __init__(self, M:int, seq_len:int, n_nodes:int, input_dim:int, lstm_hidden_dim:int, lstm_num_layers:int,
                 gcn_hidden_dim:int, sta_kernel_config:dict, gconv_use_bias:bool, gconv_activation=nn.ReLU):
        super().__init__()
        self.M = M
        self.sta_K = self.get_support_K(sta_kernel_config)

        # initiate one pair of CG_LSTM & GCN for each adj input
        self.rnn_list, self.gcn_list = nn.ModuleList(), nn.ModuleList()
        for m in range(self.M):
            cglstm = CG_LSTM(seq_len=seq_len, n_nodes=n_nodes, input_dim=input_dim,
                             lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers,
                             K=self.sta_K, gconv_use_bias=gconv_use_bias, gconv_activation=gconv_activation)
            self.rnn_list.append(cglstm)
            gcn = GCN(K=self.sta_K, input_dim=lstm_hidden_dim, hidden_dim=gcn_hidden_dim,
                      bias=gconv_use_bias, activation=gconv_activation)
            self.gcn_list.append(gcn)
        self.fc = nn.Linear(in_features=gcn_hidden_dim, out_features=input_dim, bias=True)

    @staticmethod
    def get_support_K(config:dict):
        if config['kernel_type'] == 'localpool':
            assert config['K'] == 1
            K = 1
        elif config['kernel_type'] == 'chebyshev':
            K = config['K'] + 1
        elif config['kernel_type'] == 'random_walk_diffusion':
            K = config['K'] * 2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')
        return K

    def init_hidden_list(self, batch_size:int):
        hidden_list = list()
        for m in range(self.M):
            hidden = self.rnn_list[m].init_hidden(batch_size)
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, obs_seq:torch.Tensor, sta_adj_list:list):
        '''
        On each graph do CG_LSTM + GCN -> sum -> fc output
        :param obs_seq: observation sequence - torch.Tensor (batch_size, seq_len, n_nodes, n_feats)
        :param sta_adj_list: [(K_supports, N, N)] * M_sta
        :return: y_pred (t+1) - torch.Tensor (batch_size, n_nodes, n_feats)
        '''
        assert len(sta_adj_list) == self.M
        batch_size = obs_seq.shape[0]
        hidden_list = self.init_hidden_list(batch_size)

        feat_list = list()
        for m in range(self.M):
            cg_rnn_out, hidden_list[m] = self.rnn_list[m](sta_adj_list[m], obs_seq, hidden_list[m])
            gcn_out = self.gcn_list[m](sta_adj_list[m], cg_rnn_out)
            feat_list.append(gcn_out)
        feat_fusion = torch.sum(torch.stack(feat_list, dim=-1), dim=-1)     # aggregation

        output = self.fc(feat_fusion)
        return output


