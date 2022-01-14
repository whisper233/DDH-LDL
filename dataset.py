import scipy.io as sio
import torch.utils.data as data
import torch
import utils
import numpy as np
import logging

def load_dataset(path):
    dataset_name = utils.get_dataset_name(path)
    logging.info('Loading dataset: %s' % dataset_name)
    data = sio.loadmat(path)
    
    train_feature = data['train_features']
    train_label = data['train_labels']

    test_feature = data['test_features']
    test_label = data['test_labels']

    logging.info('train:%d' % train_feature.shape[0] +'test:%d' % test_feature.shape[0] + 'feat_dim:%d' % train_feature.shape[1] + 'label_dim:%d' % train_label.shape[1])
    return train_feature, train_label, test_feature, test_label

class LdlDatasetFunc(data.Dataset):
    def __init__(self, feature, B, label):
        self.feature = torch.Tensor(feature)
        self.B = torch.Tensor(B)
        self.label = torch.Tensor(label)
        self.length = self.feature.size(0)

    def __getitem__(self, item):
        return self.feature[item, :], self.B[item, :], self.label[item, :]

    def __len__(self):
        return self.length

class LdlDatasetCode(data.Dataset):
    def __init__(self, feature, label):
        self.feature = torch.Tensor(feature)
        self.label = torch.Tensor(label)
        self.length = self.feature.size(0)

    def __getitem__(self, item):
        return self.feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length

