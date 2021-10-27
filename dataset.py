import torch
from torch.utils.data import Dataset, TensorDataset
from Autoencoder_utils_torch import Read_data, find_outlier_inlier, np_to_tensor
import numpy as np


# a=torch.rand(60_000, 5)
# print(a.__len__())
# aa
# temp = torch.rand(10, 2)
# print(temp)
# tt = TensorDataset(temp)
#
# print(tt.__getitem__(4))

#print(data)
# inliers = True uses only the inliers in the training and removes the outliers
# Mnist= true loads an mnist dataset

def load_dataset(ds_name, inliers = False, mnist=False) -> Dataset:

    if mnist == True:
        data, labels = Read_data(ds_name, type='MNIST')
    else:
        data, labels = Read_data(ds_name)

    if inliers == True:

        outlier_d, inlier_d = find_outlier_inlier(data, labels)
        inlier_d_t = np_to_tensor(inlier_d)
        return TensorDataset(inlier_d_t.float())

    else:
        data_np = np.array(data)
        data_t = torch.from_numpy(data_np)
        # Overwrite this to load your dataset
        return TensorDataset(data_t.float())


def get_data_label(ds_name, inliers= False, mnist= False):

    if mnist == True:
        data, labels = Read_data(ds_name, type='MNIST')
    else:
        data, labels = Read_data(ds_name)

    if inliers == True:

        outlier_d, inlier_d = find_outlier_inlier(data, labels)
        data_np = np.array(inlier_d)
        inlier_t = torch.from_numpy(data_np)
        labels= ['no']*data_np.shape[0]
        return inlier_d, inlier_t.float(), labels

    else:

        data_np = np.array(data)
        data_t = torch.from_numpy(data_np)
        return data, data_t.float(), labels

#print(torch.rand(10000, 512).shape[0])
