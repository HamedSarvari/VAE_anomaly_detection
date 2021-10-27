from __future__ import division
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
#from CustomLinear import *
import pandas as pd
import random
import os
import pickle
from torch import nn
import torch

#######################################################################################################################
def load_obj(name):
    with open('../factoredGatedAutoencoder/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#######################################################################################################################
def eval_model(Errors, labels_dic):

    y_true = np.array(list(labels_dic.values()))

    if 'yes' in y_true:
        y_true = (y_true == 'yes') + 0
    else:
        y_true = list(y_true)

    return average_precision_score(y_true,list(Errors.values()))
###########################################################################################################################################
# equivalent to eval_model
def eval_model_deprecated(Errors, labels_dic):
    sorted_error_inds = sorted(Errors, key=Errors.get)
    y_true = np.array([labels_dic[i] for i in sorted_error_inds])
    y_true = (y_true == 'yes') + 0
    error_scores = range(1, len(sorted_error_inds) + 1)
    AUCPR = average_precision_score(y_true, error_scores)
    #print('AUCPR:', AUCPR)
    return  AUCPR

#######################################################################################################################
# Takes a dataset as a list and converts it to a dictionary assigning the index as the key
def data_to_dic(data):
    data=np.array(data)
    dic = {}
    for i in range(data.shape[0]): dic[i] = data[i,]
    return dic
#######################################################################################################################
# Creates a simple fully connected autoencoder with layer dims given as input to constructor
class simple_autoencoder(nn.Module):
    def __init__(self,layer_dims, bias = True):

        self.encoder_dims = layer_dims[:int(np.floor(len(layer_dims) / 2) + 1)]
        self.decoder_dims = layer_dims[int(np.floor(len(layer_dims) / 2) + 1) - 1:]
        super(simple_autoencoder, self).__init__()
        self.layer_dims = layer_dims
        self.layers=[]
        self.encoder_acts, self.decoder_acts = create_activations(len(layer_dims))

        self.encoder = add_layer_simple(self.encoder_dims,self.encoder_acts)
        self.decoder= add_layer_simple(self.decoder_dims,self.decoder_acts)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
#######################################################################################################################
# Creates an autoencoder with custom connections among layers taken as input to constructor - saves selection_prob percent of the connections randomly
class custom_autoencoder(nn.Module):
    def __init__(self,layer_dims, selection_prob, complete_decoder= True, bias = True):

        self.encoder_dims = layer_dims[:int(np.floor(len(layer_dims) / 2)+1 )]
        self.decoder_dims = layer_dims[int(np.floor(len(layer_dims) / 2)+1) - 1:]

        #print(self.encoder_dims)
        #print(self.decoder_dims)
        super(custom_autoencoder, self).__init__()
        self.layer_dims = layer_dims
        self.selection_prob = selection_prob
        self.layers=[]
        self.encoder_acts, self.decoder_acts = create_activations(len(layer_dims))


        self.encoder = add_layer_custom(self.encoder_dims,self.encoder_acts, self.selection_prob)
        #@TODO remove prob
        self.decoder= add_layer_custom(self.decoder_dims,self.decoder_acts,self.selection_prob)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
#######################################################################################################################
def add_layer_simple(layer_dims, activations):

    layers=[]
    for i, input_size in enumerate(layer_dims):
        if i == 0:
            continue
        layer_shape = (layer_dims[i - 1], layer_dims[i])
        Out = nn.Linear(layer_shape[0],layer_shape[1])
        layers.append(Out)
        layers.append(activations[i-1])

    layers_input = tuple(layers)
    encoder_decoder = nn.Sequential(*layers_input)

    return encoder_decoder

######################################################################################################################
def add_layer_custom(layer_dims, activations, selection_prob = 0.66):

    layers=[]
    for i, input_size in enumerate(layer_dims):
        if i == 0:
            continue
        layer_shape = (layer_dims[i - 1], layer_dims[i])
        mask = rand_gen(layer_shape[0], layer_shape[1], selection_prob)

        Out = CustomizedLinear(mask=mask)
        layers.append(Out)
        layers.append(activations[i-1])

    layers_input = tuple(layers)
    #print(layers_input)
    encoder_decoder = nn.Sequential(*layers_input)

    return encoder_decoder
#######################################################################################################################
# def add_layer_custom(layer_dims, selection_prob = 0.66):
#
#     layers=[]
#     for i, input_size in enumerate(layer_dims):
#         if i == 0:
#             continue
#         layer_shape = (layer_dims[i - 1], layer_dims[i])
#         mask = rand_gen(layer_shape[0], layer_shape[1], selection_prob)
#
#         if i == 1:
#             Out = CustomizedLinear(mask=mask)
#             layers.append(Out)
#             layers.append(nn.Sigmoid())
#         elif i== (len(layer_dims)-1):
#             Out = CustomizedLinear(mask=mask)
#             layers.append(Out)
#             layers.append(nn.Sigmoid())
#         else:
#             Out = CustomizedLinear(mask=mask)
#             layers.append(Out)
#             layers.append(nn.ReLU(True))
#
#     layers_input = tuple(layers)
#
#     encoder_decoder = nn.Sequential(*layers_input)
# #    print('encoder_decoder',encoder_decoder[0].parameters()[0])
#
#     return encoder_decoder
#######################################################################################################################

# Generates uniform random numbers in [0,1] and if the number if less than selection_prob adds 1 to adjacency matrix and 0 otherwise.

def rand_gen(in_dim, out_dim, selection_prob):
    shape = torch.Size((in_dim, out_dim))
    x = torch.FloatTensor(shape)
    out = torch.rand(shape, out=x)
    mask = out < selection_prob
    mask=mask.type(torch.FloatTensor)
#    tensor.type(torch.DoubleTensor)
    return mask

#######################################################################################################################
# Input: training data and model with learnt weights
# Output: A list with reconstruction error of each data point in training data

def Reconstruct_error(data_train, model):

    Reconstrcut_errors=[]
    for row in range(data_train.shape[0]):
        data = data_train[row]
        data = data.reshape(1, len(data))
        data = torch.FloatTensor(data)
        data = data.cuda()
        output = model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, data)
        Reconstrcut_errors.append(loss.data.item())
    return Reconstrcut_errors

#######################################################################################################################
# Error reconstruction function for the aggregated merge model
def Reconstruct_error_adjusted(data_train, model_agg, ensemble_size):

    Reconstrcut_errors=[]
    for row in range(data_train.shape[0]):

        data = data_train[row]
        adjusted_input = list(data) * ensemble_size
        adjusted_input = np.array(adjusted_input)

        data = data.reshape(1, len(data))
        adjusted_input = adjusted_input.reshape(1, len(adjusted_input))

        data=torch.FloatTensor(data)
        adjusted_input = torch.FloatTensor(adjusted_input)

        data=data.cuda()
        adjusted_input=adjusted_input.cuda()

        output = model_agg(adjusted_input)
        criterion = nn.MSELoss()
        loss = criterion(output, data)
        Reconstrcut_errors.append(loss.data.item())

    return Reconstrcut_errors

#######################################################################################################################
# Takes data input as dictionary and constructs the error as dictionary
def Reconstruct_error_dic(data_dic, model):

    reconstruct_errors={}
    for ind in data_dic.keys():
        data = data_dic[ind]
        data = data.reshape(1, len(data))
        data = torch.FloatTensor(data)
        data = data.cuda()
        output = model(data)
        criterion = nn.MSELoss()
        loss = criterion(output, data)
        reconstruct_errors[ind]=loss.data.item()

    return reconstruct_errors
#######################################################################################################################
# Input:
# Number of input features, alpha: drop rate in layer size
# Output: dimenstions of the custom_autoencoder

def create_layer_dims(input_size, alpha= 0.5, num_layers= 9 , bottleneck_size =3):

    if num_layers%2==0:
        half_size=num_layers/2
        tmp=[]
        for i in range(half_size):
            tmp.append(int(input_size))
            input_size = np.floor(input_size * alpha)
            out = tmp + list(reversed(tmp))
    else:
        half_size=int(num_layers/2)+1
        tmp = []
        for i in range(half_size):
            tmp.append(int(input_size))
            input_size = np.floor(input_size * alpha)
            out = tmp + list(reversed(tmp))[1:]
    out=np.array(out)
    out[out<bottleneck_size]=bottleneck_size
    return list(out)

#######################################################################################################################
#Activations with first and last layer sigmoid functions
def create_activations(num_layers):
    out=[]
    tmp=num_layers-1
    for i in range(tmp):
        if i==0:
            out.append(nn.Sigmoid())
        elif i== tmp-1 :
            out.append(nn.Sigmoid())
        else:
            out.append(nn.ReLU(True))

    if num_layers%2==0:
        encoder_acts = out[:int(np.floor(num_layers / 2))]
        decoder_acts = out[int(np.floor(num_layers / 2)):]
    else:
        encoder_acts = out[:int(np.floor(num_layers / 2))]
        decoder_acts = out[int(np.floor(num_layers / 2)):]

    return encoder_acts,decoder_acts
# ########################################################################################################################
# Activations with all ReLU functions
# def create_activations(num_layers):
#
#     out=[]
#     tmp=num_layers-1
#     for i in range(tmp):
#             out.append(nn.ReLU(True))
#
#     if num_layers%2==0:
#         encoder_acts = out[:int(np.floor(num_layers / 2))]
#         decoder_acts = out[int(np.floor(num_layers / 2)):]
#     else:
#         encoder_acts = out[:int(np.floor(num_layers / 2))]
#         decoder_acts = out[int(np.floor(num_layers / 2)):]
#
#     return encoder_acts,decoder_acts
###########################################################################################################################################
# def eval_model(Errors, labels_dic):
#     sorted_error_inds = sorted(Errors, key=Errors.get)
#     y_true = np.array([labels_dic[i] for i in sorted_error_inds])
#     y_true = (y_true == 'yes') + 0
#     error_scores = range(1, len(sorted_error_inds) + 1)
#     AUCPR = average_precision_score(y_true, error_scores)
#     #print('AUCPR:', AUCPR)
#     return  AUCPR
###########################################################################################################################################
def plot_loss(history, dataset):

    plt.plot(history.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('validation loss')
    plt.savefig(dataset+'_val_loss.png')

####################################### Data initialization ############################################################
# Checks for columns with the same value for all data points
def check_data_modify(data):
    data_min= data.min(axis=0)
    data_max= data.max(axis=0)
    col_names = data.columns
    bad_inds=[]
    #print(data)
    for i in range(data.shape[1]):
        if data_min[i]==data_max[i]:
            bad_inds.append(i)
            print('Problematic columns detected')
    #print('Problematic columns: ',bad_inds)
    #print('Column labels: ', col_names[bad_inds])
    for col in bad_inds:
        data= data.drop(labels=col_names[col],axis=1)
    return data
########################################################################################################################
#Read the dataset and remove the index column
def Read_data(dataset, type= 'arff', normalization='minmax'):
    if type == 'arff':
        address = './DataSets/' + dataset + '.arff'
        # address='/home/hamed/PycharmProjects/DeepEnsembles/DataSets/' + dataset + '.arff'

        data, meta = arff.loadarff(address)
        data = pd.DataFrame(data)
        if 'id' in data.columns:
            data = data.drop(labels='id', axis=1)

        labels = data['outlier']
        # Converting bytes to utf
        labels = [x.decode('utf-8') for x in labels]
        data = data.drop(labels='outlier', axis=1)
        data = check_data_modify(data)

        # Normalize - Push data points to [0,1]
        # data=np.array(data)
        if normalization == 'minmax':
            data_normal = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        elif normalization == 'max':
            data_normal = data / data.max(axis=0)

    elif type == 'MNIST':

        # address= './DataSets/MNIST/MNIST_inlier=' + str(dataset) + '.csv'
        address = './DataSets/MNIST/MNIST_inlier=' + str(dataset) + '.csv'

        data = pd.read_csv(address, header=0)
        data = pd.DataFrame(data)
        # print(data)
        if 'id' in data.columns:
            data = data.drop(labels='id', axis=1)
        labels = list(data['outlier'])
        data = data.drop(labels='outlier', axis=1)
        data = np.array(data)
        # data_normal=(data-data.min(axis=0)) / (data.max(axis=0)-data.min(axis=0))
        data_normal = data / 255


    elif type == 'TwoGauss':

        data_loaded = load_obj(dataset)
        data = data_loaded['data']
        labels = data_loaded['labels']
        if normalization=='minmax':
            data_normal = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        elif normalization=='max':
            data_normal = data / data.max(axis=0)

    else:
        print('Wrong type')

    return data_normal,labels

########################################################################################################################
# Gets a numpy array as data and a labels array
# Outputs two numpy arrays: inliers and outliers
def find_outlier_inlier(data, labels):

    labels = np.array(labels)
    outliers = data[labels == 'yes']
    inliers = data[labels == 'no']

    return outliers, inliers
########################################################################################################################
# gets a numpy array and returns a tensor
def np_to_tensor(data):
    data_np = np.array(data)
    return torch.from_numpy(data_np)
########################################################################################################################

#d,l=Read_data('TwoGauss_data_7dim',type='TwoGauss')
#print(d)
#print(l)
