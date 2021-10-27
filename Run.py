# Not a part of package ... I wrote this. test.
from VAE_tf1 import VAE
import torch
from torch.utils.data import Dataset, TensorDataset

import tensorflow as tf

#######################################################################################################################

#data = tf.random.uniform(shape=[10,10])
#data =  TensorDataset(torch.rand(6000, 100))
#print(data.shape)
encode_sizes= [200,50,20]
latent_size= 5
model = VAE((5000,200,1), encode_sizes, latent_size, decode_sizes=None, mu_prior=None, sigma_prior=None,
                  lr=10e-4,  momentum=0.9, save_model=True)


