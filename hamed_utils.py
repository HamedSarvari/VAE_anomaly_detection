import numpy as np

#######################################################################################################################
#######################################################################################################################

# Input:
# Number of input features, alpha: drop rate in layer size
# Output: dimensions of the custom_autoencoder

def create_layer_dims(input_size, alpha= 0.5, num_layers= 7 , bottleneck_size = 3):

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
    out[out<bottleneck_size*2]=bottleneck_size*2
    return list(out)


#print(create_layer_dims(36,num_layers=7))