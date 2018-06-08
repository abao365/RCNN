
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='./finetune_alexnet/checkpoints/model_epoch10.ckpt' # 不加后面的data-00000-of-00001
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

alexnet={}
alexnet_layer = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
add_info = ['weights','biases']

alexnet={'conv1':[[],[]],'conv2':[[],[]],'conv3':[[],[]],'conv4':[[],[]],'conv5':[[],[]],'fc6':[[],[]],'fc7':[[],[]],'fc8':[[],[]]}


for key in var_to_shape_map:
    #print ("tensor_name",key)

    str_name = key
    # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    if str_name.find('Adam') > -1:
        continue

    print('tensor_name:' , str_name)

    if str_name.find('/') > -1:
        names = str_name.split('/')
        # first layer name and weight, bias
        layer_name = names[0]
        layer_add_info = names[1]
    else:
        layer_name = str_name
        layer_add_info = None

    if layer_add_info == 'weights':
        alexnet[layer_name][0]=reader.get_tensor(key)
    elif layer_add_info == 'biases':
        alexnet[layer_name][1] = reader.get_tensor(key)
    else:
        alexnet[layer_name] = reader.get_tensor(key)

# save npy
np.save('./Models/AlexNet/alexnet_finetuned.npy',alexnet)
print('save npy over...')
#print(alexnet['conv1'][0].shape)
#print(alexnet['conv1'][1].shape)
