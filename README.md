# A simple implements of RCNN with tensorflow
# Requirements
- python 3.6
- tensorflow 1.2.1
# Content
- `images/*`: Some example imgs to test. 
- `finetune_alexnet/`: Storage each epoch's weights in checkpints and log the finetuning progress in TensorBoard.
- `Models/`: To storage alexnet weights and SVM Models.
- `alexnet.py`: an TensorFLow implementation of [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Alex Krizhevsky at all.
- `validate_alexnet`: If you wanna to validate alexnet, this script is helpful, you need are the pretrained weights, which you can find [here](http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy), place the weights file in 'Models/AlexNet/' and try command "python validat_alexnet.py":
                          ![images](https://github.com/asensioatgithub/RCNN/blob/master/validate.png)
- `finetune.py`、`datagenerator.py`: Scripts to finetune alexnet, [here](https://github.com/kratzert/finetune_alexnet_with_tensorflow) is the detail instructions to finetune alexnet.
- `generate_finetune_data.py`: To generate the fixed format of train data and val data required by `finetune.py`. Notice this file only works on VOCdevkit:

if you use other data sets, the implementation may not be the same. overall，each of them list the complete path to your train/val images together with the class number in the following structure.
```
Example train.txt:
/path/to/train/image1.png 0
/path/to/train/image2.png 1
/path/to/train/image3.png 2
/path/to/train/image4.png 0
.
.
```
- `ckpt2npy`: Change the format form '.ckpy' to '.npy' of alexnet weights.
- `tool.py`: Some func implements for other files calls, such as Non Maximum Suppression(NMS)、Intersection over Union(IOU)..., and also a func to call module `selectivesearch.selective_search`, you can try "python tool.py" to check the proposal rectangles generate by 'selective_search':
                          ![image](https://github.com/asensioatgithub/RCNN/blob/master/pro_rect.png)</br>
**Note**： The proposal rectangles are different according to the argu `scale, sigma, min_size` in func selective_search().

# Usage
1. Clone the Fast R-CNN repository
```
git clone https://github.com/asensioatgithub/RCNN.git
```
2. Click [here]() to download my tinetuned AlexNet weights \
3. Install the required python modules, and then instruct the follow cmd:
```
python demo.py
```

# Train Model 
here are my steps to experience RCNN：
1. wget the VOCPascll2012 datasets, or click [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar):
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2011.tar
```
2. Extract the tar to repo, so the directory looks like this:
```
images
Models
VOCtrainval_11-May-2012
alexnet.py
caffe_classes.py
...
```
3. Adjust parameter in config.py scenes
```
class_name = ["bus", "person", "car"] # The classes what rcnn to detect
...
```
4. run generate_finetune_data.py to generate `finetune_train_list.txt` and `finetune_val_list.txt`, they are looks like:
