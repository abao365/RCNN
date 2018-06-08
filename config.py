class_name = ["bus", "person", "car"]
num_class = len(class_name)
background_index = num_class

proposal_scale = 800
proposal_sigma = 0.5
proposal_minsize = 800

epoch = 10
batch_size = 128
pos_size_each_batch = 32
neg_size_each_batch = 96
learning_rate = 0.001
finetune_IOU_threshold = 0.5

SVM_IOU_threshold = 0.7

JPEGImages = "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
ImageSets = "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main"
Annotations = "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"

finetune_train_list = "./finetune_train_list.txt"
finetune_val_list = "./finetune_val_list.txt"
