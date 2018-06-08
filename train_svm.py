#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  
# gpuConfig = tf.ConfigProto(allow_soft_placement=True)
# gpuConfig.gpu_options.allow_growth = True  
import re
import cv2
import config
import pickle
import joblib
import alexnet
import xml.etree.cElementTree as ET 
import selectivesearch
import numpy as np

from sklearn import svm
from tool import image_rect_proposal, IOU

#with tf.device('/device:GPU:2'):
 #   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  #  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

JPEGImages = config.JPEGImages
ImageSets = config.ImageSets
Annotations = config.Annotations

class_name = config.class_name

def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X,Y


def img_XML_ref_rect(name):
    ref_rects = []
    xml_file = Annotations + "/" + name+".xml"
    tree = ET.parse(xml_file)     #打开xml文档
    root = tree.getroot()         #获得root节点 
    for each_object in root.findall('object'): #找到root节点下的object节点 
        bndbox = each_object.find('bndbox')
        ref_rect = []
        ref_rect.append(bndbox.find('xmin').text)
        ref_rect.append(bndbox.find('ymin').text)
        ref_rect.append(bndbox.find('xmax').text)
        ref_rect.append(bndbox.find('ymax').text)
        ref_rects.append(ref_rect)
    # print(ref_rect)
    return ref_rects


def load_trainSet(ImageSet, classNum, classType, threshold = 0.7):
    images = []
    labels = []
    with open(ImageSet, "r") as f:
        directory = f.read()
        # print(text)
    pattern = re.compile(r'\d+_\d+\s+1')
    result = pattern.findall(directory)
    train_list = [item.split(' ')[0] for item in result]
    # print(len(train_list))
    for each in train_list:
        # each = train_list[0]
        # 读取对应路径的img
        img_path = JPEGImages+"/"+each+".jpg"
        img =  cv2.imread(img_path)

        # 从xml解析出该img的人工标注框
        ref_rects = img_XML_ref_rect(each)

        # 生成img的候选框, pro_imgs[i]为pro_rects[i]
        pro_imgs, pro_rects = image_rect_proposal(img) 
        # print(pro_imgs[0])
        # print("%s.img have %d proposal rects"%(each,len(pro_rects)))
        
        # 将生成的候选框图像加入到样本集中
        images.extend(pro_imgs)
        
        # 计算该样本集中中的每个候选图像的正负label (iou大于threshold则为正样本，否则为负样本(背景))
        for i in range(len(pro_rects)):
            iou = 0
            # 计算该候选框在人工标注框中的最大iou
            for j in range(len(ref_rects)):
                tmp = IOU(pro_rects[i], ref_rects[j]) 
                if(tmp > iou):
                    iou = tmp
            # print("rects[%d]'s IOU: %f"%(i, iou))
            if iou < threshold:
                labels.append(config.background_index) # The classNumth represent the background
            else:
                labels.append(classType)
    # print(pro_imgs)
    return images, labels


dropoutPro = 0.5
classNum = config.num_class +1
skip = []

imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 227, 227, 3])

model = alexnet.AlexNet(x, dropoutPro, classNum, skip, weights_path="./Models/AlexNet/alexnet_finetuned.npy")


def train_svms():
    SVMS = []
    # with tf.Session() as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)
        # 为每一类训练一个SVM二分类器
        for i in range(len(class_name)):
            print("Traning %s SVM classifier"%class_name[i])
            train_features = []

       #     print "Generating training data..."
           # datapath = './SVM_Data/%s_train_data.pkl'%class_name[i]
      #      if os.path.isfile(datapath):
                #images, labels = load_from_pkl(datapath)
            #else:   
            ImageSet = ImageSets+"/"+ class_name[i] +"_train.txt"
            images, labels = load_trainSet(ImageSet, classNum, i, threshold = config.SVM_IOU_threshold)
               # pickle.dump((images, labels), open(datapath, 'wb'))

            print("Extracting features from CNN")
            for j in range(len(images)):
                # 将img输入到AlexNet, 提取第7层的输出特征
                img = images[j] - imgMean
                feature = sess.run(model.fc7, feed_dict = {x: img.reshape((1, 227, 227, 3))})
                train_features.append(feature[0])

            print("Fitting %s's SVM"%class_name[i])
            sigSVM = svm.SVC(probability=True)
            sigSVM.fit(train_features, labels)

            print("SVM %s is fitted!"%class_name[i])
            joblib.dump(sigSVM, "./Models/SVM_Models/%s.pkl"%class_name[i])#保存SVM模型
            
        print("Train SVNS compeletly!")
        

def main():
    train_svms()

if __name__ == '__main__':
    main()








        





