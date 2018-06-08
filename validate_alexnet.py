import os
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes


#some params
dropoutPro = 1
classNum = 2
skip = []

imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 227, 227, 3])

model = alexnet.AlexNet(x, dropoutPro, classNum, skip, weights_path="./Models/Alexnet/bvlc_alexnet.npy")
score = model.fc8
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    img = cv2.imread("./VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000549.jpg")
    #img preprocess
    resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
    probs = sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))})
    # print(probs.size)
    class_name = caffe_classes.class_names[np.argmax(probs)]
    res = "class:"+class_name+"  probabliity:%.4f"%probs[0,np.argmax(probs)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, res, (0, 20), font, 0.5, (0, 0, 255), 1)
    cv2.imshow("demo", img)
    cv2.waitKey(0)

