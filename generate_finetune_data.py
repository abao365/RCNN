import re
import cv2
import config
import random
import numpy as np
import selectivesearch
import xml.etree.cElementTree as ET
from tool import clip_pic, resize_image, IOU

JPEGImages = config.JPEGImages
ImageSets = config.ImageSets
Annotations = config.Annotations

class_name = config.class_name 

def img_XML_ref_rect(name, class_name):
    ref_rects = []
    xml_file = Annotations + "/" + name+".xml"
    tree = ET.parse(xml_file)     #打开xml文档
    root = tree.getroot()         #获得root节点 
    for each_object in root.findall('object'): #找到root节点下的object节点 
        if class_name !=  each_object.find("name").text:
            continue
        bndbox = each_object.find('bndbox')
        ref_rect = []
        ref_rect.append(bndbox.find('xmin').text)
        ref_rect.append(bndbox.find('ymin').text)
        ref_rect.append(bndbox.find('xmax').text)
        ref_rect.append(bndbox.find('ymax').text)
        ref_rects.append(ref_rect)
    # print(ref_rect)
    return ref_rects

def do(type, threshold):
    img_path_list = []
    for i in range(len(class_name)):
        print("Extracting %s..."%class_name[i])
        # f_list = open("finetune_train_list.txt", "a")
        ImageSet = ImageSets+"/"+ class_name[i] +"_train.txt"
        if type == 0:
            # f_list = open("finetune_val_list.txt", "a")
            ImageSet = ImageSets+"/"+ class_name[i] +"_val.txt"
        with open(ImageSet, "r") as f:
            directory = f.read()
        pattern = re.compile(r'\d+_\d+\s+1')
        result = pattern.findall(directory)
        train_list = [item.split(' ')[0] for item in result]
        path = ""
        for each in train_list:
            # 读取对应路径的img
            img_path = JPEGImages+"/"+each+".jpg"
            img =  cv2.imread(img_path)

            # 从xml解析出该img的人工标注框
            ref_rects = img_XML_ref_rect(each, class_name[i])

            # 生成该img的候选框
            imgs, regions = selectivesearch.selective_search(img, scale=500, sigma=0.5, min_size=500)
#            print(len(regions))
            candidates = set()
            for r in regions:
                
                # 剔除重复的方框
                if r['rect'] in candidates:
                    continue
                proposal_img, proposal_vertice = clip_pic(img, r['rect'])
                if len(proposal_img) == 0:
                    continue
                x, y, w, h = r['rect']
                # 长或宽为0的方框，剔除z
                if w == 0 or h == 0:
                    continue
                [a, b, c] = np.shape(proposal_img)
                if a == 0 or b == 0 or c == 0:
                    continue
                candidates.add(r['rect'])
                resized_proposal_img = resize_image(proposal_img, 227, 227)
                img_float = np.asarray(resized_proposal_img, dtype="float32")
                
                if type == 1:
                    path = 'finetune_images/train/%s_%s_%d-%d-%d-%d.jpg'%(class_name[i], each, proposal_vertice[0], proposal_vertice[1], proposal_vertice[2],proposal_vertice[3])
                    cv2.imwrite(path, img_float, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    path = 'finetune_images/val/%s_%s_%d-%d-%d-%d.jpg'%(class_name[i], each, proposal_vertice[0], proposal_vertice[1], proposal_vertice[2],proposal_vertice[3])
                    cv2.imwrite(path, img_float, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
                # 计算该样本集中中的每个候选图像的正负label (iou大于threshold则为正样本，否则为负样本(背景))
                iou = 0
                # 计算该候选框在人工标注框中的最大iou
                for j in range(len(ref_rects)):
                    tmp = IOU(proposal_vertice, ref_rects[j]) 
                    if(tmp > iou):
                        iou = tmp
                if iou < threshold:
                    img_path_list.append("%s %d\n"%(path, config.background_index))
                else:
                    img_path_list.append("%s %d\n"%(path, i))
    
    ## oversampling (echa batch: 128, pos: 36, neg: 96)
    neg = []
    pos = []
    for line in img_path_list:
        if line.strip().split(' ')[1] == str(config.background_index):
            neg.append(line)
        else:
            pos.append(line)
    random.shuffle(neg)
    random.shuffle(pos)
    neg_size = config.neg_size_each_batch
    pos_size = config.pos_size_each_batch
    count = int(len(neg)/neg_size)
    p = config.finetune_train_list
    if type == 0:
         p = config.finetune_val_list
    f_r = open(p,"w")
    for i in range(count):
        result = []
        for j in range(i*neg_size, (i+1)*neg_size):
            result.append(neg[j])
        for k in range(i*pos_size, (i+1)*pos_size):
            result.append(pos[int(k%len(pos))])
        print(len(result))
        random.shuffle(result)
        # print(asd)
        for l in range(len(result)):
            f_r.write(result[l])

def main():
    print("generating train...")
    do(1, config.finetune_IOU_threshold) # 生成训练
    print("generating val...")
    do(0, config.finetune_IOU_threshold) # 生成验证集
    
    print("succesfully !")

if __name__ == '__main__':
    main()
