import os
import cv2
import pickle
import skimage
import numpy as np
import selectivesearch
import matplotlib.pyplot as plt

# 非极大值抑制算法
def NMS(rects, threshold):
    # 按probability升序排序
    rects = sorted(rects, key=operator.itemgetter('prob'), reverse = True)
    length = len(rects)
    flag = [0] * length # 是否抑制标记数组
    # result.append(rects[0]) # rects[0]是当前分数最大的窗口，肯定保留 
    for i in range(len(rects)-1):
        if flag[i] == 1:
            continue
        for j in range(i+1, len(rects)):
            if flag[j] == 1:
                continue 
            if IOU(rects[i]['rect'], rects[j]['rect']) > threshold:
                flag[j] = 1
    result = []
    for i in range(len(flag)):
        if flag[i] == 0:
            result.append(rects[i])
    return result

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    '''
    
    :param in_image: 输入的图片
    :param new_width: resize后的新图片的宽
    :param new_height: resize后的新图片的长
    :param out_image: 保存resize后的新图片的地址
    :param resize_mode: 用于resize的cv2中的模式
    :return: resize后的新图片
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

def clip_pic(img, rect):
    '''
    :param img: 输入的图片
    :param rect: rect矩形框的4个参数
    :return: 输入的图片中相对应rect位置的部分 与 矩形框的一对对角点和长宽信息
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


def IOU(vertice1, vertice2):
    '''
    用于计算两个矩形框的IOU = area of overlap / area of union
    :param ver1: 第一个矩形框(pro_rect): [xmin, ymin, xmax, ymax, w, h]
    :param vertice2: 第二个矩形框(ref_rect): [xmin, ymin, xmax, ymax]
    :return: 两个矩形框的IOU值
    '''
    vertice1 = [int(e) for e in vertice1]
    vertice2 = [int(e) for e in vertice2]
    area_overlap = if_intersection(vertice1[0], vertice1[1], vertice1[2], vertice1[3], vertice2[0], vertice2[1], vertice2[2], vertice2[3])
    # # 如果有交集，计算IOU
    if area_overlap:
        area_1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1]) 
        area_2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1]) 
        iou = float(area_overlap) / (area_1 + area_2 - area_overlap)
        return iou
    return 0





def if_intersection(xmin_a, ymin_a, xmax_a, ymax_a, xmin_b, ymin_b, xmax_b, ymax_b):
    if_intersect = False
    # 通过四条if来查看两个方框是否有交集。如果四种状况都不存在，我们视为无交集
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        # 在有交集的情况下，我们通过大小关系整理两个方框各自的四个顶点， 通过它们得到交集面积
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

def image_rect_proposal(img):
    '''
    输入要进行候选框提取的图片
    利用图片的各像素点的特点进行候选框的提取，由于候选框数量太多且针对不同的问题背景所需要的候选框的尺寸是不一样的
    因此要经过一系列的规则加以限制来进一步减小特征框的数量
    '''
    # python的selective search函数
    imgs, regions = selectivesearch.selective_search(img, scale=500, sigma=0.5, min_size=500)
  #  print("count: ",len(regions))
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        
        # 剔除重复的方框
        if r['rect'] in candidates:
            continue
        # # 剔除太小的方框
        # if r['size'] < 1000:
        #     continue
        # if (r['rect'][2] * r['rect'][3]) < 1000:
        #     continue
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
        images.append(img_float)
        vertices.append(proposal_vertice)
    # print(len(images))
    return images, vertices



def show_rect(img_path, regions, message):
    '''
    :param img_path: 要显示的原图片
    :param regions: 要在原图片上标注的矩形框的参数
    :param message: 在矩形框周围添加的信息
    :return: 
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y, x1, y1 ,w, h in regions:
        x, y, x1, y1 =int(x),int(y),int(x1),int(y1)
        rect = cv2.rectangle(
            img,(x, y), (x1, y1), (0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message, (x+20, y+40),font, 1,(255,0,0),2)
    plt.imshow(img)
    plt.show()


# 测试候选框
def main():
    
    img_path = "images/2008_004711.jpg"
    img = cv2.imread(img_path)
    imgs, verts =  image_rect_proposal(img)
    # print(verts)
    show_rect(img_path, verts, ' ')

if __name__ == '__main__':
    main()


