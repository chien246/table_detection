import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def detection(img):
    #thresholding the image to a binary image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #inverting the image
    img_bin = 255-img_bin
    cv2.imwrite('cv_inverted.png',img_bin)

    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite('vertical.jpg', vertical_lines)

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    minPoint = (vertical_lines.shape[1], vertical_lines.shape[0])
    maxPoint = (0,0)

    for i in range(vertical_lines.shape[0]):
        for j in range(vertical_lines.shape[1]):
            if((int(vertical_lines[i][j]) + int(horizontal_lines[i][j])) > 100):
                if(j < minPoint[0] and i < minPoint[1]):
                    minPoint = (j,i)
                if(j > minPoint[0] and i > minPoint[1]):
                    maxPoint = (j,i)
    bbox = (minPoint[0], minPoint[1], maxPoint[0] - minPoint[0], maxPoint[1] - minPoint[1])
    return bbox

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluation():
    his_eval = [] #lưu các chỉ số iou của mỗi lần so sánh
    fileObject = open("dataFwork/annotations/instances_default.json", "r")
    jsonContent = fileObject.read()
    iList = json.loads(jsonContent)["images"]
    aList = json.loads(jsonContent)["annotations"]

    preId = 0
    
    for ano in aList:
        idImage = ano["image_id"]
        bbox_actual = ano["bbox"]
        imagePath = list((filter(lambda id: id["id"] == idImage, iList)))[0]["file_name"]
        imagePath = "dataFwork/images/" + imagePath

        img = cv2.imread(imagePath)
        bbox_detect = detection(img)
        iou = IOU(bbox_detect, bbox_actual)   
        his_eval.append((imagePath, iou, bbox_detect, tuple(bbox_actual)))

    return his_eval 

if __name__ == "__main__":
    #read your file
    img = cv2.imread("dataFwork/images/1_Dap_an_bai_hoc_Tinh_don_dieu_cua_ham_so_1_2.jpeg")

    his = evaluation()
    for h in his:
        # print(h)
        if(h[1] > 0):
            print(h)
            img_path = h[0]
            img = cv2.imread(img_path)
            bbox_detect = h[2]
            bbox_actual = h[3]
            x,y,w,h = bbox_detect
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"detect", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
            x,y,w,h = bbox_actual
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"actual", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
            plt.imshow(img)
            plt.show()

