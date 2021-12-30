import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

def pre_process_image(img, morph_size):
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    return pre


def find_text_boxes(pre, min_text_height_limit=8, max_text_height_limit=40):
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    table_cells = [list(sorted(tb)) for tb in table_cells]
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines

def get_bbox(table_cells):
    point1 = table_cells[0][0][:2]
    point2 = np.add(table_cells[-1][-1][:2],table_cells[-1][-1][2:4])
    bbox = (point1[0], point1[1], (point2[0]-point1[0]), (point2[1] - point1[1]))

    return bbox

def detection(img, morph_size):
    pre_processed = pre_process_image(img, morph_size)
    text_boxes = find_text_boxes(pre_processed)
    cells = find_table_in_boxes(text_boxes)

    vis = img.copy()
    bbox = get_bbox(cells)

    return bbox

def multi_images_detection(folderPath):
    for currentFileName in os.listdir(folderPath):
        imgPath = os.path.join(folderPath, currentFileName)
        img = cv2.imread(imgPath)
        bbox_detect = detection(img)
        x,y,w,h = bbox_detect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,"detect", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        # plt.imshow(img)
        # plt.show()

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
    morph_size=(20, 10)
    print(morph_size)
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
        bbox_detect = detection(img, morph_size)

        x,y,w,h = bbox_actual
        bbox_actual = (x,y,x+w,y+h)
        x,y,w,h = bbox_detect
        bbox_detect = (x,y,x+w,y+h)
        iou = IOU(bbox_detect, bbox_actual)   
        his_eval.append((imagePath, iou, bbox_detect, tuple(bbox_actual)))

    return his_eval   

if __name__ == "__main__":
    his = evaluation()
    for h in his:
        iou = str(round(h[1], 2))
        img_path = h[0]
        img = cv2.imread(img_path)
        bbox_detect = h[2]
        bbox_actual = h[3]
        x,y,x2,y2 = bbox_detect
        cv2.rectangle(img,(x,y),(x2,y2),(0,255,0),2)
        cv2.rectangle(img, (x, y-45), (x+190,y), (0,0,0), -1)
        cv2.putText(img,"detect", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
        x,y,x2,y2 = bbox_actual
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(img,(x,y),(x2,y2),(255,0,0),2)
        cv2.rectangle(img, (x, y-45), (x+190,y), (0,0,0), -1)
        cv2.putText(img,"actual", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
        
        cv2.putText(img, "IOU: "+ iou, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imwrite("evaluation/solution1/"+img_path[17:len(img_path)], img)

    his = [h[1] for h in his]
    histogram = np.histogram(his, bins = (0, 0.25, 0.5, 0.75, 1))
    print(histogram[0])
    print(histogram[1])
    # multi_images_detection(folderPath = "D:/FtechWork/tableDetection/table-detection-dataset-master/images")