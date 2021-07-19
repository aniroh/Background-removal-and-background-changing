import cv2
import numpy as np
# cap = cv2.VideoCapture(0)
import time
import glob

#background removal function

    # font = cv2.FONT_HERSHEY_PLAIN

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# print(classes)
image_list = []
for filename in glob.glob('.\car' + '/*.jpeg'):
    image_list.append(filename)
# Loading image
# img = cv2.imread("t4.webp") #this line of code for testing purposes, input image
for j in image_list:
    img = cv2.imread(j) #this line of code used for taking live image from pc
    img = cv2.resize(img, None, fx=1.0, fy=1.0)
    img_h=img.shape[0]
    img_w=img.shape[1]
    dim1=(img_w,img_h)
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    height, width, channels = resized.shape


    # Detecting objects
    blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # print(len(outs[0]))

    # Showing informations on the screen
    # class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                # class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    #background removal
    s1=time.time()

    for i in range(len(boxes)):
        # if True: 
        if i in indexes:
            x, y, w, h = boxes[i]
            mask = np.zeros(resized.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            # bgdModel = cv2.imread('bgimage.png')
            fgbModel = np.zeros((1, 65), np.float64)
            # extract the bounding box coordinates
            rect = (x, y, w, h)
            print(i, rect)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), 2)
            cv2.grabCut(resized, mask, rect, bgdModel, fgbModel, 1, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            break
    img1 = resized * mask2[:, :, np.newaxis]
    img1=cv2.resize(img1,dim1, interpolation = cv2.INTER_AREA)
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(img, img, mask=img1)
    fname=j[:-4]+'out.jpg'
    cv2.imwrite(fname,masked)
    
    s2=time.time()
    print(s2-s1)

    #background change
    bg= img - masked
    bg[np.where((bg > [0,0,0]).all(axis = 2))] =[1,1,1]
    # bg = cv2.bitwise_not(bg)/255
    bg1 = cv2.imread('bgimage.png')
    bg1 = cv2.resize(bg1,dim1, interpolation = cv2.INTER_AREA)
    bg1 = bg1*bg
    final=bg1+masked
    fname1=j[:-4]+'output.jpg'
    cv2.imwrite(fname1,final)
    # cv2.imshow("Image", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
