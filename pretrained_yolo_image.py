# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:00:44 2022

@author: kerolos
"""

import cv2
import numpy as np

# image to detect:
img=cv2.imread('D:/ahmed el sallab/yolo/the-street-rev.jpg')

# extracting the height and width of the image:
img_height=img.shape[0]
img_width=img.shape[1]

#convert to blob to pass the image to the model:

img_blob=cv2.dnn.blobFromImage(img,0.003922, (416,416), swapRB=True , crop=False)


# set the 80 class label:

class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


#declare a list of colours for the differenrt objects as an array
class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors=[np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors=np.array(class_colors)
class_colors=np.tile(class_colors,(16,1))

len(class_colors[0])


# now let us load our model configuratiions and weights:

yolo_model=cv2.dnn.readNetFromDarknet('D:/ahmed el sallab/yolo/yolov3.cfg','D:/ahmed el sallab/yolo/yolov3.weights')

# get all model layers:
yolo_layers=yolo_model.getLayerNames()

len(yolo_layers)

yolo_output_layer=[yolo_layers[yolo_layer[0]-1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]



yolo_model.setInput(img_blob)
obj_detection_layers=yolo_model.forward(yolo_output_layer)

for obj_detection_layer in obj_detection_layers:
    for obj_detection in obj_detection_layer:
        all_score=obj_detection[5:]
        predicted_class_ID=np.argmax(all_score)
        prediction_confedince =all_score[predicted_class_ID]
        
        if prediction_confedince>0.2:
            predicted_class_label=class_labels[predicted_class_ID]
            bounding_box=obj_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            (box_center_x_pt,box_center_y_pt,box_width_pt,box_height_pt)=bounding_box.astype("int")
            
            start_x_pt=int(box_center_x_pt -(box_width_pt)/2 )
            start_y_pt=int(box_center_y_pt -(box_height_pt)/2 )
            end_x_pt = start_x_pt+box_width_pt
            end_y_pt=start_y_pt+box_height_pt
            
            box_color=class_colors[predicted_class_ID]
            
            box_color=[int(c) for c in box_color]
            
            
            predicted_class_label="{} {:.2f}%".format(predicted_class_label,prediction_confedince*100)
            print("predected_class_label {}".format(predicted_class_label))
            
            
            cv2.rectangle(img,(start_x_pt,start_y_pt),(end_x_pt,end_y_pt),box_color,3)
            cv2.putText(img,predicted_class_label,(start_x_pt,start_y_pt-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            
cv2.imshow("object_detected",img)
            
            
            




