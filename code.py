from ultralytics import YOLO
import cv2
import numpy as np

cap = cv2.VideoCapture("Student Stops Bus After Driver Passes Out.mp4")

item_list = {1: 'person',
             2: 'bicycle',
             3: 'car',
             4: 'motorcycle',
             5: 'airplane',
             6: 'bus',
             7: 'train',
             8: 'truck',
             9: 'boat',
             10: 'traffic light',
             11: 'fire hydrant',
             12: 'stop sign',
             13: 'parking meter',
             14: 'bench',
             15: 'bird',
             16: 'cat',
             17: 'dog',
             18: 'horse',
             19: 'sheep',
             20: 'cowphant',
             22: 'bear',
             23: 'zebra',
             24: 'giraffe',
             25: 'backpack',
             26: 'umbrella',
             27: 'handbag',
             28: 'tie',
             29: 'suitcase',
             30: 'frisbee',
             31: 'skis',
             32: 'snowboard',
             33: 'sports ball',
             34: 'kite',
             35: 'baseball bat',
             36: 'baseball glove',
             37: 'skateboard',
             38: 'surfboard',
             39: 'tennis racket',
             40: 'bottle',
             41: 'wine glass',
             42: 'cup',
             43: 'fork',
             44: 'knife',
             45: 'spoon',
             46: 'bowl',
             47: 'banana',
             48: 'apple',
             49: 'sandwich',
             50: 'orange',
             51: 'broccoli',
             52: 'carrot',
             53: 'hot dog',
             54: 'pizza',
             55: 'donut',
             56: 'cake',
             57: 'chair',
             58: 'couch',
             59: 'potted plant',
             60: 'bed',
             61: 'dining table',
             62: 'toilet',
             63: 'tv',
             64: 'laptop',
             65: 'mouse',
             66: 'remote',
             67: 'keyboard',
             68: 'cell phone',
             69: 'microwave',
             70: 'oven',
             71: 'toaster',
             72: 'sink',
             73: 'refrigerator',
             74: 'book',
             75: 'clock',
             76: 'vase',
             77: 'scissors',
             78: 'teddy bear',
             79: 'hair dryer',
             80: 'toothbrush'}

model = YOLO('yolov8n.pt') 

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

def count(item, items):
    return str(sum([1 for item_ in items if item_ == item]))
a = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if a % 5 ==0:
        z = model.predict(frame)
        items_ = list(z[0].boxes.cls)
        items = [int(item) for item in items_]
    # Draw the video output rectangle
    cv2.rectangle(frame, (0, 0), (225,75+ 15 * len(set(items))), (245, 117, 16), -1)
    
    # Draw the class other in the video output rectangle
    
    for i, item in enumerate(list(set(items))):
        cv2.putText(frame, count(item,items) + "  `" +item_list[item + 1], (15, 12 + 20 * (i + 1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Write the frame to the output video file
    out.write(frame)
    
    a += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release everything and close the windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Play the output video file
cap = cv2.VideoCapture('output.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Output Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()