import cv2 as cv
import pandas as pd
import numpy as np
import os
from gtts import gTTS
from cv2 import waitKey 

net = cv.dnn.readNet('yolov3-tiny.weights', 'yolov3_tiny.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

#CODE FOR COLOUR DETECTION
def webcam():
    
    cam = cv.VideoCapture(0)
    count = 0
    while True:
        ret, img = cam.read()
        cv.imshow("Test", img)
        if not ret:
            break

        k=cv.waitKey(1)

        if k%256 == 27:
            print("Camera closed ")
            cam.release()
            cv.destroyAllWindows()

            break

        if k%256==32:
            #For Space key

            print("Image "+str(count)+"saved")
            file='Color Detection'+str(count)+'.jpg'
            cv.imwrite(file, img)
            count +=1
            break

    cam.release()
    cv.destroyAllWindows()
    return file

def colour_name(k):
    data= pd.read_csv(r"wikipedia_color_names.csv")
    minimum = 1000
    for i in range(1,len(data)):
        d = abs(int(k[2])- int(data.loc[i,'Red (8 bit)'])) + abs(int(k[1])-int(data.loc[i,'Green (8 bit)'])) + abs(int(k[0]) - int(data.loc[i,'Blue (8 bit)']))
        if d <= minimum:
            minimum = d
            cname = data.loc[i,'Name']
    print(cname)
    speech(cname)

def click_event(event,x,y,flags,params):
    if event == cv.EVENT_LBUTTONDOWN:
        colour_name(img[y][x])
    if event == cv.EVENT_RBUTTONDOWN:
        colour_name(img[y][x])
        

def speech(name):
    mytext = name
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=True)
    myobj.save("Colourname.mp3")
    os.system("start Colourname.mp3")
    cv.imshow("Read",img)
    
#CODE FOR OBJECT DETECTION

def object_detection(img):

    height, width, _ = img.shape
    blob = cv.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
            


    cv.imshow('Image', img)

    

os.system('cls')

print("_______________________COLOUR DETECTION AND OBJECT DETECTION CODE___________________________")
print("\n Enter your choice:")
print("1. COLOUR DETECTION")
print("2. OBJECT DETECTION")
print("_______________________________________________________________________")
ch = int(input("Enter your Choice: "))

os.system('cls')

if ch == 1:
    print("_______________________COLOUR DETECTION___________________________")
    print("\n Enter your choice:")
    print("1. Upload a Image")
    print("2. Take a Picture")
    print("_______________________________________________________________________")
    ch2 = int(input("Enter your Choice: "))

    if ch2 == 1:
        print("____________________________Upload Your Image______________________")
        print("\nEnter the Path of The image ")
        path = input("\n\tPath:")

    elif ch2 == 2:
        print("____________________________Take a Picture_________________________")
        print("\n\n\t\t********Enter Space To Take a Picture********")
        print("\n\t Press Any Key")
        waitKey(0)
        path = webcam()

    img = cv.imread(path) 
    cv.imshow("Read",img)
    cv.setMouseCallback("Read",click_event)

elif ch == 2:
    print("_______________________OBJECT DETECTION CODE___________________________")
    print("\n Enter your choice:")
    print("1. Upload a Image")
    print("2. Switch on your camera")
    print("_______________________________________________________________________")
    ch2 = int(input("Enter your Choice: "))
    if ch2 == 1:
        print("____________________________Upload Your Image______________________")
        print("\nEnter the Path of The image ")
        path = input("\n\tPath:")
        image = cv.imread(path) 
        object_detection(image)
    elif ch2 == 2:
        print("____________________________Switch on your Camera_________________________")
        cap = cv.VideoCapture(0)
        while True:
            _, image = cap.read()
            object_detection(image)
            key = cv.waitKey(1)
            if key==27:

                cv.destroyAllWindows()
                break
        cap.release()    
            
cv.waitKey(0)
cv.destroyAllWindows()
