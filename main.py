#This is a demonstration of my BSc. Project (An Intelligent Software That Aids Surveillance)

#Libraries -->
import numpy as np
import cv2 as cv
import os
import sys
import pymediainfo
import time

#Commencing the code

#This handles the intelligent surveillance class
class Surveillance:
    def __init__(self) -> None:
        self.boxes: list = []
        self.confidences: list = []
        self.classIds: list = []
        self.font = cv.FONT_HERSHEY_PLAIN
        self.cfgPath: str = f"{os.getcwd()}\models\yolo.cfg"
        self.modelPath: str = f"{os.getcwd()}\models\yolov3.weights"
        self.minConfidence: float = 0.8
        self.classes: list = []
        with open("assets/classes.txt", "r") as f:
            self.classes: list[str] = f.read().splitlines()
        
        #print(f"Colors: {self.colors}")
        self.getChoice()

    #This function gets the user's choice of file input
    def getChoice(self) -> None:
        fileStat: str = input("Enter 0 to use webcam or Enter name of the video file: ")
        if fileStat == "0":
            file = 0 
            print("webcam")
        else:
            if os.path.exists(fileStat):
                if self.checkFile(fileStat):
                    file = fileStat
                    print("Video file")
                else:
                    raise Exception("Wrong file input, upload a video file")
                    sys.exit(0)
            else:
                raise Exception("File doesn't exist, enter correct file path")
                sys.exit(0)

        print("Starting surveillance in 3 seconds...")
        time.sleep(3)
        self.detectObject(file)

    #This function check for the type of file is a video
    def checkFile(self, file: str) -> bool:
        fileInfo: object = pymediainfo.MediaInfo.parse(file)
        for track in fileInfo.tracks:
            if track.track_type == "Video":
                return True

    #This function handles the detection of the object
    def detectObject(self, file: str) -> None:
        net = cv.dnn.readNet(self.modelPath, self.cfgPath)
        print(f"Net: {net}")
        self.cap = cv.VideoCapture(file)
        colors = np.random.uniform(0, 255, size = (100, 3))
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if ret:
                height, width = image.shape[0], image.shape[1]
                blob = cv.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
                #print(f"Blob: {blob}")
                net.setInput(blob)
                outputLayersNames = net.getUnconnectedOutLayersNames()
                layerOutputs = net.forward(outputLayersNames)

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]
                        if confidence > self.minConfidence:
                           centerX = int(detection[0]*width)
                           centerY = int(detection[1]*height)
                           w = int(detection[2]*width)
                           h = int(detection[3]*height)

                           x = int(centerX - w/2)
                           y = int(centerY - h/2)

                           self.boxes.append([x, y, w, h])
                           self.confidences.append((float(confidence)))
                           self.classIds.append(classId)

            indexes: any = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.2, 0.4)

            if len(indexes) > 0 and len(indexes) < 100:
                for i in indexes.flatten():
                    x, y, w, h = self.boxes[i]
                    label = str(self.classes[self.classIds[i]])
                    confidence = str(round(self.confidences[i],2))
                    print("colors: ", len(colors))
                    #color = colors[i]
                    cv.rectangle(image, (x,y), (x+w, y+h), (21, 0, 214), 2)
                    cv.putText(image, label + " " + confidence, (x, y+20), self.font, 2, (255,255,255), 2)

            cv.imshow('Image', image)
            key = cv.waitKey(1)
            if key == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    print("An Intelligent Software that aids surveillance \n")
   
    survey: Surveillance = Surveillance()

    print("\n Successfully executed the program!")