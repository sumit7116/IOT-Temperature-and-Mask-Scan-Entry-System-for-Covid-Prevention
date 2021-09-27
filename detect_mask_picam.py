# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from threading import Timer
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pyttsx3

import RPi.GPIO as gpio
import picamera
import time
 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formatdate
COMMASPACE = ', '
from email import encoders
 
from smbus2 import SMBus
from mlx90614 import MLX90614

fromaddr = "Enter Your Email"    # change the email address accordingly
toaddr = "Enter Your Email"
 
mail = MIMEMultipart()
 
mail['From'] = fromaddr
mail['To'] = toaddr
mail['Subject'] = "Temperature value exceed alert"
body = "Please find the attached image"
 
data=""


engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.setProperty('voice', 'english_rp+f3')
engine.setProperty('volume',1.0)


def sendMail(data):
    mail.attach(MIMEText(body, 'plain'))
    print (data)
    dat='%s.jpg'%data
    print (dat)
    attachment = open(dat, 'rb')
    image=MIMEImage(attachment.read())
    attachment.close()
    mail.attach(image)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "Enter Mail Password here")
    text = mail.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    
def capture_image():
    engine.say("your temparature")
    engine.runAndWait()
    engine.say("is high")
    engine.runAndWait()
    engine.say("Sending data to authority")
    engine.runAndWait()
    camera = picamera.PiCamera()
    camera.rotation=0
    camera.awb_mode= 'auto'
    camera.brightness=55
    data= time.strftime("%d_%b_%Y|%H:%M:%S")
    camera.start_preview()
    time.sleep(5)
    print (data)
    camera.capture('%s.jpg'%data)
    camera.stop_preview()
    time.sleep(1)
    sendMail(data)
    
 
def get_temp():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    print ("Ambient Temperature :", round(sensor.get_ambient()))
    print ("Object Temperature :", round(sensor.get_object_1()))
    temp = sensor.get_object_1()
    temp3 = str(sensor.get_ambient()).split('.')[0]
    temp4 = str(sensor.get_object_1()).split('.')[0]
    temp5 = int(temp4)+1
    temp6 = int(temp4)-1
    print(temp5)
    print(temp3)
    bus.close()
    if temp>17:
        capture_image()
    elif int(temp3)==int(temp4):
        engine.say("Please place your hand over temprature sensor")
        engine.runAndWait()
        t=Timer(5.0,get_temp)
        t.start()
        print("Same temp")
    elif int(temp3)==int(temp5):
        engine.say("Please place your hand over temprature sensor")
        engine.runAndWait()
        t=Timer(5.0,get_temp)
        t.start()
        print("same")
    elif int(temp3)==int(temp6):
        engine.say("Please place your hand over temprature sensor")
        engine.runAndWait()
        t=Timer(5.0,get_temp)
        t.start()
        print("same")
    else:
        engine.say("You are safe! please Welcome")
        engine.runAndWait()
        #t=Timer(5.0,detect_mask(mask_detect_state=False))
        #t.start()

def detect_mask(mask_detect_state=False):        
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    #print("[INFO] loading face detector model...")
    #engine.say("Loading")
    engine.runAndWait()
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    #engine.say("Detecting Mask")
    #engine.runAndWait()
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    engine.say("Detecting Mask")
    engine.runAndWait()
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if withoutMask > mask:
                label = "No Face Mask Detected"
                color = (0, 0, 255)
                engine.say("Mask Not Detected")
                engine.runAndWait()
                mask_detect_state=False
            else:
                label = "Thank You. Mask On."
                color = (0, 255, 0)
                engine.say("Mask Detected")
                engine.runAndWait()
                mask_detect_state=True
                
            
            #label = "Thank you" if mask > withoutMask else "Please wear your face mask"
            #color = (0, 255, 0) if label == "Thank you" else (0, 0, 255)

            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX-50, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Face Mask Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break
        
        if mask_detect_state==True:
            engine.say("Place your hand over temparature sensor")
            engine.runAndWait()
            cv2.destroyAllWindows()
            vs.stop()
            t = Timer(5.0, get_temp)
            t.start()
            break



detect_mask()

# do a bit of cleanup

