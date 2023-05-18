import cv2 as cv
import numpy as np
import math
import random
import serial
import time
import struct


def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def magnitude(vector):
    mag = math.sqrt(sum(pow(element, 2) for element in vector))
    return mag


def write_read(x):
    #    data = struct.pack('f', x)
    arduino.write(x.encode('utf-8'))
    #arduino.write('1.0909\n'.encode('utf-8'))
    time.sleep(0.05)
    #data = arduino.readline()
    data = arduino.readline().decode('utf-8')
    time.sleep(0.05)
    return data


# Reading Videos
capture = cv.VideoCapture(1)
#capture.set(3, 640)
#capture.set(4, 480)
counter = 0
angle = 0

# lower bound and upper bound for Green color
lower_bound = np.array([40, 60, 70])
upper_bound = np.array([70, 255, 255])
#define kernel size
kernel = np.ones((7, 7), np.uint8)
# create connection with arduino
arduino = serial.Serial('COM5', 115200, timeout=.1)
time.sleep(1)  # give the connection a second to settle
while True:
    isTrue, frame = capture.read()
    frame = cv.resize(frame,(1920,1080),fx=0,fy=0, interpolation=cv.INTER_CUBIC)
    frame = rescaleFrame(frame, scale=0.6)
    #dimensions = frame.shape
    #ROWS = frame.shape[0]
    #COLS = frame.shape[1]

    # if cv.waitKey(20) & 0xFF==ord('d'):
    # This is the preferred way - if `isTrue` is false (the frame could
    # not be read, or we're at the end of the video), we immediately
    # break from the loop.
    if (isTrue):
        
        blank = np.zeros(frame.shape, dtype='uint8')
        counter += 1
        #frame = cv.GaussianBlur(frame, (7, 7), cv.BORDER_DEFAULT)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # find the colors within the boundaries
        mask = cv.inRange(hsv, lower_bound, upper_bound)
        # Remove unnecessary noise from mask
        #mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        contours, hierarchy = cv.findContours(
            mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cont_output = cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
        x_points = []
        y_points = []
        j = 0

        # Identify the centers of the stickers
        for i in contours:
            M = cv.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x_points.append(cx)
                y_points.append(cy)
                #cv.drawContours(frame, [i], -1, (0, 255, 0), 2)
                #cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                #cv.putText(frame, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                #print(f"x: {x_points[j]} y: {y_points[j]}")
                j += 1
    
        for i in range(len(x_points)):
            if i == 0:
                cont_output[y_points[i], x_points[i]] = 0, 0, 255 # red
            if i == 1:
                cont_output[y_points[i], x_points[i]] = 0, 0, 255 # red
            if i == 2:
                cont_output[y_points[i], x_points[i]] = 255, 255, 255 # white
            if i == 3:
                cont_output[y_points[i], x_points[i]] = 255, 255, 255 # white
        cv.imshow('Centroids', cont_output)
        
        #print(counter)
        if (len(x_points) == 4) and ((counter % 30) == 0):
            
            counter = 1
            # insertion sort code to sort from least to greatest x-values
            for i in range(1, len(x_points)):
                key = x_points[i]
                ykey = y_points[i]
                j = i - 1
                while (j >= 0) and (key < x_points[j]):
                    x_points[j+1] = x_points[j]
                    y_points[j+1] = y_points[j]
                    j = j - 1
                x_points[j+1] = key
                y_points[j+1] = ykey
            '''  
            # insertion sort check 
            print("x and y in order")
            for i in range(len(x_points)):
                print(f"x: {x_points[i]} y: {y_points[i]}")
            '''

            # Create vectors from pairs of points then calculate angle between vectors.
            vector1 = np.array([x_points[1]-x_points[0], y_points[1]-y_points[0]])
            vector2 = np.array([x_points[3]-x_points[2], y_points[3]-y_points[2]])
            angle = np.arccos(np.dot(vector1, vector2) / (magnitude(vector1)*magnitude(vector2)))*(180/math.pi)
            if (y_points[3] > y_points[1]):
                angle = -1*angle
            angle = '{:.3f}'.format(angle)+'\n'

            # Send the data to an arduino using PySerial
            value = write_read(angle)

            print(value)  # printing the value
            cv.putText(frame, "Angle: "+angle[:-1], (500, 25), cv.FONT_HERSHEY_TRIPLEX, 0.8, (255,0, 0), 1) 

        elif (len(x_points) != 4):
            angle = 0
        if (angle != 0):
            cv.putText(frame, "Angle: "+angle[:-1], (490, 25),cv.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 1)
 
        cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
