#Chad Williamson
#CS 435 Fall 2019

import os
import numpy as np
import cv2


videoName = "flash.avi"

def main():
    #
    frameArray = []
    contourArray = []
    
    getVideoFrames(contourArray)
    framePersistence(contourArray, frameArray)
    writeFrames(frameArray)
    print("Complete")
    
    
    
    
    
def getVideoFrames(contourArray):
    path = 'C:/Users/Chad/Documents/IUPUI/Fall 2019/CSCI43500 Multimedia Systems/Traffic_Signal_Detection/imageData/'
    sourceVideo = cv2.VideoCapture(videoName)
    
    total = int(sourceVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    mask = cv2.imread('mask.png',0)
    
    frameCounter = 0
    validFrame = 1
    while (validFrame): 

        
        #image variable is the frame of the video, validFrame is loop condition
        validFrame, image = sourceVideo.read()
        
        
        if(validFrame):
            height, width, layers = image.shape
            newImage = image.copy()

            
            #create grayscale
            grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #add mask to block portion of video that is not needed
            masked = cv2.bitwise_and(grayscaleImage,mask)
            
            
            #apply thresholding
            ret, thresh = cv2.threshold(masked, 245, 255, cv2.THRESH_BINARY)
            
            #set kernel of size 3 by 3
            kernel = np.ones((3,3),np.uint8)
            
            #apply morphological transformations
            erosion = cv2.erode(thresh,kernel,iterations = 1)
            
            #set elliptical kernel of size 3 by 3
            Ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            dilate = cv2.dilate(erosion,Ekernel,iterations = 1)
            blackhat = cv2.morphologyEx(dilate, cv2.MORPH_BLACKHAT, kernel)
            dilate = cv2.bitwise_xor(dilate,blackhat)

            # Find the edges in the image using canny edge detector                        
            edges = cv2.Canny(dilate, 180, 200)  
            
            #contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

            #temporary arrays to hold frame contours
            redArray = []
            greenArray = []
            contourCount = 0
            
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 00.01*cv2.arcLength(contour,True),True)
                area = cv2.contourArea(contour)
                x,y,w,h = cv2.boundingRect(contour)
                
                #scalar defines the minimum cutoff value. as contour y value approaches the lower part of the frame, the scalar is smaller
                scalar = (((((320 - y) / 320)  * 2) + 1) * 16)
                if (y >= 320):
                    scalar = 16
                #allow if within size threshold
                if ((cv2.arcLength(contour,True) > scalar) and (cv2.arcLength(contour,True) < 60)):
                    if (area > cv2.arcLength(contour, True)):
                        bloomSize = 4;
                        if (x + w + bloomSize >= (width)) or (y + h + bloomSize >= (height)):
                            bloomSize = 0
                            x  -= 1
                            y -= 1
                        xInc = x - bloomSize
                        yInc = y - bloomSize
                        sizeToAvg = (w + (bloomSize * 2)) * (h + (bloomSize * 2))
                        avgR = 0; avgB = 0; avgG = 0
                        while (xInc <= (x + w + bloomSize)):
                            while (yInc <= (y + h + bloomSize)):
     
                                avgB += image[yInc,xInc][0]
                                avgG += image[yInc,xInc][1]
                                avgR += image[yInc,xInc][2]
                                yInc += 1
                            yInc = y - bloomSize
                            xInc += 1
                        avgR = avgR / sizeToAvg 
                        avgG = avgG / sizeToAvg 
                        avgB = avgB / sizeToAvg
                        aspectRatio = float(w)/h
                        if (aspectRatio > .7) and (aspectRatio < 1.46):
                            #calculate HUE
                            hR = float(avgR / 255)
                            hG = float(avgG / 255)
                            hB = float(avgB / 255)
                            hue = 0
                            lightness = (avgR + avgB + avgG) / 3
                            saturation = 0
                            if (hR > hG and hR > hB):
                                if (hG > hB):
                                    hue = (avgG - avgB)/(avgR-avgB)
                                    saturation = (255 * (1 - ((3 * (avgB)) / (avgR + avgG + avgB))));
                                else:
                                    hue = (avgG - avgB)/(avgR-avgG)
                                    saturation = (255 * (1 - ((3 * (avgG)) / (avgR + avgG + avgB))));
                            elif (hG > hR and hG > hB):
                                if (hR > hB):
                                    hue = 2.0 + (avgB-avgR)/(avgG-avgB)
                                    saturation = (255 * (1 - ((3 * (avgB)) / (avgR + avgG + avgB))));
                                else:
                                    hue = 2.0 + (avgB-avgR)/(avgG-avgR)
                                    saturation = (255 * (1 - ((3 * (avgR)) / (avgR + avgG + avgB))));
                            else:
                                if (hG > hR):
                                    hue = 4.0 + (avgR-avgG)/(avgB-avgR)
                                    saturation = (255 * (1 - ((3 * (avgR)) / (avgR + avgG + avgB))));
                                else:
                                    hue = 4.0 + (avgR-avgG)/(avgB-avgG)
                                    saturation = (255 * (1 - ((3 * (avgG)) / (avgR + avgG + avgB))));
                            hue = hue * 60
                            if (hue < 0):
                                hue = hue + 360
                            #end of hue calculation
                            contourCount += 1
                            
                            #RED values
                            if(avgR > avgG) and (avgR > avgB) and ((avgR - avgG) > 26):
                                if (hue < 20 or hue > 340):
                                    if (x + w + 4 >= width) or (y + h + 4 >= height):
                                        q = 0
                                    else:
                                        if (determineLight("red", image[(y-4):(y+h+4), (x-4):(x+w+4)], frameCounter, contourCount)):
                                            redArray.append(contour)
                            
                            #GREEN values
                            elif ((avgG > avgB and avgG > avgR) and (lightness < 233)):
                                    if (x + w + 4 >= width) or (y + h + 4 >= height):
                                        q = 0
                                    else:
                                        if (determineLight("green", image[(y-4):(y+h+4), (x-4):(x+w+4)], frameCounter, contourCount)):
                                            greenArray.append(contour)

            #send to contourArray
            contourArray.append([image, redArray, greenArray])
            print("Processing: " + str(round((frameCounter / total) * 100)) + "%")
            frameCounter += 1


def determineLight(color, image, frameNum, contourNum):
    path = 'C:/Users/Chad/Documents/IUPUI/Fall 2019/CSCI43500 Multimedia Systems/Traffic_Signal_Detection/imageData/'
    height, width, layers = image.shape
    if (height == 0 or width == 0):
        return False
    dupImage = image.copy()
    detBool = True
    detCount = 0
    centerVal = image[int(width / 2),int(height / 2)]
    x = 0; y = 0; q = 0 
    while (y < height):
        while (x < width):
            if (y == 0) or (y == height - 1) or (x == 0) or (x == width - 1):
                #if a border pixel is less than center value, darker than center val
                if (image[y,x][0] < centerVal[0] and image[y,x][1] < centerVal[1] and image[y,x][2] < centerVal[2]):
                    q = 0
                else:
                    detBool = False
            x +=1 
        x = 0
        y += 1
    
    if (not(detBool)):
        image = cv2.bilateralFilter(image, 5, 175, 175)
        image = cv2.Canny(image, 75, 200)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, height / 2, param1=100, param2=12, minRadius=int(height * .3), maxRadius=int(height * .96))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(image, center, radius, (255, 0, 255), 3)
            detBool = True

    return detBool

def framePersistence(pArray, fArray):
            
    #set to 2 and read to (end of array - 2) to allow for reading future and previous frame data
    frameNum = 2
    while (frameNum < (len(pArray) - 2)):
        #current frame information
        currentImage = pArray[frameNum][0]
        currentRed = pArray[frameNum][1]
        currentGreen = pArray[frameNum][2]
        
        #previous frame information
        previousRed = pArray[frameNum - 1][1]
        previousGreen = pArray[frameNum - 1][2]
        
        #previously previous frame information
        furthestPreviousRed = pArray[frameNum - 2][1]
        furthestPreviousGreen = pArray[frameNum - 2][2]
        
        #next frame information
        nextRed = pArray[frameNum + 1][1]
        nextGreen = pArray[frameNum + 1][2]
        
        #next next frame information
        furthestNextRed = pArray[frameNum + 2][1]
        furthestNextGreen = pArray[frameNum + 2][2]
        
        drawR = 0
        drawG = 0
        #red contours
        for contour in currentRed:

            #current
            x,y,w,h = cv2.boundingRect(contour)
            drawR = .5
            
            #look at all contours in previous frame. was there one close to this contour?
            for pastContour in previousRed:
                px, py, pw, ph = cv2.boundingRect(pastContour)
                if (abs((px - x) < 5)) and (abs((py - y) < 5)):
                    drawR += .3
                    
            #look at all contours in 2nd previous frame. was there one close to this contour?
            for past2Contour in furthestPreviousRed:
                px, py, pw, ph = cv2.boundingRect(past2Contour)
                if (abs((px - x) < 5)) and (abs((py - y) < 5)):
                    drawR += .2
                    
            #do the same for all contours in next frame. is there one close to this contour?
            for futureContour in nextRed:
                fx, fy, fw, fh = cv2.boundingRect(futureContour)
                if (abs((fx - x) < 5)) and (abs((fy - y) < 5)):
                    drawR += .3
                    
            #do the same for all contours in next next frame. is there one close to this contour?
            for future2Contour in furthestNextRed:
                fx, fy, fw, fh = cv2.boundingRect(future2Contour)
                if (abs((fx - x) < 5)) and (abs((fy - y) < 5)):
                    drawR += .2
            #draw
            if (drawR > 1):
                cv2.rectangle(currentImage, (x - 5, y - 5), (x+w+5, y+h+5), (0, 0, 255), 2)
                
                
        for contour in currentGreen:
            #current
            x,y,w,h = cv2.boundingRect(contour)
            drawG = .5
            
            #look at all contours in previous frame. was there one close to this contour?
            for pastContour in previousGreen:
                px, py, pw, ph = cv2.boundingRect(pastContour)
                if (abs((px - x) < 5)) and (abs((py - y) < 5)):
                    drawG += .3
                    
            #look at all contours in 2nd previous frame. was there one close to this contour?
            for past2Contour in furthestPreviousGreen:
                px, py, pw, ph = cv2.boundingRect(past2Contour)
                if (abs((px - x) < 5)) and (abs((py - y) < 5)):
                    drawG += .2
            #do the same for all contours in next frame. is there one close to this contour?
            for futureContour in nextGreen:
                fx, fy, fw, fh = cv2.boundingRect(futureContour)
                if (abs((fx - x) < 5)) and (abs((fy - y) < 5)):
                    drawG += .3
                    
            #do the same for all contours in next next frame. is there one close to this contour?
            for future2Contour in furthestNextGreen:
                fx, fy, fw, fh = cv2.boundingRect(future2Contour)
                if (abs((fx - x) < 5)) and (abs((fy - y) < 5)):
                    drawG += .2
                    
            #draw
            if (drawG > 1):
                cv2.rectangle(currentImage, (x - 5, y - 5), (x+w+5, y+h+5), (0, 255, 0), 2)

        frameNum += 1
        fArray.append(currentImage)
    
def writeFrames(frameArray):
    size = (1280,720)
    out = cv2.VideoWriter('projectVideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(frameArray)):
        out.write(frameArray[i]) 
        print("Writing: " + str(round((i / len(frameArray)) * 100)) + "%")
    out.release()


#at end
if __name__== "__main__":
  main()
