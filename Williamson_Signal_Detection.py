#Chad Williamson

import os
import numpy as np
import cv2
import glob
import colorsys

def main():
    frameArray = []
    
    getVideoFrames(frameArray)
    writeFrames(frameArray)
    print("Complete")
    
    
    
    
    
def getVideoFrames(frameArray):
    path = 'C:/Users/Chad/Documents/IUPUI/Fall 2019/CSCI43500 Multimedia Systems/Traffic_Signal_Detection/imageData/'
    sourceVideo = cv2.VideoCapture('comp.avi')
    total = int(sourceVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    mask = cv2.imread('mask.png',0)
    frameCounter = 0
    validFrame = 1
    while (validFrame): #currently only reading the first 100 frames
    #while (validFrame and (frameCounter < 50)): #currently only reading the first 100 frames
        x = 0
        y = 0
        pxVal = 0
        newPxVal = 0
        
        #image variable is the frame of the video, validFrame is loop condition
        validFrame, image = sourceVideo.read()
        
        if(validFrame):
            height, width, layers = image.shape
            newImage = image

            
            #create grayscale
            grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                #grayscale step
            
            #add mask
            masked = cv2.bitwise_and(grayscaleImage,mask)
            
            #this will block lower portion of the image
            ###while (x < height):
            ###    if (x > height * .515):
            ###        while (y < width):
            ###            grayscaleImage[x,y] = 0
            ###            y += 1 #increment width loop condition
            ###        
            ###    y = 0 #reset width loop var
            ###    x += 1  #increment height loop condition
            
            #########################################################
            #apply thresholding
            ret, threshB = cv2.threshold(masked, 245, 255, cv2.THRESH_BINARY)
            #get a kernel
            kernel = np.ones((3,3),np.uint8)
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            erosion = cv2.erode(threshB,kernel,iterations = 1)
            #extract the background from image
            Ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            dilate = cv2.dilate(erosion,Ekernel,iterations = 1)
            #
            blackhat = cv2.morphologyEx(dilate, cv2.MORPH_BLACKHAT, kernel)
            dilate = cv2.bitwise_xor(dilate,blackhat)

            # Find the edges in the image using canny detector                        
            edgesB = cv2.Canny(dilate, 180, 200)  
            

            
            #contours
            contoursB, hierarchyB = cv2.findContours(edgesB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#contours  
            #blankImage = np.zeros((height,width,3), np.uint8)

            redArray = []
            greenArray = []
            contourCount = 0
            for contour in contoursB:
  
                #borderVal = 0
                #borderDet = 0
                approx = cv2.approxPolyDP(contour, 00.01*cv2.arcLength(contour,True),True)
                area = cv2.contourArea(contour)
                x,y,w,h = cv2.boundingRect(contour)
                #perimeter = ((w + w + h + h) - 4)
                #centerVal = ((image[int(y + (.5 * h)),int(x + (.5 * w))][0] + image[int(y + (.5 * h)),int(x + (.5 * w))][1] + image[int(y + (.5 * h)),int(x + (.5 * w))][2]) / 3)
                scalar = (((((320 - y) / 320)  * 2) + 1) * 16)
                if (y >= 320):
                    scalar = 16
                if ((cv2.arcLength(contour,True) > scalar) and (cv2.arcLength(contour,True) < 60)):  #16 is min cutoff #60 is max cutoff
                    if (area > cv2.arcLength(contour, True)):
                        bloomSize = 4;
                        if (x + w + bloomSize > width) or (y + h + bloomSize > height):
                            bloomSize = 0
                        xInc = x - bloomSize
                        yInc = y - bloomSize
                        sizeToAvg = (w + (bloomSize * 2)) * (h + (bloomSize * 2))
                        avgR = 0; avgB = 0; avgG = 0
                        while (xInc <= (x + w + bloomSize)):
                            while (yInc <= (y + h + bloomSize)):
     
                                avgB += image[yInc,xInc][0]
                                avgG += image[yInc,xInc][1]
                                avgR += image[yInc,xInc][2]
                                #if(yInc == (y - bloomSize)) or (yInc == (y + h + bloomSize)) or (xInc == (x - bloomSize)) or (xInc == (x + w + bloomSize)):
                                    ##image[yInc, xInc] = 0 #only to view bounding boxes as they are drawn
                                    #borderVal = ((image[yInc,xInc][0] + image[yInc,xInc][1] + image[yInc,xInc][2]) / 3)
                                    #if (borderVal < centerVal):
                                    #    borderDet += 1
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
                                    #cv2.imwrite(os.path.join(path , (str(frameCounter) + "_" + str(contourCount) +  ".png")), image[y-5:y+h+5, x-5:x+w+5])
                                    ##cv2.rectangle(image, (x - 5, y - 5), (x+w+5, y+h+5), (0, 0, 255), 2)
                                    if (determineLight("red", image[(y-4):(y+h+4), (x-4):(x+w+4)], frameCounter, contourCount)):
                                        redArray.append(contour)
                                    ####cv2.drawContours(image, contour, -1, (0, 0, 255), 3) #draws red
                            
                            #GREEN values
                            elif ((avgG > avgB and avgG > avgR) and (lightness < 233)):
                                    #cv2.imwrite(os.path.join(path , (str(frameCounter) + "_" + str(contourCount) +  ".png")), image[y-5:y+h+5, x-5:x+w+5])
                                    if (determineLight("green", image[(y-4):(y+h+4), (x-4):(x+w+4)], frameCounter, contourCount)):
                                        greenArray.append(contour)
                                    #cv2.rectangle(image, (x - 2, y - 2), (x+w+2, y+h+2), (0, 255, 0), 2)
                                    #cv2.drawContours(image, contour, -1, (0,255,0),3)

         
                            #else:
                            #    print("no val")
                                


            
            for contour in redArray:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(newImage, (x - 5, y - 5), (x+w+5, y+h+5), (0, 0, 255), 2)
            for contour in greenArray:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(newImage, (x - 5, y - 5), (x+w+5, y+h+5), (0, 255, 0), 2)
                
            #########################################################
            #kernel operator of size 3
            ###operator = np.ones((9,9),np.uint8)
        
            #blur the image slightly
            ###blur = cv2.GaussianBlur(masked,(3,3),0)                                         #blur step
            
            #apply top hat filter to grayscaled blurred image
            ###iImage = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, operator)                             #tophat step
            
            #threshold
            #thresh = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY)[1]                            #threshold step
            ###ret, thresh = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY)
            
            # Find the edges in the image using canny detector                        
            ###edges = cv2.Canny(thresh, 75, 200)                                                      #canny edge detection
            
            ###contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#contours
            #blankImage = np.zeros((height,width,3), np.uint8)

            ###contourCount = 0
            ###contour_list = []
            ###for contour in contours:
            ###    contourCount += 1
            ###    approx = cv2.approxPolyDP(contour, 00.01*cv2.arcLength(contour,True),True)
            ###    area = cv2.contourArea(contour)
            ###    #if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
            ###        #contour_list.append(contour)
            ###    #k=cv2.isContourConvex(approx)
            ###    if ((area > 20) and (area > cv2.arcLength(contour, True))):
            ###        perimeter = cv2.arcLength(contour,True)
            ###        if (perimeter < 50):
            ###            bloomSize = 4;
            ###            contour_list.append(contour)
            ###            x,y,w,h = cv2.boundingRect(contour)
            ###            xInc = x - bloomSize
            ###            yInc = y - bloomSize
            ###            sizeToAvg = (w + (bloomSize * 2)) * (h + (bloomSize * 2))
            ###            avgR = 0; avgB = 0; avgG = 0
            ###            while (xInc < (x + w + bloomSize)):
            ###                while (yInc < (y + h + bloomSize)):
     
            ###                    avgB += image[yInc,xInc][0]
            ###                    avgG += image[yInc,xInc][1]
            ###                    avgR += image[yInc,xInc][2]
            ###                    #image[yInc, xInc] = 0 #only to view bounding boxes as they are drawn
            ###                    yInc += 1
            ###                yInc = y - bloomSize
            ###                xInc += 1
            ###            avgR = avgR / sizeToAvg 
            ###            avgG = avgG / sizeToAvg 
            ###            avgB = avgB / sizeToAvg 
            ###            #hsl = (colorsys.rgb_to_hls(avgR / 255, avgG / 255, avgB / 255))
                        
                        ####
            ###            if not ((w * .6 >= h) or (h * .6 >= w)):
            ###                #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            ###            ####
            ###                #print(hsl, frameCounter)
            ###                if (avgR > avgG) and (abs(avgR - avgG) > 20):
            ###                    #cv2.drawContours(image, contour, -1, (0, 0, 255), 3) #draws red
            ###                    print("RED")
            ###                else:
            ###                    if (avgR < avgG) and (avgB < avgG):
            ###                        #cv2.drawContours(image, contour, -1, (0, 255, 0), 3) #draws green
            ###                        print("GREEN")
            ###                    else:
            ###                        #cv2.drawContours(image, contour, -1, (255, 0, 0), 3) #draws blue 
            ###                        print("BLUE")
                            
                            
                        #cv2.drawContours(image, contour, -1, (avgB, avgG, avgR), 3) #accurate colors
                        #print(contourCount)


                    

            #cv2.drawContours(image, contour_list,  -1, (255,0,0), 2)
           
            
            #convert 1 channel image to 3 channel for writing. then append to frameArray
            #newImage = np.zeros_like(image)
            #newImage[:,:,0] = edgesB
            #newImage[:,:,1] = edgesB
            #newImage[:,:,2] = edgesB
            frameArray.append(newImage)
            #frameArray.append(blankImage)
            
            
            

            
            #gradient
            ###rows = blur.shape[0]
            ###circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=8, minRadius=3, maxRadius=20)
            ###if circles is not None:
            ###    circles = np.uint16(np.around(circles))
            ###    for i in circles[0, :]:
            ###        center = (i[0], i[1])
            ###        # circle center
            ###        cv2.circle(grayscaleImage, center, 1, (0, 100, 100), 3)
            ###        # circle outline
            ###        radius = i[2]
            ###        cv2.circle(grayscaleImage, center, radius, (255, 0, 255), 3)
            
            
            #while (x < height):
            #    while (y < width):
            #        #
            #        #access to all pixels in image here
            #        
            #        #remove lower half of video
            #        #if (x > (height * .42)):
            #        #    newImage[x,y] = [0,0,0]
            #        #else:
            #            #this is the top portion of the image
            #            #pxVal = image[x,y]
            #
            #        newImage[x,y] = iImage[x,y] * thresh[x,y]

                        
            #           #end of the top portion of the image
            #        #end of access to all pixels in image
            #        y += 1 #increment width loop condition
            #    y = 0 #reset width loop var
            #    x += 1  #increment height loop condition

            #add
            #dst = cv2.add(iImage,thresh)




            ####can manipulate each frame before appending to the frame array
            
            ####append to frame array, output is written from frameArray
            #frameArray.append(newImage)
            print(str(round((frameCounter / total) * 100)) + "%")
            #print(str(frameCounter) + "%") #just for 100 frames
            frameCounter += 1


def determineLight(color, image, frameNum, contourNum):
    path = 'C:/Users/Chad/Documents/IUPUI/Fall 2019/CSCI43500 Multimedia Systems/Traffic_Signal_Detection/imageData/'
    height, width, layers = image.shape
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
        #else:       
        #    cv2.imwrite(os.path.join(path , (str(frameNum) + "_" + str(contourNum)+ "_" + color +  ".png")), dupImage)

    return detBool
    
def writeFrames(frameArray):
    #for j in range(len(frameArray)):
        #wrImg = frameArray[j]
        ##height, width, layers = wrImg.shape
        #size = (width,height)
    size = (1280,720)
    out = cv2.VideoWriter('projectVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    for i in range(len(frameArray)):
        out.write(frameArray[i])
        print((i / len(frameArray)) * 100)
    out.release()


#at end
if __name__== "__main__":
  main()
