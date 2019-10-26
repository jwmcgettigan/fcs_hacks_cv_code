import cv2, numpy as np
import argparse
import matplotlib as plt

"""
A fcs_hackathon Project.
Author: Justin McGettigan

This program will do the following:
- take input images or footage
- determine the borders of the game field
- crop the game field out of the camera view
- detect all of the players on the field
- identify what team each player is on
"""

class Field:
    def __init__(self):
        pass

class Player:
    def __init__(self):
        pass

class Camera:
    def __init__(self):
        self.window_name = "Test"
        self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(self.window_name, 600, 600)

    def process_image3(self, image):
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #green range
        lower_green = np.array([0, 0, 0])
        upper_green = np.array([255, 255, 255])

        #Define a mask ranging from lower to uppper
        mask = cv2.inRange(hsv, lower_green, upper_green)
        #Do masking
        res = cv2.bitwise_and(image, image, mask=mask)
        #convert to hsv to gray
        res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

        self.display(res_bgr)

    def process_image(self, image):
        # make image gray
        # apply gaussian blur
        # find canny edges
        # find the hough lines
        # 

        """lower = [1, 0, 20]
        upper = [60, 40, 200]

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        ret,thresh = cv2.threshold(mask, 40, 255, 0)
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        edges = cv2.Canny(blur,100,200)
        #cv2.threshold(image_gray, 127,(0,255,0),0)
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(output, contours, -1, 255, 3)

            #find the biggest area
            c = max(contours, key = cv2.contourArea)

            x,y,w,h = cv2.boundingRect(c)
            # draw the book contour (in green)
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)"""

        """blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur,50,150,apertureSize = 3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)"""

        self.display(filtered_image)

    def mask(self, image):
        lower = [1, 0, 20]
        upper = [60, 40, 200]

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        ret,thresh = cv2.threshold(mask, 40, 255, 0)
        return thresh

    def process_video(self):
        pass

    def process_image2(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel_size = 3

        bilarImg = cv2.bilateralFilter(gray,7,7,7)
        image_enhanced = cv2.equalizeHist(bilarImg)

        #plt.imshow(image_enhanced)

        masked_edges = cv2.Canny(image_enhanced, 100, 170, apertureSize = 3)

        self.display(masked_edges)
        #plt.imshow(masked_edges, cmap='gray')

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 110     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 90 #minimum number of pixels making up a line
        max_line_gap = 20   # maximum gap in pixels between connectable line segments

        line_image = np.copy(img)*0 #creating a blank to draw lines on

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

        # Draw the lines on the edge image
        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

        # remove small objects
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, se1)
        self.display(combo)

    def display(self, image):
        image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(2000)

if __name__ == '__main__':
    camera = Camera()

    image = cv2.imread('images/view1.jpg')
    camera.process_image3(image)

