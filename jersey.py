import pytesseract
import cv2, numpy as np
import argparse
from pytesseract import Output

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
        #self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def ocr(self, image):
        #pytesseract.pytesseract.tesseract_cmd = r''
        """boxes = pytesseract.image_to_boxes(image, lang='eng',
        config='--psm 5 --oem 1 -c tessedit_char_whitelist=0123456789')
        print(boxes)"""
        #self.display(image)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #bilarImg = cv2.bilateralFilter(gray,7,7,7)
        #image_enhanced = cv2.equalizeHist(bilarImg)
        #image = self.histogram_equalize(image)

        d = pytesseract.image_to_data(image, output_type=Output.DICT, config='--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789')
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(d)
        self.display(image)

    def display(self, image):
        #image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(2000)

    def histogram_equalize(self, img):
        b, g, r = cv2.split(img)
        red = cv2.equalizeHist(r)
        green = cv2.equalizeHist(g)
        blue = cv2.equalizeHist(b)
        return cv2.merge((blue, green, red))

if __name__ == '__main__':
    camera = Camera()

    #image = cv2.imread('images/example_03.png')
    image = cv2.imread('images/view6.jpg')
    #image = Image.open('images/example_03.png')
    #image = Image.open('images/view2.png')
    camera.ocr(image)