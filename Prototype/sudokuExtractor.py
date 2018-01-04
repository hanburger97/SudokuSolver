import cv2
import numpy as np
import pickle

from helpers import Helpers
from cells import Cells



class Extractor(object):
    '''
        Stores and manipulates the input image to extract the Sudoku puzzle
        all the way to the cells
    '''
    digits = []
    
    def __init__(self, path):
        self.helpers = Helpers()  # Image helpers
        if(path=='error1'):
            self.errorimg = self.loadImage('input.jpeg')
            W = self.errorimg.shape[0]
            H = self.errorimg.shape[1]
            cv2.putText(self.errorimg,"Grid not Detected",(0,int(H/2)) , cv2.FONT_HERSHEY_SIMPLEX, int(1*(W/400)), (0,0,255),3)
            self.helpers.show(self.errorimg, 'Final Sudoku grid')
            return
        if(path=='error2'):
            self.errorimg = self.loadImage('input.jpeg')
            W = self.errorimg.shape[0]
            H = self.errorimg.shape[1]
            cv2.putText(self.errorimg,"Could not Solve",(0,int(H/2)) , cv2.FONT_HERSHEY_SIMPLEX, int(1*(W/400)), (0,0,255),3)
            self.helpers.show(self.errorimg, 'Final Sudoku grid')
            return
        self.image = self.loadImage(path)
        self.preprocess()
        #self.helpers.show(self.image, 'After Preprocessing')
        self.sudoku = self.cropSudoku()
        #self.helpers.show(sudoku, 'After Cropping out grid')
        self.sudoku = self.straighten(self.sudoku)
        
        #self.helpers.show(self.sudoku, 'Final Sudoku grid')
        self.digits = Cells(self.sudoku).cells

        
    def displayDigets(self,sudoku,inputArray,digets):
        sudokuDisplay = cv2.cvtColor(sudoku, cv2.COLOR_GRAY2RGB)
        sudokuDisplay = cv2.bitwise_not(sudokuDisplay)
        
        W = sudoku.shape[0]
        H = sudoku.shape[1]
        W=W//9
        H=H//9
        for i in range(0,81):
            if(not(digets[i]==0) and inputArray[i]==0):
                cv2.putText(sudokuDisplay,str(digets[i]), (int(W*(i%9)+W/4),int(H*(i//9)+3*H/4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        self.helpers.show(sudokuDisplay, 'Final Sudoku grid')
        return sudoku
        
    def loadImage(self, path):
        color_img = cv2.imread(path)

        if color_img is None:
            raise IOError('Image not loaded')
        print('Image loaded.')
        return color_img

    def preprocess(self):
        print('Preprocessing...', end=' ')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.helpers.thresholdify(self.image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        print('done.')

    def cropSudoku(self):
        print('Cropping out Sudoku...', end=' ')
        contour = self.helpers.largestContour(self.image.copy())
        sudoku = self.helpers.cut_out_sudoku_puzzle(self.image.copy(), contour)
        print('done.')
        return sudoku

    def straighten(self, sudoku):
        print('Straightening image...', end=' ')
        largest = self.helpers.largest4SideContour(sudoku.copy())
        app = self.helpers.approx(largest)
        corners = self.helpers.get_rectangle_corners(app)
        sudoku = self.helpers.warp_perspective(corners, sudoku)
        print('done.')
        return sudoku
