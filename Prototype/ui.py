# import the GUI manager package
from tkinter import *
from soduko import solvePuzzle
from soduko import puzzleOutput
from soduko import convertToIntArray
from soduko import printPuzzle
from soduko import fixPuzzle
from soduko import isValid
from soduko import getInvalids
import numpy as np
import sudokuExtractor as s
import time

from Brain.Brain import *

#UNCOMMENT FOR PI
#from camera import takePhoto


# make a window pop up
def showCamera():
    print("cam enabled")
    #TODO

def displayPuzzle(array):
    s = puzzleOutput(array)
    L1 = Label(root,bg="red",text=s,wraplength=200)
    L1.place(x=0,y=0,width=200,height=200)

def savePuzzle(grid):
    file = open("output.txt","w")
    file.write("-"*30+"\n")
    s=""
    for i in range(0,len(grid)):
        s+= " "+str(grid[i])+" "
        if((i+1)%3==0):
            s += "|"
        if((i+1)%9==0):
            file.write(s+"\n")
            s=""
        if((i+1)%27==0):
            file.write("-"*30+"\n")
    file.close() 

def convertToHan(board):
    newBoard = []
    for diget in board:
        newDiget = []
        for row in diget:
            for pixel in row:
                if pixel ==0:
                    newDiget.append(0)
                elif pixel==1:
                    newDiget.append(1) 
        newBoard.append(newDiget)
    npa = np.asarray(newBoard, dtype=np.int)
    return npa

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}').format(filename, lineno, line.strip(), exc_obj)



def main():

    

    while True:
        
        #UNCOMMENT FOR PI
        #takePhoto('input.jpeg')
        
        try:
            board = s.Extractor('input.jpeg')
        except:
            board = s.Extractor('error1')
            continue
        
        try:
            #HAN UR NUMPY ARRAY
            b = Brain()
            digets = convertToHan(board.digits)
            #b.train_model_mnist()
            inputArray = b.predictMultiple(inputs=digets,modelPath="./model/mnist")
            inputString = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
            #inputString= "030010060750030048006984300003000800912000674004000500001675200680090015090040030"
            #inputArray = convertToIntArray(inputString)
            #inputString = "0"*81
            #inputString= "070230000000740809058109002005400008007802001300000750000608190004021000000974081"
            #inputArray = convertToIntArray(inputString)
            
            inputForDisplay = list(inputArray)
            print("INPUT:")
            
            if(isValid(inputArray)):
                output = solvePuzzle(inputArray)
                printPuzzle(inputArray)
                print("")
            else:
                inputForDisplay = list(inputArray)
                printPuzzle(inputArray)
                invalids = getInvalids(list(inputArray))
                fixedInput = fixPuzzle(list(inputArray),invalids)
                print("")
                print("FIXED INPUT:")
                printPuzzle(fixedInput)
                output = solvePuzzle(fixedInput)

                
            print("")
            print("OUTPUT:")
            printPuzzle(output)
            savePuzzle(output)
            board.displayDigets(board.sudoku,inputForDisplay,output)

        except Exception as e:
            board = s.Extractor('error2')
            PrintException()
            
        

    
main()



