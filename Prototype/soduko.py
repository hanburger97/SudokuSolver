
def same_row(i,j): return (int(i/9) == int(j/9))
def same_col(i,j): return (i-j) % 9 == 0
def same_block(i,j): return (int(i/27) == int(j/27) and int(i%9/3) == int(j%9/3))
def isValid(gridString):
    if(len(gridString)!=81):
        return False
    for i in range(0, 81):
        if(gridString[i]==0 or isinstance(gridString[i], list)):
            continue
        for j in range(0,81):
            if(i==j):
                continue
            if(gridString[i]==gridString[j]):
                if(same_row(i,j) or same_col(i,j) or same_block(i,j)):
                    return False
    return True
def isPosValid(grid,pos,value):
    i = pos
    for j in range(0,81):
        if(i==j):
            continue
        if(value==grid[j]):
            if(same_row(i,j) or same_col(i,j) or same_block(i,j)):
                return False
    return True
            
def getPossibilities(gridString):
    
    for i in range(0,81):
        if(gridString[i]==0):
            gridString[i]=[]
            for j in range(1,10):
                for k in range(0,81):
                    if(j==gridString[k]):
                        if(same_row(i,k) or same_col(i,k) or same_block(i,k)):
                            break
                    if(k==80):
                        gridString[i].append(j)
    
    return gridString

def recursive(grid, index):
    if(index==81 and isValid(grid)):
        return grid
    
   
    if(isinstance(grid[index], list)):
        filledGrid = None
        for i in range(0,len(grid[index])):
            if(filledGrid==None and isPosValid(grid,index,grid[index][i])):
                newgrid = list(grid)
                newgrid[index] = grid[index][i]
                filledGrid = recursive(newgrid,index+1)
        return filledGrid
    else:
        return recursive(list(grid),index+1)
    
      
def fixPuzzle(inputArray,conflicts):
    inputArray = list(inputArray)
    for i in range(0,81):
        if(i in conflicts):
            inputArray[i]=0
    return inputArray
              
    
def convertToIntArray(stringInput):
    output = []
    for i in range(0,len(stringInput)):
        output.append(int(stringInput[i]))
    return output


def printPuzzle(grid):
    s = ""
    print("-"*30)
    for i in range(0,len(grid)):
        s+= " "+str(grid[i])+" "
        if((i+1)%3==0):
            s += "|"
        if((i+1)%9==0):
            print(s)
            s=""
        if((i+1)%27==0):
            print("-"*30)
            
def solvePuzzle(inputArray):
    assert(isValid(inputArray)) 
    possibilities = getPossibilities(list(inputArray))
    result = recursive(possibilities,0)
    assert(isValid(result)) 
    return result
            
def puzzleOutput(grid):
    s = "-"*30+"\n"
    for i in range(0,len(grid)):
        s+= " "+str(grid[i])+" "
        if((i+1)%3==0):
            s += "|"
        if((i+1)%9==0):
            s+="\n"
        if((i+1)%27==0):
            s+="-"*30+"\n"
    return s
            
    
def getInvalids(inputArray):
    invalids = []
    
    for i in range(0,len(inputArray)):
        if(inputArray[i]==0):
            continue
        if(not isPosValid(inputArray,i,inputArray[i])):
            invalids.append(i)
    return invalids


           




                       
                            
