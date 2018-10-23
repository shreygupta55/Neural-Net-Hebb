import numpy as np
import copy

result =[]
def getCombinations(pattern,path,index,misclassifiedLength):
    if misclassifiedLength == 0:
        temp =copy.deepcopy(path)
        result.append(temp)
        return
    for i in range(index,15):
        path.append(i)
        getCombinations(pattern,path,i+1,misclassifiedLength-1)
        path.pop()

flippedPatterns =[]
def flipping(combinationIndex,pattern):
    for i in range(len(combinationIndex)):
        temp = copy.deepcopy(pattern)
        for j in range(len(combinationIndex[i])):
            num = -1*temp.item(combinationIndex[i][j])
            temp.itemset(combinationIndex[i][j],num)
            flippedPatterns.append(temp)

result2 =[]
def getCombinationsForUndetermined(pattern,path,index,misclassifiedLength):
    if misclassifiedLength == 0:
        temp =copy.deepcopy(path)
        result2.append(temp)
        return
    for i in range(index,15):
        path.append(i)
        getCombinationsForUndetermined(pattern,path,i+1,misclassifiedLength-1)
        path.pop()

undeterminedPatterns=[]
def undeterminedFlipping(combinationIndex,pattern):
    for i in range(len(combinationIndex)):
        temp2 = copy.deepcopy(pattern)
        for j in range(len(combinationIndex[i])):
            num = 0*temp2.item(combinationIndex[i][j])
            temp2.itemset(combinationIndex[i][j],num)
            undeterminedPatterns.append(temp2)

result3 =[]
def getCombinationsForUndetermined3(pattern,path,index,misclassifiedLength):
    if misclassifiedLength == 0:
        temp =copy.deepcopy(path)
        result3.append(temp)
        return
    for i in range(index,15):
        path.append(i)
        getCombinationsForUndetermined3(pattern,path,i+1,misclassifiedLength-1)
        path.pop()

undeterminedPatterns3=[]
def undeterminedFlipping3(combinationIndex,pattern):
    for i in range(len(combinationIndex)):
        temp2 = copy.deepcopy(pattern)
        for j in range(len(combinationIndex[i])):
            num = 0*temp2.item(combinationIndex[i][j])
            temp2.itemset(combinationIndex[i][j],num)
            undeterminedPatterns3.append(temp2)

class neural:
    def training(self,trainData,trainOutput):
        model = {}
        weight = np.matrix('0;0;0;0;0;0;0;0;0;0;0;0;0;0;0')
        for i in range(len(trainData)):
            fresh = trainData[i].transpose()
            weight = weight + (fresh*trainOutput[i])
        bias = np.matrix('0')
        for a in range(len(trainOutput)):
            bias = bias + trainOutput[a]
        model['weight'] = weight
        model['bias'] = bias
        return model

    def testingMultiple(self,patternGiven,combinations,model):
        weight = model['weight']
        bias = model['bias']
        for i in range(len(combinations)):
            out = np.matmul(combinations[i],weight)
            if out <=0:
                return combinations[i]
        return patternGiven

    def testingMultipleForX(self,patternGiven,combinations,model):
        weight = model['weight']
        bias = model['bias']
        for i in range(len(combinations)):
            out = np.matmul(combinations[i],weight)
            if out >=0:
                return combinations[i]
        return patternGiven

def main():
    pattern1 = np.matrix('-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1')
    misclassifiedLen = 4
    getCombinations(pattern1,[],0,misclassifiedLen)
    flipping(result,pattern1)
    pattern2 = np.matrix('1,-1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1')
    trainData = [pattern1,pattern2]
    trainOutput = [np.matrix('1'),np.matrix('-1')]
    obj = neural()
    model = obj.training(trainData,trainOutput)
    pat = obj.testingMultiple(pattern1,flippedPatterns,model)
    print("Original Pattern:")
    print(pattern1)
    if (pat == pattern1).all():
        print("Correctly Classified as C")
    else:
        print("Vector which crashed the neural with k = ", misclassifiedLen," : ")
        print(pat)

    undeterminedLenForC = 8
    getCombinationsForUndetermined(pattern1,[],0,undeterminedLenForC)
    undeterminedFlipping(result2,pattern1)
    patForUndeterminedC = obj.testingMultiple(pattern1,undeterminedPatterns,model)
    print("--------------------------------------------------------------------")
    print("For undetermined C")
    print("Original Pattern of C:")
    print(pattern1)
    if(patForUndeterminedC == pattern1).all():
        print("Classified as C")
    else:
        print("Vector which got misclassified at k = " ,undeterminedLenForC," :")
        print(patForUndeterminedC)

    undeterminedLenForX = 8
    getCombinationsForUndetermined3(pattern2,[],0,undeterminedLenForX)
    undeterminedFlipping3(result3,pattern2)
    patForUndeterminedX = obj.testingMultipleForX(pattern2,undeterminedPatterns3,model)
    print("---------------------------------------------------------------------")
    print("For undetermined X")
    print("Original Pattern for X:")
    print(pattern2)
    if(patForUndeterminedX == pattern2).all():
        print("Classified as X")
    else:
        print("Vector which got misclassified at k = ",undeterminedLenForX, " : ")
        print(patForUndeterminedX)

if __name__ == '__main__':
    main()