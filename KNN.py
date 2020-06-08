# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import operator
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
group,labels= createDataSet()
plt.scatter(group[:2][0],group[:2][1],label='A')
plt.scatter(group[2:][0],group[2:][1],label='B')
plt.legend(loc='best')
plt.show()

def classify0(inX,dataSet,labels,k,p=2):
    dataSetSize=dataSet.shape[0]
    distance=np.linalg.norm((inX-dataSet),axis=1)
    sortedDistIndicies=np.argsort(distance)
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda x:x[-1])[-1][0]
    return sortedClassCount
#print(classify0([0,0],group,labels,3))

def file2matrix(filename):
    fr=open(filename)
#    rows=len(fr.readline().strip().split('\t'))-1    
    arrayOLines=fr.readlines()

    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip().split('\t')
        returnMat[index,:]=line[0:3]
        classLabelVector.append(int(line[-1]))
        index+=1
    return returnMat,classLabelVector
datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
print(datingDataMat[:5])
print(datingLabels[:5])

fig=plt.figure()
ax=fig.add_subplot(121)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])

ax=fig.add_subplot(122)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))#尺寸，颜色
plt.show()

def autoNorm(dataSet):
    minVals=np.min(dataSet,axis=0)
    maxVals=np.max(dataSet,axis=0)
    ranges=maxVals-minVals
    normDataSet=(dataSet-minVals)/ranges
    return normDataSet,ranges,minVals
#norMat,ranges,minVals=autoNorm(datingDataMat)
def datingClassTest():
    hoRatio=0.10
    dataingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    numTestVecs=int(normMat.shape[0]*hoRatio)
    errCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:,:],datingLabels[numTestVecs:],3)
        print("the classifier came back with :%d,the real answer is : %d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errCount+=1.0
    print("the total error rate is: %f"%(errCount/float(numTestVecs)))
    return 0
#print(datingClassTest())
#datingClassTest()

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per years?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    dataingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=np.array([percentTats,ffMiles,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult-1])
    return 0
#classifyPerson()

def img2vector(filename):
    returnvector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnvector[0,32*i+j]=int(lineStr[j])
    return returnvector
#testVector=img2vector('testDigits/0_13.txt')
#print(testVector[0,0:31])

from os import listdir
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=listdir('testDigits')
    errCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with : %d,the real answer is : %d"%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errCount==1.0
    print("\nthe total number for errors is : %d"%errCount)
    print("\nthe total error rate is : %f"%(errCount/float(mTest)))
    return 0
handwritingClassTest()