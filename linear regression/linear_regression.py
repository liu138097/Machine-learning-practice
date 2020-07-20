import numpy as np
#线性回归
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')#strip去除前后空格，split以空格划分
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegress(xArr,yArr):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    XTX=xMat.T*xMat
    if np.linalg.det(XTX)==0.0:
        print("This martixis singular,cannot do inverse")
        return
    ws=XTX.I*(xMat.T*yMat)
    return ws
xArr,yArr=loadDataSet('ex0.txt')#相对路径，看的是py文件的地址再加上ex0.txt
ws=standRegress(xArr,yArr)
#print('权重=',ws)
xMat=np.mat(xArr) ; yMat=np.mat(yArr).T
yHat=np.array(xMat*ws)
X=np.array(xMat[:,1])
Y=np.array(yMat[:,0])

print(np.corrcoef(yHat.T,yMat.T))#相关系数
#画图
import matplotlib.pyplot as plt
plt.figure(num=1,figsize = (6,6))
plt.subplot(111)#前两个代表行列，第三个代表第几个图，从左到右，从上到下
plt.scatter(X,Y,color='r',marker='o')
plt.plot(X,yHat,'k')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#plt.figure(num=2,figsize = (6,6))
#plt.show()

#局部加权线性回归
def lwlr(testpoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    m=xMat.shape[0]
    W=np.mat(np.eye(m))
    for i in range(m):
        diffMat=testpoint-xMat[i,:]
        W[i,i]=np.exp(diffMat*diffMat.T/(-2*k**2))
    XTWX=xMat.T*W*xMat
    if np.linalg.det(XTWX)==0.0:
        print("This martixis singular,cannot do inverse")
        return
    w=XTWX.I*xMat.T*W*yMat
    return testpoint*w
def lwlrTest(testArr,xArr,yArr,k=1.0):
    testMat=np.mat(testArr)
    m=testMat.shape[0]
    yHat=np.zeros((m,1))
    for i in range(m):
        yHat[i]=lwlr(testMat[i,:],xArr,yArr,k)
    return yHat
xArr,yArr=loadDataSet('ex0.txt')
#yHat=lwlrTest(xArr[0],xArr,yArr)
yHat=lwlrTest(xArr,xArr,yArr,k=0.01)
#print(yHat[0,0])
X=np.array(xArr)[:,1]
Y=np.array(yArr)

Xindex=np.argsort(X)
Xsort=X[Xindex]
ysort=yHat[Xindex]
plt.figure(num=2,figsize = (6,6))#创建窗口
plt.subplot(111)#创建窗口里的子图
plt.scatter(X,Y,color='r',marker='o')
plt.plot(Xsort,ysort,'k')
plt.show()

#误差计算Ein
def ressError(yArr,yHat):
    return ((yArr-yHat)**2).sum()

abX,abY=loadDataSet('abalone.txt')
yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1.0)
yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

err01=ressError(yHat01.T,abY[0:99])#列表可以看成只有一行，因此yHat要转成行向量
err1=ressError(yHat1.T,abY[0:99])
err10=ressError(yHat10.T,abY[0:99])
print(err01 ,err1 ,err10)

#误差计算Eout
yHat01=lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1=lwlrTest(abX[100:199],abX[0:99],abY[0:99],1.0)
yHat10=lwlrTest(abX[100:199],abX[0:99],abY[0:99],10.0)
err01=ressError(yHat01.T,abY[100:199])#列表可以看成只有一行，因此yHat要转成行向量
err1=ressError(yHat1.T,abY[100:199])
err10=ressError(yHat10.T,abY[100:199])
print(err01,'\n',err1,'\n',err10)

ws=standRegress(abX[0:99],abY[0:99])
yHat=np.mat(abX[100:199])*ws
err=ressError(yHat.T.A,abY[100:199])#.A是将矩阵转化为数组
print(err)

#岭回归
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    demo=xTx+np.eye(xMat.shape[1])*lam
    if np.linalg.det(demo)==0.0:
        print("This martixis singular,cannot do inverse")
        return
    ws=demo.I*(xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)#求每一列的平均，输出为一行
#    yMat=yMat-yMean
    xMeans=np.mean(xMat[:,1:],0)
    xVar=np.var(xMat[:,1:],0)
    xMat[:,1:]=(xMat[:,1:]-xMeans)/xVar
    numTestPts=30#迭代次数
    wMat=np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return xMeans,xVar,yMean,wMat
abX,abY=loadDataSet('abalone.txt')

abX=np.mat(abX);
one=np.ones((abX.shape[0],1))#添加常数项x0
abX=np.hstack((one,abX))

xMeans,xVar,yMean,ridgeWeights=ridgeTest(abX[0:3000],abY[0:3000])#特征归一化,xMeans1*8,xVar1*8,训练集的mean，var用到测试集上
#print(ridgeWeights.shape)

import matplotlib.pyplot as plt
fig=plt.figure()
ax=plt.subplot(111)
plt.plot(ridgeWeights)#为什么可以直接出来？
plt.xlabel('log(lamda)')
plt.ylabel('weights')#lamda越大，惩罚越大，weights越趋近于0
plt.show()

def yHatlabel(xArr,ridgeWeights):
    xMat=np.mat(xArr)
    xMat[:,1:]=(xMat[:,1:]-xMeans)/xVar
    yHat=np.zeros((30,xMat.shape[0]))#30*2800
    for i in range(30):
        ridgeweight=np.mat(ridgeWeights)[i,:]#1*8
        yHat[i,:]=ridgeweight*xMat.T#1*8 8*2800 1*2800
    return yHat#+yMean#30*2800

yHat=yHatlabel(abX[3000:4177],ridgeWeights)
print(yHat)
        
def ressError(yHat,abY):
    error=yHat-abY
    error=np.power(error,2).sum(axis=1)#按行求和
    return error/1777
error=ressError(yHat,abY[3000:4177])
print(np.argmin(error))#如果要选择lamda，那必须用验证集来选，用训练集当然lamda越小越好，因为这样就没限制啦

#y-ymean可以消去常数项，前提是x需要特征归一化
#也可以用np.hastack()加一列x0，推荐这一种

#lasso