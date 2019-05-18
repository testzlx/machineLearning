from numpy import  *

def loadSimpData():
    dataMat=matrix([[1,1.1],
                    [2,1.9],
                    [2.9,3.1],
                    [4,4.2],
                    [5,5.3]])
    classLabels =[1.0,2.0,3.0,4.0,5.0]
    return dataMat,classLabels

def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat=mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("can not inverse")
        return
    ws = xTx.I * (xMat.T *yMat)
    return ws

# 局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat =mat(xArr);yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print(" can not reverse")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom= xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("can not inverse")
        return
    ws = denom.I *(xMat.T*yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean=mean(yMat,0)
    yMat = yMat-yMean
    xMeans = mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    #print(standRegres(dataMat,classLabels))
    #print(lwlr(dataMat[0],dataMat,classLabels,0.7))
    #print(lwlr(dataMat[0], dataMat, classLabels, 1.0))
    #print(lwlrTest(dataMat, dataMat, classLabels, 0.5))
    print(ridgeTest(dataMat,classLabels))
