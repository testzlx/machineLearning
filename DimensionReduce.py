from numpy import *

def loadSimpData():
    dataArr=[[1,1.1,8,7,4],
                    [2,1.9,5,7,3],
                    [2.9,3.1,4,7],
                    [4,4.2,3,6,7],
                    [5,5.3,1,2,3]]
    dataMat = [map(float,line) for line in dataArr]
    return mat(dataMat)

def pca(dataMat,topNfeat = 10):
    #todo，此处执行报错，没排查出结论
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat=cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigvects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigvects
    reconMat = (lowDDataMat*redEigvects.T)+meanVals
    return lowDDataMat,reconMat

if __name__ == '__main__':
    print(pca(loadSimpData()))