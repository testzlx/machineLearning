from numpy import *

def createDataSet():
    dataSet = [[1, 1, 1.1],
               [1, 1, 0.9],
               [1, 1, 1.2],
               [0, 1, 0.5],
               [0, 1, 0.4]]
    return dataSet

def  distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ +rangeJ * random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] =  minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

#二分k-均值算法程序有些复杂，没看懂
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centriod0 = mean(dataSet,axis = 0).tolist()[0]
    centList=[centriod0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centriod0),dataSet[j,:])**2
    while(len(centList) < k):
        lowerSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit=sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowerSSE :
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowerSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is:",bestCentToSplit)
        print("the len of bestClustAss is:",len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment




if __name__ == '__main__':
    dataSet = createDataSet()
    #print(randCent(mat(dataSet),2))
    print(biKmeans(mat(dataSet),3))
