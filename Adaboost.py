from numpy import  *

def loadSimpData():
    dataMat=matrix([[1,2.1],
                    [2,1.1],
                    [1.3,1],
                    [1,1],
                    [2,1]])
    classLabels =[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump ={};bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
        stepSize= (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ('lt','gt'):
                threshVal = (rangeMin+float(j) * stepSize )
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weekClassArr=[]
    m=shape(dataArr)[0]
    D =mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weekClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print('total error:',errorRate)
        if errorRate==0.0:
            break
    print('------------------------------------')
    return weekClassArr
def adaClssify(dataToClass,classifyArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifyArr)):
        classEst = stumpClassify(dataMatrix,classifyArr[i]['dim'],\
                                 classifyArr[i]['thresh'],\
                                 classifyArr[i]['ineq'])
        aggClassEst+= classifyArr[i]['alpha']* classEst
        print(aggClassEst)
    return sign(aggClassEst)
if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    retArray = stumpClassify(dataMat,0,1.5,'lt')
    #print(retArray)
    D = mat(ones((5,1))/5)
    #print(buildStump(dataMat,classLabels,D))
    arr = adaBoostTrainDS(dataMat, classLabels)
    #print('result: ',arr)
    print(adaClssify([[5,5],[0,0]],arr))

