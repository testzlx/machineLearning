from numpy import *


def loadDataSet():
    postingList=[['my','dog','has','flea' \
                  'problem','help','please'],
                 ['maybe','not','take','him'\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute'\
                  'I','love','it'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['stop','him']]
    classVec=[0,1,0,1,0]
    return postingList,classVec
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
    return returnVec

#只能用作0/1分类
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbuseive = sum(trainCategory)/float(numTrainDocs)
    #防止其中一个概率值为0；防止下溢出
    p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom;p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbuseive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0 :
        return  1;
    else:
        return 0;


if __name__ == '__main__':
        dataSet,classList = loadDataSet();
        vocabList = createVocabList(dataSet)
        print(vocabList)
        trainMatrix = []
        for postingDoc in dataSet:
            trainMatrix.append(setOfWords2Vec(vocabList,postingDoc))
        p0Vect,p1Vect,pAbuseive = trainNB0(trainMatrix,classList)
        print("p0Vect:{},p1Vect:{},pAbuseive:{}".format(p0Vect,p1Vect,pAbuseive))
        testEntry =['stupid','garbage']
        thisDoc = array(setOfWords2Vec(vocabList,testEntry))
        print(testEntry,'classify as:',classifyNB(thisDoc,p0Vect,p1Vect,pAbuseive))