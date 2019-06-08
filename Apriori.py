from numpy import *


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)

def scanD(D,Ck,minSupport):
    ssCnt={}
    dataList = list(Ck)
    for tid in D:
        for can in dataList:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] =1
                else:
                    ssCnt[can] +=1
    #todo  numItems为D的长度,4为魔法数字
    numItems = float(4)
    retList = []
    supportData={}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData


def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1=createC1(dataSet)
    D = map(set,dataSet)
    #todo map(set,集合)为啥只能做一次遍历，第二次遍历就为空集合了
    dataList2 = list(D)
    L1,supportData = scanD(dataList2,C1,minSupport)
    L = [L1]
    k =2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(dataList2,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData


def rulesFromConseq(freqSet, H1, supportData, bigRuleList, minconf):
    m=len(H1[0])
    if(len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H1,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,bigRuleList,minconf)
        if(len(Hmp1) >1):
            rulesFromConseq(freqSet,Hmp1,supportData,bigRuleList,minconf)


def calcConf(freqSet, H1, supportData, bigRuleList, minconf):
    prunedH=[]
    for conseq in H1:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minconf:
            print(freqSet-conseq,'-->',conseq,'conf',conf)
            bigRuleList.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH


def generateRules(L,supportData,minconf=0.7):
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if (i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minconf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minconf)
    return bigRuleList


if __name__ == '__main__':
    dataSet = loadDataSet();
    L, supportData =apriori(dataSet, 0.5)
    print(L)
    print('-----------------------------')
    print(generateRules(L,supportData))