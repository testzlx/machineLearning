from numpy import *

class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink=None
        self.parent=parentNode
        self.children={}

    def inc(self,numOccur):
        self.count += numOccur

    def disp(self,ind=1):
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

def loadSimpDat():
    simDat = [['r','z','h','j','p'],
              ['z','y','x','w','v','u','t','s'],
              ['z'],
              ['r','x','n','o','s'],
              ['y','r','x','z','q','t','p'],
              ['y','z','x','e','q','s','t','m']]
    return simDat
def createInitset(dataSet):
    retDict={}
    for trans in dataSet:
        retDict[frozenset(trans)] =1
    return  retDict

def classTest():
    rootNode = treeNode('pyramid',9,None)
    rootNode.children['eye'] = treeNode('eye',13,None)
    rootNode.disp()


def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(orderedItems, retTree, headerTable, count):
    if orderedItems[0] in retTree.children:
        retTree.children[orderedItems[0]].inc(count)
    else:
        retTree.children[orderedItems[0]] = treeNode(orderedItems[0],count,retTree)
        if headerTable[orderedItems[0]][1] == None:
            headerTable[orderedItems[0]][1] = retTree.children[orderedItems[0]]
        else:
            updateHeader(headerTable[orderedItems[0]][1],retTree.children[orderedItems[0]])
    if len(orderedItems) > 1:
        updateTree(orderedItems[1::],retTree.children[orderedItems[0]],headerTable,count)



def createTree(dataSet,minSup=1):
    headerTable={}
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) ==0:
        return  None,None
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    retTree = treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localID={}
        for item in tranSet:
            if item in freqItemSet:
                localID[item]=headerTable[item][0]
        if len(localID) > 0:
            orderedItems = [v[0] for v in sorted(localID.items(),key=lambda  p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    condPats ={}
    while treeNode != None:
        prefixPath =[]
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) >1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


if __name__ == '__main__':
    #classTest()
    initSet = createInitset(loadSimpDat())
    myFPtree,myHeaderTab = createTree(initSet,3)
    myFPtree.disp()