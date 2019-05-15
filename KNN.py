from numpy import  * # todo finish the code
import operator
#print ("hello world")
#print (random.rand(4,4))

def test():
    randMat = mat(random.rand(3, 3))
    print(randMat)
    invRandMat = randMat.I
    print(invRandMat)
    print(randMat * invRandMat)
    print(eye(4))

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def createDataSet1():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = [4,2,0,0]
    return group,labels
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0];
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances** 0.5
    sortDistIndicies = distances.argsort();
    classCount ={}
    for i in range(k):
        voteLabel = labels[sortDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    groups ,labels = createDataSet();
    print(classify0([0,0,],groups,labels,3))