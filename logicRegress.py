from numpy import  *
def createDataSet():
    group = array([[1.0,4.9],[1.8,2.2],[2.9,3.0],[4,1]])
    labels = [1,1,0,0]
    return group,labels


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.01
    maxCycle = 500
    weight = ones((n,1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weight)
        error = labelMatrix - h
        weight = weight + alpha * dataMatrix.transpose() * error
    return weight

#随机梯度上升，减少运算（没有向量运算）
def stocGradAscent0(dataMatIn,classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weight = ones(n)
    for k in range(m):
        h = sigmoid(sum(dataMatIn[k] * weight))
        error = classLabels[k] - h
        weight = weight + alpha  * error * dataMatIn[k]
    return weight
if __name__ == '__main__':
    group,label = createDataSet();
    print(gradAscent(group,label))
    print(stocGradAscent0(group, label))