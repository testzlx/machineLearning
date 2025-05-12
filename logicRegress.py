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

#随机梯度上升，减少运算（没有向量运算）  每个样本的损失函数对参数的梯度，是整体损失函数梯度的无偏估计。
def stocGradAscent0(dataMatIn,classLabels):
    m, n = shape(dataMatIn)
    alpha = 0.01
    weight = ones(n)
    for k in range(m):
        h = sigmoid(sum(dataMatIn[k] * weight))
        error = classLabels[k] - h
        weight = weight + alpha  * error * dataMatIn[k]
    return weight


def predict(data, weight, threshold=0.5):
    """
    使用训练好的权重进行预测

    参数:
        data: 新的输入数据，可以是一条或多条样本（二维数组或一维数组）
        weight: 训练得到的权重向量（列向量）
        threshold: 分类阈值（默认 0.5）

    返回:
        preds: 预测结果（0 或 1）
    """
    data = array(data)

    # 如果是一维数据（单样本），转换为二维
    if data.ndim == 1:
        data = data.reshape(1, -1)

    weight = array(weight).reshape(-1, 1)  # 保证是列向量
    probs = sigmoid(data @ weight)  # shape: (m, 1)
    preds = (probs >= threshold).astype(int)
    return preds.flatten()  # 返回 1D 的预测结果



if __name__ == '__main__':
    group,label = createDataSet();
    print(gradAscent(group,label))
    print(stocGradAscent0(group, label))
    weight = stocGradAscent0(group, label)
    test_data=[5,2]
    print(predict(test_data,weight))



