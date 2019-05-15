import  matplotlib
import matplotlib.pyplot as plt
from numpy import  *
import  KNN

def drawFigure(datingDataMat,labels):
    fig = plt.figure();
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
             15.0* array(labels), 15.0* array(labels) )
    plt.show()
if __name__ == '__main__':
    datingDataMat,labels = KNN.createDataSet1();
    drawFigure(datingDataMat,labels)