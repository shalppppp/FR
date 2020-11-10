
#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: auc.py
#   @time: 2020/11/4 22:55
#   @Function:
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
def plot_auc(data):
    fpr, tpr,_ = roc_curve(data[:,0], data[:,1])
    x = metrics.auc(fpr, tpr)#label， score
    print(x)

    plt.plot(fpr, tpr)
    plt.show()


if __name__ == "__main__":
    # plt.title("ROC curve of %s (AUC = %.4f)"%('lightgbm', x))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for i in range(9):

        # print(i)
        filepath = "./log/auc_" + str(i)
        data = np.loadtxt(filepath)
        data = data.T

        print(data.shape)
        fpr, tpr,_ = roc_curve(data[:,0], data[:,1])
        choice = fpr > 0
        fpr = fpr[choice]
        tpr = tpr[choice]
        x = metrics.auc(fpr, tpr)#label， score
        line = plt.plot(fpr, tpr)
        plt.legend(handles = line, labels = str(i),loc =  'lower right')

    plt.show()

    # print(np.sum(data[:,0]))
    # plot_auc(data)



