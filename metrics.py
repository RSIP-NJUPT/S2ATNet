
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

# 计算每个类和平均精度
def report_AA_CA(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成分类报告并评估指标
def report_metrics(y_test, y_pred_test):
    #data/augsburg
    # label_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area','Water']

    #Houston
    # label_names = ['Healthy Grass', 'Stressed Grass', 'Synthetic Grass', 'Trees', 'Soil', 'Water',
    #                 'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot1', 'Parking Lot2', 'Tennis Court', 'Running Track']
    classification = classification_report(y_test, y_pred_test, digits=4)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = report_AA_CA(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100
