import torch
import numpy as np


def cala(label, pre_tags, met):
    if label.shape != pre_tags.shape:
        raise Exception('different dimension of Tensor')
    if len(label.shape) != 2:
        raise Exception('wrong shape of label')
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == pre_tags[i, j] and label[i, j] == 1:
                met[j][0] += 1
            elif label[i, j] == pre_tags[i, j] and label[i, j] != 1:
                met[j][1] += 1
            elif label[i, j] != pre_tags[i, j] and label[i, j] != 1:
                met[j][2] += 1
            else:
                met[j][3] += 1
    return met


def acc_f1(met):
    if len(np.array(met).shape) >= 2:
        count = np.sum(met, axis=0)
        tp = count[0]
        tn = count[1]
        fp = count[2]
        fn = count[3]
    else:
        tp = met[0]
        tn = met[1]
        fp = met[2]
        fn = met[3]
    eps = 1e-5
    acc = round((tp + tn) / (tp + tn + fp + fn + eps), 4)
    precision = round(tp / (tp + fp + eps), 4)
    recall = round(tp / (tp + fn + eps), 4)
    f1 = round(2 * precision * recall / (precision + recall + eps), 4)
    return acc, precision, recall, f1


if __name__ == "__main__":
    label = torch.Tensor([[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1]]).long()
    pre_tags = torch.Tensor([[1, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1]]).long()
    met = np.zeros([6, 4])# TP TN FP FN
    met = cala(label, pre_tags, met)
    for i in range(2):
        print(met[i], acc_f1(met[i]))
