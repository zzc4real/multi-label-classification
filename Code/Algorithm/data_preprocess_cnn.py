import pandas as pd
import numpy as np
from collections import Counter

def load_data(filename):
    df = pd.read_csv(filename)
    selected = ['Labels','ImageID']
    non_selected = list(set(df.columns) - set(selected))

    # drop Caption here
    df = df.drop(non_selected, axis=1)
    # delete empty
    df = df.dropna(axis=0, how='any', subset=selected)

    labels = []
    labels_exe = []
    raw_labels = df[selected[0]].tolist()
    for each in raw_labels:
        temp = each.split(' ')
        labels.append(temp)
        for i in temp:
            labels_exe.append(int(i))
    # there are 18 different labels, without 12
    label_dim = len(Counter(labels_exe).keys())+1
    label_num = len(labels)

    # create one_hot 30000 line 19 column
    one_hot = np.zeros((label_num, label_dim), int)
    # print(one_hot.shape)

    m = 0
    label_int = []
    for each in labels:
        temp = []
        for i in range(len(each)):
            temp.append(int(each[i]))
            one_hot[m][int(each[i])-1] = 1
        m += 1
        label_int.append(temp)
    # print(one_hot[3])

    # match one_hot with original labels
    label_dict = dict(zip(raw_labels, one_hot))

    return one_hot