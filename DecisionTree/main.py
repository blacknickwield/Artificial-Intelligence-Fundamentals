# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
import math
from math import log


# 测试数据集
def create_data():
    datasets = [
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
    ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    print(type(datasets))
    # 返回数据集和每个维度的名称
    return datasets, labels


# 经验熵
def empirical_entropy_calculating(datasets):
    data_size = len(datasets)
    dictionary = {}
    for data in datasets:
        class_type = data[-1]
        if dictionary.get(class_type) is None:
            dictionary[class_type] = 1
        else:
            dictionary[class_type] += 1
    entropy = -sum([((value / data_size) * log((value / data_size), 2)) for value in dictionary.values()])
    # print(entropy)
    return entropy


# 经验条件熵
def conditional_empirical_entropy_calculating(datasets, attribute_index):
    data_size = len(datasets)
    attribute_dictionary = {}
    for data in datasets:
        attribute_value = data[attribute_index]
        if attribute_dictionary.get(attribute_value) is None:
            attribute_dictionary[attribute_value] = []
        attribute_dictionary[attribute_value].append(data)
    conditional_entropy = sum(((len(value) / data_size) * empirical_entropy_calculating(value))
                              for value in attribute_dictionary.values())
    # print(conditional_entropy)
    return conditional_entropy


# 信息增益
def information_gain_calculating(empirical_entropy, conditional_empirical_entropy):
    return empirical_entropy - conditional_empirical_entropy


# 决策树节点
class Node:
    def __init__(self, is_leaf=None, attribute=None):
        self.is_leaf = is_leaf
        self.attribute = attribute
        self.children = {}
        self.datasets = datasets
        self.labels = labels

    def add_child(self, attribute, node):
        self.children[attribute] = node


# 决策树
class DecisionTree:
    def __init__(self, threshold=0.1, datasets=None, labels=None):
        self.labels = labels
        self.datasets = datasets
        self.threshold = threshold
        self.root = None



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasets, labels = create_data()
    training_data = pd.DataFrame(datasets, columns=labels)
    print(training_data)
    # for label in labels:
    #     print(label)
    # for data in datasets:
    #     print(data)
    empirical_entropy = empirical_entropy_calculating(datasets)
    for i in range(len(labels) - 1):
        conditional_empirical_entropy = conditional_empirical_entropy_calculating(datasets, i)
        information_gain = information_gain_calculating(empirical_entropy, conditional_empirical_entropy)
        print(information_gain)

    decision_tree = DecisionTree()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
