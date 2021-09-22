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
    def __init__(self, is_leaf=None, label=None, class_type=None):
        # 是否为叶节点,若为叶节点则有类标记
        self.is_leaf = is_leaf
        # 类标记
        self.class_type = class_type
        # 若不为叶节点，则有用于分类的属性
        self.label = label
        # 若不为叶节点，则有子树节点集合
        self.children = None

    def add_child(self, label, node):
        self.children[label] = node


# 决策树
class DecisionTree:
    def __init__(self, threshold=0.1, datasets=None, labels=None):
        self.labels = labels
        self.datasets = datasets
        # 阈值
        self.threshold = threshold
        # 决策树根节点
        self.root = None
        # 记录属性是否已用于分类过
        self.labels_vis = {}
        for label in labels:
            self.labels_vis[label] = False

    def __ID3_train__(self, datasets):
        # 数据集中所有实例属于同一类，已该类作为类标记，节点为叶节点
        total = []
        for data in datasets:
            if data[-1] == '是':
                total.append(data)

        if len(total) == len(datasets):
            return Node(is_leaf=True, class_type='是')
        if len(total) == 0:
            return Node(is_leaf=True, class_type='否')

        # 属性已用完,选择实例数最大的类作为类标记
        available_label_number = 0
        for x in self.labels_vis.values():
            if x is False:
                available_label_number += 1

        if available_label_number == 0:
            yes_total = 0
            no_total = 0
            for data in datasets:
                if data[-1] == '是':
                    ++yes_total
                else:
                    ++no_total
            if yes_total >= no_total:
                return Node(is_leaf=True, class_type='是')
            else:
                return Node(is_leaf=True, class_type='否')
        # 计算最大信息增益
        max_information_gain = 0
        index = -1
        empirical_entropy = empirical_entropy_calculating(datasets)
        for j in range(len(self.labels) - 1):
            # print(self.labels_vis[self.labels[i]])
            if self.labels_vis[self.labels[i]] is True:
                continue
            conditional_empirical_entropy = conditional_empirical_entropy_calculating(datasets, j)
            information_gain = information_gain_calculating(empirical_entropy, conditional_empirical_entropy)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                index = j
        # 最大信息增益小于阈值,选择实例数最大的类作为类标记
        if max_information_gain < self.threshold:
            yes_total = 0
            no_total = 0
            for data in datasets:
                if data[-1] == '是':
                    ++yes_total
                else:
                    ++no_total
            if yes_total >= no_total:
                return Node(is_leaf=True, class_type='是')
            else:
                return Node(is_leaf=True, class_type='否')
        # print(3)
        node = Node(is_leaf=False, label=labels[index])
        self.labels_vis[index] = True
        node.children = {}
        # 划分datasets
        value_datasets = {}
        for data in datasets:
            attribute_value = data[index]
            if value_datasets.get(attribute_value) is None:
                value_datasets[attribute_value] = []
            value_datasets[attribute_value].append(data)

        for key in value_datasets.keys():
            # node.children[key] = Node()
            node.children[key] = self.__ID3_train__(value_datasets[key])
        return node

    def visit(self, node, depth=0):
        if node is None:
            return
        for i in range(depth):
            print(' ', end="")
        if node.children is None:
            print(node.class_type)
            return
        if node.is_leaf is False:
            print(node.label)

        for child in node.children.items():
            self.visit(child[1], depth + 1)

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
        # print(information_gain)

    decision_tree = DecisionTree(datasets=datasets, labels=labels)
    decision_tree.root = decision_tree.__ID3_train__(decision_tree.datasets)
    decision_tree.visit(decision_tree.root)
