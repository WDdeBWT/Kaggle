import copy
import time
import math
import random
import operator

import numpy as np


class FCM:

    def __init__(self, inp_mat, clusters, max_iter, m = 2.00, show_detail = False):
        self.inp_mat = inp_mat
        self.item_num = inp_mat.shape[0]
        self.feat_num = inp_mat.shape[1]
        self.clusters = clusters
        self.max_iter = max_iter
        self.m = m
        self.show_detail = show_detail

    # 返回的矩阵是每个点属于不同的集群的概率（隶属度）
    # 随机生成
    def initialize_membership_matrix(self):
        membership_mat = list()
        for i in range(self.item_num):
            random_num_list = [random.random() for i in range(self.clusters)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]
            membership_mat.append(temp_list)
        return membership_mat

    def calculate_cluster_center(self, membership_mat):
        item_mem_of_each_clu = list(zip(*membership_mat))
        # item_mem_of_each_clu 's shape is (k, n)
        cluster_centers = list()
        for j in range(self.clusters):  # 选定一个类别
            one_clu_mems = item_mem_of_each_clu[j]
            xraised = [one_mem ** self.m for one_mem in one_clu_mems]  # m is a Fuzzy parameter
            denominator = sum(xraised)  # 求和

            # 将对应点向量乘上它在该类别上的概率 得到概率向量
            temp_num = list()
            for i in range(self.item_num):
                data_point = list(self.inp_mat[i])
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)

            # 将所有点在对应特征下的概率累积求和
            numerator = map(sum, zip(*temp_num))
            center = [z / denominator for z in numerator]
            # the shape of center is (num_attr,) , which is the number of the features.
            # 本质上做的是加权平均，关于不同的特征，在不同的点上做加权。权重即为初始概率
            cluster_centers.append(center)
        return cluster_centers

    def update_membership_value(self, membership_mat, cluster_centers):
        # m is the Fuzzy parameter.
        p = float(2 / (self.m - 1))
        # cluster_centers : (k, num_attr)
        for i in range(self.item_num):
            data_point = list(self.inp_mat[i])
            # 算和两个中心的距离，这里采用的是二范数
            distances = [np.linalg.norm(list(map(operator.sub, data_point, cluster_centers[j]))) for j in range(self.clusters)]
            # 更新概率矩阵
            for j in range(self.clusters):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(self.clusters)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    def get_loss(self, membership_mat, cluster_centers):
        loss = 0
        for i in range(self.item_num):
            for j in range(self.clusters):
                data_point = list(self.inp_mat[i])
                distance = np.linalg.norm(list(map(operator.sub, data_point, cluster_centers[j])))
                loss += membership_mat[i][j] ** self.m * distance
        return loss

    def get_result(self):
        # Membership Matrix
        membership_mat = self.initialize_membership_matrix()
        curr = 0
        loss_old = 100000000
        loss_new = 100000000
        while curr < self.max_iter:
            cluster_centers = self.calculate_cluster_center(membership_mat)
            # cluster_centers: (k, num_attr). k means k clusters. And num_attr means num_attr features.
            # 更新 membership_mat 矩阵
            membership_mat = self.update_membership_value(membership_mat, cluster_centers)
            # cluster_labels = getClusters(membership_mat)
            loss_old = loss_new
            loss_new = self.get_loss(membership_mat, cluster_centers)
            if loss_new < loss_old:
                best_membership_mat = copy.deepcopy(membership_mat)
            curr += 1
            if self.show_detail:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('-time:{} - curr: {}/{} - loss: {}'.format(time_str, curr, self.max_iter, loss_new))
            if abs(loss_old - loss_new) / loss_old < 0.0001:
                print('(loss_old - loss_new) / loss_old < 0.01%, FCM finish')
                break
        print(best_membership_mat[0])
        return best_membership_mat
