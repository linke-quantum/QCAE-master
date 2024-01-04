# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2023/12/17 20:54
# @Author   : deviludier @ LINKE
# @File     : exp1-draw.py
# @Software : PyCharm
import json

import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ###############################################################################
    # ############ 1. training and validation with small scale systems ##############
    # ###############################################################################
    # path = '../../results/exp1/noiseless/SmallScale/10-pqc/5-qubit/5-4-mu0-sigma0.2.json'
    # # path = '../results/exp1/noiseless/SmallScale/10-pqc/5-qubit/5-4-mu0-sigma0.2.json'
    # data = json.load(open(path))
    # print(data)
    # train_loss = data['loss']
    # val_loss = data['validation']
    # print(train_loss)
    # print(val_loss)
    #
    # print(val_loss)
    # val_data = []
    # for loss in val_loss:
    #     val_data.append(loss[1])
    #
    # # 对每个epoch的val_data进行1-操作
    # val_data_minus_1 = [[1 - x for x in epoch_data] for epoch_data in val_data]
    #
    # # 取出每5个epoch的验证数据
    # val_data_5_epochs = val_data_minus_1[::5]
    #
    # # 计算每组验证数据的均值和方差
    # val_mean = [np.mean(epoch_data) for epoch_data in val_data_5_epochs]
    # val_std = [np.std(epoch_data) for epoch_data in val_data_5_epochs]
    # print(train_loss[-1], val_mean[-1], val_std[-1])
    # # 每5个epoch取一个点
    # val_x = list(range(0, len(val_mean) * 5, 5))
    #
    # # 绘制折线图
    # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    #
    # plt.plot(train_loss, label='Train Loss')
    # plt.errorbar(val_x, val_mean, yerr=val_std, label='Validation Data', fmt='o-', capsize=3)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Infidelity', fontsize=14)
    # plt.legend(fontsize=14)
    #
    # pathw = '../figs/exp1/'
    # if not os.path.exists(pathw):
    #     os.makedirs(pathw)
    # fig_path = pathw + 'exp1-5-4qubits-norm(0,0.2)-10.pdf'
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.show()

    # ###############################################################################
    # ############2.  training on large scale systems(9,10,11 qubits) #############
    # ###############################################################################
    # qubit_pair_list = [[9, 8], [10, 9], [11, 9]]
    # marker_list = ['o', 's', '>']
    # for i in range(len(qubit_pair_list)):
    #     qubit_pair = qubit_pair_list[i]
    #     num_qubits, num_latent = qubit_pair[0], qubit_pair[1]
    #     path = '../../results/exp1/noiseless/LargeScale/10-pqc/' + str(num_qubits) + '-qubit/' + str(num_qubits) + '-' + str(num_latent) + '-mu0-sigma0.2.json'
    #     data = json.load(open(path))
    #     print(data)
    #     train_loss = data['loss']
    #     print(train_loss)
    #
    #     # 绘制折线图
    #     plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    #     plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    #
    #     plt.plot(train_loss, marker=marker_list[i], label=str(num_qubits) + '-' + str(num_latent))
    #
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Infidelity', fontsize=14)
    # plt.legend(fontsize=14)
    #
    # pathw = '../figs/exp1/'
    # if not os.path.exists(pathw):
    #     os.makedirs(pathw)
    # fig_path = pathw + 'exp1-LargeScale.pdf'
    # plt.savefig(fig_path, bbox_inches='tight')
    # plt.show()

    ###############################################################################
    ############ 3. training on noise estimator ##############
    ###############################################################################
    error_name_list = ['depolarizing', 'thermal-relaxation', 'backend', None]
    error_name = error_name_list[1]
    path = '../../results/exp1/noise/' + error_name + '10-pqc/4-qubit/4-3-mu0-sigma0.2.json'
    data = json.load(open(path))
    print(data)
    train_loss = data['loss']
    val_loss = data['validation']
    print(train_loss)
    print(val_loss)

    print(val_loss)
    val_data = []
    for loss in val_loss:
        val_data.append(loss[1])

    # 对每个epoch的val_data进行1-操作
    val_data_minus_1 = [[1 - x for x in epoch_data] for epoch_data in val_data]

    # 取出每5个epoch的验证数据
    val_data_5_epochs = val_data_minus_1[::5]

    # 计算每组验证数据的均值和方差
    val_mean = [np.mean(epoch_data) for epoch_data in val_data_5_epochs]
    val_std = [np.std(epoch_data) for epoch_data in val_data_5_epochs]
    print(train_loss[-1], val_mean[-1], val_std[-1])
    # 每5个epoch取一个点
    val_x = list(range(0, len(val_mean) * 5, 5))

    # 绘制折线图
    plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小

    plt.plot(train_loss, label='Train Loss')
    plt.errorbar(val_x, val_mean, yerr=val_std, label='Validation Data', fmt='o-', capsize=3)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Reconstruction Error', fontsize=14)
    plt.legend(fontsize=14)

    pathw = '../figs/exp1/'
    if not os.path.exists(pathw):
        os.makedirs(pathw)
    fig_path = pathw + 'exp1-noise-' + error_name +'.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()