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
# @Time     : 2023/12/17 21:02
# @Author   : deviludier @ LINKE
# @File     : exp2-draw.py
# @Software : PyCharm
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    num_qubits, num_latent = 5, 4
    is_noise = True
    if is_noise:
        path = '../exp2/noise/depolarizing' + str(num_qubits) + '-qubit/' + str(num_qubits) + '-' + str(num_latent) + '/AnomalyScores.json'
        pathw = "../figs/exp2/depolarizing/"
    else:
        path = '../exp2/noiseless/' + str(num_qubits) + '-qubit/' + str(num_qubits) + '-' + str(
            num_latent) + '/AnomalyScores.json'
        pathw = "../figs/exp2/noiseless/"

    data = json.load(open(path))
    print(data)
    print(data['Normal fidelity: '])
    print(data['Abnormal fidelity: '])
    print(data['Normal score: '])
    print(data['Abnormal score: '])
    normal_scores = data['Normal score: '][1]
    anomalous_scores = data['Abnormal score: '][1]
    #
    sns.histplot(data=normal_scores, bins=10, label='Normal')
    sns.histplot(data=anomalous_scores, bins=10, label='Anomalous')
    # sns.kdeplot(data=normal_scores)
    # sns.kdeplot(data=anomalous_scores)
    plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    plt.xlabel('Anomaly Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    #

    if not os.path.exists(pathw):
        os.makedirs(pathw)
    fig_path = pathw + "exp2-density_hist.pdf"
    # fig_path = path + "density_hist-norm-random.pdf"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()

    # import matplotlib.pyplot as plt
    # from scipy.stats import gaussian_kde
    #
    # # 假设你有两个包含重构误差的列表：normal_errors和anomaly_errors
    # normal_errors = normal_scores
    # anomaly_errors = anomalous_scores
    #
    # # 绘制两个重构误差列表的分布图
    # plt.hist(normal_errors, bins=15, alpha=0.8, label='Normal Data')
    # plt.hist(anomaly_errors, bins=15, alpha=0.8, label='Anomaly Data')
    # plt.xlabel('Reconstruction Error')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Reconstruction Errors')
    # plt.legend()
    # plt.show()