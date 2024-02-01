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
import numpy as np
import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '../exp3/noise2/depolarizing/100-pqc/4-qubit/' + '4-3-norm(0,0.1)/'
    # path = '../exp3/noiseless/depolarizing/40-pqc/4-qubit/' + '4-3-norm(0,0.1)/'
    p_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    noise_mean = []
    noise_std = []
    denoised_mean = []
    denoised_std = []
    for p in p_list:
        file_path = path + str(p) + '.json'
        data = json.load(open(file_path))
        noise_mean.append(data['noise_mean'])
        noise_std.append(data['noise_var'])
        denoised_mean.append(data['denoise_mean'])
        denoised_std.append(data['denoise_var'])
    plt.plot(noise_mean, marker='o', label='Noise mean')
    plt.plot(denoised_mean, marker='>', label='Denoised mean')
    plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小

    plt.xlabel('p in the Depolarizing error', fontsize=14)
    plt.ylabel('Reconstruction Fidelity', fontsize=14)
    X = [_ for _ in range(len(p_list))]
    plt.xticks(X, p_list, fontsize=14, rotation=20)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    pathw = '../figs/exp3/'
    if not os.path.exists(pathw):
        os.makedirs(pathw)
    fig_path = pathw + 'exp3-Denoise-impact.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()