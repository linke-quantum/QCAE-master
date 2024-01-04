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
# @Time     : 2023/12/12 20:09
# @Author   : deviludier @ LINKE
# @File     : exp1-information-compression.py
# @Software : PyCharm
import numpy as np
import os
import json
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
import qiskit.quantum_info as qi

from src.QCAE import QCAE
from src.QCAENoise import QCAENoise

from scipy.optimize import minimize


# ########################################################################################################
# Exp1.1: information compression on small scale, whose goal is to verify the similarity between loss function and validation
# ########################################################################################################
# 1. Initilize the latent system size and the trash system size
num_latent, num_trash = 4, 1
num_qubits = num_latent+num_trash

# 2. Generate the circuit list need for compression
mu, sigma = 0, 0.2
num_circuits = 10
target_op_list = []
qc = RealAmplitudes(num_qubits, reps=1).decompose()
for i in range(num_circuits):
    params = np.random.normal(mu, sigma, len(qc.parameters))
    target_op_list.append(qc.assign_parameters(parameters=params))

# 3. Initilize the QCAE and begin train
qcae = QCAE(num_latent=num_latent, num_trash=num_trash, reps=2)
res, hist = qcae.run(target_op_list=target_op_list, noValidation=False, max_it=50)
print("optimal parameters:", hist['x'][-1])
print("training loss:", hist['loss'])
print('Validation:', hist['validation'])

# 4. Save the results.
path = '../results/exp1/noiseless/SmallScale/' + str(num_circuits) + '-pqc/' + str(
    num_qubits) + '-qubit/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + str(num_qubits) + '-' + str(num_latent) + '-mu' + str(mu) + '-sigma' + str(
        sigma) + '.json', 'w') as file:
    json.dump(hist, file)
# 关闭文件
file.close()

# # ########################################################################################################
# # Exp1.2: information compression on large scale, whose goal is to verify the convergence ability of loss function on large
# # ########################################################################################################
# # 1. Initilize the latent system size and the trash system size
# num_latent, num_trash = 8, 1
# num_qubits = num_latent + num_trash
#
# # 2. Generate the circuit list need for compression
# mu, sigma = 0, 0.2
# num_circuits = 10
# target_op_list = []
# qc = RealAmplitudes(num_qubits, reps=1).decompose()
# for i in range(num_circuits):
#     params = np.random.normal(mu, sigma, len(qc.parameters))
#     target_op_list.append(qc.assign_parameters(parameters=params))
#
# # 3. Initilize the QCAE and begin train
# qcae = QCAE(num_latent=num_latent, num_trash=num_trash, reps=2)
# res, hist = qcae.run(target_op_list=target_op_list, noValidation=True, max_it=50)
#
# # 4. Save the results.
# path = '../results/exp1/noiseless/LargeScale/' + str(num_circuits) + '-pqc/' + str(
#     num_qubits) + '-qubit/'
# if not os.path.exists(path):
#     os.makedirs(path)
# with open(path + str(num_qubits) + '-' + str(num_latent) + '-mu' + str(mu) + '-sigma' + str(
#         sigma) + '.json', 'w') as file:
#     json.dump(hist, file)
# # 关闭文件
# file.close()

# # ########################################################################################################
# # Exp1.3: information compression under noise, whose goal is to verify the convergence ability of loss function under noise environments
# # ########################################################################################################
# # 1. Initilize the latent system size and the trash system size
# num_latent, num_trash = 3, 1
# num_qubits = num_latent + num_trash
#
# # 2. Generate the circuit list need for compression
# mu, sigma = 0, 0.2
# num_circuits = 10
# target_op_list = []
# qc = RealAmplitudes(num_qubits, reps=1).decompose()
# for i in range(num_circuits):
#     params = np.random.normal(mu, sigma, len(qc.parameters))
#     target_op_list.append(qc.assign_parameters(parameters=params))
#
# # 3. Initilize the QCAE and begin train
# qcae = QCAENoise(num_latent=num_latent, num_trash=num_trash, reps=3)
# error_name_list = ['depolarizing', 'thermal-relaxation', 'backend', None]
# error_name = error_name_list[2]
# res, hist = qcae.run(target_op_list=target_op_list, noValidation=False, max_it=50, error_name=error_name)
#
#
# # 4. Save the results.
# path = '../results/exp1/noise/' + error_name + '/'+ str(num_circuits) + '-pqc/' + str(
#     num_qubits) + '-qubit/'
# if not os.path.exists(path):
#     os.makedirs(path)
# with open(path + str(num_qubits) + '-' + str(num_latent) + '-mu' + str(mu) + '-sigma' + str(
#         sigma) + '.json', 'w') as file:
#     json.dump(hist, file)
# # 关闭文件
# file.close()

