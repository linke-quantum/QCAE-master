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
import qiskit.quantum_info as qi
from qiskit.circuit.library import RealAmplitudes

from src.QCAE import QCAE

from scipy.optimize import minimize
class QCAEInformationCompressionLargeScale(QCAE):
    def validation(self, params):
        return 0
    def run(self, target_op_list: list, max_it=100):
        self.target_op_list = target_op_list

        execute_qc = self.QCAE_circuit(target_op=target_op_list[0])

        initial_point = np.random.random(len(execute_qc.parameters))
        print(self.cost_func(params=initial_point))

        self.random_state_list = []
        for i in range(100):
            state = qi.random.random_density_matrix(2 ** (self.num_qubits + self.num_trash))
            self.random_state_list.append(state)

        self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)],
                     'validation': [self.validation(initial_point)]}

        res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback, tol=1e-20,
                       options={'maxiter': max_it})
        print(res)
        print("optimal parameters:", self.hist['x'][-1])
        print("training loss:", self.hist['loss'])
        print('Validation:', self.hist['validation'])

        path = '../results/exp1/noiseless/LargeScale/' + str(len(self.target_op_list)) + '-pqc/' + str(self.num_qubits) + '-qubit/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + str(self.num_qubits) + '-' + str(self.num_latent) + '-mu' + str(mu) + '-sigma' + str(sigma) + '.json', 'w') as file:
            json.dump(self.hist, file)
        # 关闭文件
        file.close()
        return 0

if __name__ == '__main__':
    num_latent, num_trash = 10,1
    num_qubits = num_latent+num_trash

    # construct quantum circuits list
    mu, sigma = 0, 0.2
    num_circuits = 10
    pqc = RealAmplitudes(num_qubits=num_qubits, reps=1)
    target_op_list = []
    for i in range(num_circuits):
        params = np.random.normal(mu, sigma, len(pqc.parameters))
        target_op_list.append(pqc.assign_parameters(parameters=params))

    qcae = QCAEInformationCompressionLargeScale(num_latent=num_latent, num_trash=num_trash)

    qcae.run(target_op_list=[QuantumCircuit(num_qubits)], max_it=50)