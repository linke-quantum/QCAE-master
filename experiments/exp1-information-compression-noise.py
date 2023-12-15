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

# Noise Estimator
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

def NoiseEstimator(noise_name):
    if noise_name == 'depolarizing':
        # construct the error model
        # 1. depolarizing error
        noise_model = NoiseModel()
        cx_depolarizing_prob = 0.01
        gate_depolarizing_prob = 0.01
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(gate_depolarizing_prob, 1), ["rx", 'ry', 'rz']
        )
        noisy_estimator = Estimator(
            backend_options={"noise_model": noise_model}, approximation=True
        )
        return noisy_estimator
    elif noise_name == 'thermal-relaxation':
        # 2. T1/T2 error
        # T1 and T2 values for qubits 0-3
        T1s = np.random.normal(50e3, 10e3, 4)  # Sampled from normal distribution mean 50 microsec
        T2s = np.random.normal(70e3, 10e3, 4)  # Sampled from normal distribution mean 50 microsec

        # Truncate random T2s <= T1s
        T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

        # Instruction times (in nanoseconds)
        time_u1 = 0  # virtual gate
        time_u2 = 50  # (single X90 pulse)
        time_u3 = 100  # (two X90 pulses)
        time_cx = 300
        time_reset = 1000  # 1 microsecond
        time_measure = 1000  # 1 microsecond

        # QuantumError objects
        errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                        for t1, t2 in zip(T1s, T2s)]
        errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                          for t1, t2 in zip(T1s, T2s)]
        errors_u1 = [thermal_relaxation_error(t1, t2, time_u1)
                     for t1, t2 in zip(T1s, T2s)]
        errors_u2 = [thermal_relaxation_error(t1, t2, time_u2)
                     for t1, t2 in zip(T1s, T2s)]
        errors_u3 = [thermal_relaxation_error(t1, t2, time_u3)
                     for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
            thermal_relaxation_error(t1b, t2b, time_cx))
            for t1a, t2a in zip(T1s, T2s)]
            for t1b, t2b in zip(T1s, T2s)]

        # Add errors to noise model
        noise_thermal = NoiseModel()
        for j in range(4):
            noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
            noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
            noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
            noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
            noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(4):
                noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        # print(noise_thermal)
        noisy_estimator = Estimator(
            backend_options={"noise_model": noise_thermal}, approximation=True
        )
        return noisy_estimator
    else:
        return 0

class QCAEInformationCompressionNoise(QCAE):
    def cost_func(self, params=None):
        infid_list = []
        for target_op in self.target_op_list:
            qc = self.initial_state_circuit()
            qc.barrier()
            QCAE_circuit = self.QCAE_circuit(target_op=target_op)
            qc.append(QCAE_circuit, [i for i in range(self.num_latent, self.num_latent * 2 + self.num_trash)])

            hams = self.hams()

            fid1 = self.noisy_estimator.run(circuits=qc,
                                 observables=[hams],
                                 parameter_values=params).result().values

            infid = 1 - fid1[0]
            infid_list.append(infid)
        return np.mean(infid_list)

    def run(self, target_op_list: list, max_it=100, error_name=None):
        self.target_op_list = target_op_list
        self.noisy_estimator = NoiseEstimator(error_name)

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

        path = '../results/exp1/noise/' + error_name + str(len(self.target_op_list)) + '-pqc/' + str(self.num_qubits) + '-qubit/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + str(self.num_qubits) + '-' + str(self.num_latent) + '-mu' + str(mu) + '-sigma' + str(sigma) + '.json', 'w') as file:
            json.dump(self.hist, file)
        # 关闭文件
        file.close()
        return 0

if __name__ == '__main__':
    num_latent, num_trash = 3,1
    num_qubits = num_latent+num_trash

    # construct the input quantum circuits list
    mu, sigma = 0, 0.2
    num_circuits = 10
    pqc = RealAmplitudes(num_qubits=num_qubits, reps=1)
    target_op_list = []
    for i in range(num_circuits):
        params = np.random.normal(mu, sigma, len(pqc.parameters))
        target_op_list.append(pqc.assign_parameters(parameters=params))

    qcae = QCAEInformationCompressionNoise(num_latent=num_latent, num_trash=num_trash)
    #
    # error_name = ['depolarizing' | 'thermal-relaxation']
    qcae.run(target_op_list=target_op_list, max_it=50, error_name='depolarizing')