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
# @Time     : 2023/12/16 19:45
# @Author   : deviludier @ LINKE
# @File     : exp2-anomaly-detection.py
# @Software : PyCharm
import numpy as np
import os
import json
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
import qiskit.quantum_info as qi
from qiskit.quantum_info import state_fidelity, partial_trace
from src.QCAE import QCAE
from scipy.optimize import minimize

def add_gaussian_noise(values, mean, std):
    noise = np.random.normal(mean, std, values.shape)
    noisy_values = values + noise
    return noisy_values

def choi_state_fidelity(qc1, qc2):
    choi_state1 = qi.DensityMatrix(data=qi.Choi(qc1).data/(2**qc1.num_qubits))
    choi_state2 = qi.DensityMatrix(data=qi.Choi(qc2).data/(2**qc2.num_qubits))
    return state_fidelity(choi_state1, choi_state2)
def generate_anomaly_detection_dataset(num_qubits):
    pqc = RealAmplitudes(num_qubits=num_qubits)

    params = np.random.random(len(pqc.parameters))
    # params = np.zeros(len(pqc.parameters))
    base_qc = pqc.assign_parameters(parameters=params)
    fid_list1 = []
    normal_qc_list = []
    for i in range(100):
        noise_params1 = add_gaussian_noise(params, mean=0, std=0.01)
        new_qc1 = pqc.assign_parameters(noise_params1)
        fid1 = choi_state_fidelity(base_qc, new_qc1)

        if fid1>0.95:
            normal_qc_list.append(new_qc1)
            fid_list1.append(fid1)


    fid_list2 = []
    abnormal_qc_list = []
    for i in range(100):
        noise_params2 = add_gaussian_noise(params, mean=0, std=0.5)
        new_qc2 = pqc.assign_parameters(noise_params2)
        fid2 = choi_state_fidelity(base_qc, new_qc2)

        if(fid2<0.2):
            abnormal_qc_list.append(new_qc2)
            fid_list2.append(fid2)

    return fid_list1, fid_list2, normal_qc_list, abnormal_qc_list


class AD_QCAE(QCAE):
    def recovery_fidelity(self, params, target_qc_list:list):
        ME_circuit = QuantumCircuit(2 * self.num_trash)
        for i in range(self.num_trash):
            ME_circuit.h(i)
        for i in range(self.num_trash):
            ME_circuit.cx(i, i + self.num_trash)
        ME_state = qi.DensityMatrix.from_instruction(ME_circuit)
        fid_list = []
        for target_op in target_qc_list:
            qc = self.QCAE_circuit(target_op=target_op)
            qc = qc.assign_parameters(parameters=params)
            choi_state = self.circuit_to_choi_state(qc)

            partial_trace_list = []
            for i in range(self.num_latent + self.pqc_ancila_num,
                           self.num_latent + self.num_trash + self.pqc_ancila_num):
                partial_trace_list.append(i)
            for i in range(2 * self.num_latent + self.num_trash + 2 * self.pqc_ancila_num,
                           2 * self.num_latent + 2 * self.num_trash + 2 * self.pqc_ancila_num):
                partial_trace_list.append(i)
            reduced_choi_state = partial_trace(choi_state, partial_trace_list)

            tmp_half_recovery_choi_state = ME_state.tensor(reduced_choi_state)

            self.recover_swap_circuit()

            half_recovery_choi_state = tmp_half_recovery_choi_state.evolve(self.swap_qc)
            fid = state_fidelity(choi_state, half_recovery_choi_state)
            fid_list.append(fid)
        return np.mean(fid_list), fid_list

    def callback(self, xk):
        # print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1], 'Validation:', self.hist['validation'][-1][0], np.var(self.hist['validation'][-1][1]))
        print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1])
        self.hist['x'].append(xk.tolist())
        self.hist['loss'].append(self.cost_func(xk))
        # self.hist['validation'].append(self.validation(xk))

    def run(self, target_op_list: list, max_it=100):
        self.target_op_list = target_op_list

        execute_qc = self.QCAE_circuit(target_op=target_op_list[0])

        initial_point = np.random.random(len(execute_qc.parameters))
        print(self.cost_func(params=initial_point))

        self.random_state_list = []
        for i in range(100):
            state = qi.random.random_density_matrix(2 ** (self.num_qubits + self.num_trash))
            self.random_state_list.append(state)

        self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)]}

        res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback, tol=1e-20,
                       options={'maxiter': max_it})
        # res = minimize_parallel(self.cost_func, initial_point)
        print(res)
        print("optimal parameters:", self.hist['x'][-1])
        print("training loss:", self.hist['loss'])
        # print('Validation:', self.hist['validation'])

        return res

if __name__ == '__main__':
    num_latent, num_trash = 5,1
    num_qubits = num_latent + num_trash

    fid_list1, fid_list2, normal_qc_list, abnormal_qc_list = generate_anomaly_detection_dataset(num_qubits=num_qubits)
    print(len(fid_list1))
    print(len(fid_list2))
    print(np.mean(fid_list1))
    print(np.mean(fid_list2))

    train_circuit_number = 10
    target_op_list = normal_qc_list[:train_circuit_number]

    test_circuit_number = 40
    test_normal_qc = normal_qc_list[train_circuit_number:train_circuit_number+test_circuit_number]
    test_abnormal_qc = abnormal_qc_list[:test_circuit_number]

    # print('normal')
    # for qc in test_normal_qc:
    #     print(choi_state_fidelity(qc, normal_qc_list[0]))
    # print('abnormal')
    # for qc in test_abnormal_qc:
    #     print(choi_state_fidelity(qc, normal_qc_list[0]))
    print("Normal fidelity: ", np.mean(fid_list1), fid_list1)
    print("Abnormal fidelity: ", np.mean(fid_list2), fid_list2)

    qcae = AD_QCAE(num_latent=num_latent, num_trash=num_trash)
    res = qcae.run(target_op_list=target_op_list, max_it=10)

    # params = np.random.random(len(qcae.QCAE_circuit(target_op=target_op_list[0]).parameters))
    params = res.x
    normal_score = qcae.recovery_fidelity(params=params, target_qc_list=test_normal_qc)
    abnormal_score = qcae.recovery_fidelity(params=params, target_qc_list=test_abnormal_qc)

    print("Normal fidelity: ", np.mean(fid_list1), fid_list1)
    print("Abnormal fidelity: ", np.mean(fid_list2), fid_list2)
    print("Normal score: ", normal_score)
    print("Abnormal score: ", abnormal_score)

    path = '../results/exp2/noiseless/' + str(num_qubits) + '-qubit/' + str(
        num_qubits) + '-' + str(num_latent) + '/'

    tmp_dict = {"Normal fidelity: ": [np.mean(fid_list1), fid_list1], "Abnormal fidelity: ": [np.mean(fid_list2), fid_list2], "Normal score: ": normal_score,
                "Abnormal score: ": abnormal_score}
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'AnomalyScores.json', 'w') as file:
        json.dump(tmp_dict, file)