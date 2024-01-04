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
# @Time     : 2023/12/5 20:52
# @Author   : deviludier @ LINKE
# @File     : qcae.py
# @Software : PyCharm
import time
import os
import json
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library.n_local.two_local import TwoLocal
from qiskit.primitives import Estimator, BackendEstimator
from qiskit.primitives.utils import bound_circuit_to_instruction
import qiskit.quantum_info as qi
from qiskit.quantum_info import SparsePauliOp, partial_trace, state_fidelity
from qiskit.providers.fake_provider.fake_qasm_backend import FakeQasmBackend

from scipy.optimize import minimize
from optimparallel import minimize_parallel

from qiskit.circuit.random import random_circuit

import matplotlib.pyplot as plt


class QCAE:
    def __init__(
            self,
            num_qubits: int = None,
            num_latent: int = None,
            num_trash: int = None,
            reps: int = 5,
    ) -> None:
        if not num_latent is None:
            self.num_latent = num_latent
        if not num_trash is None:
            self.num_trash = num_trash
        if not num_qubits is None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = self.num_latent + self.num_trash
        self.pqc_ancila_num = 0
        self.reps = reps

    # RealAmplitudes ansatz
    def ansatz(self, num_qubits, parameter_prefix=None, ansatz_name='HardwareEfficient',reps=None):
        '''
        :param num_qubit: anzatz的线路的比特数目
        :param parameter_prefix: 参数名前缀
        :param reps: ansatz重复的层数
        :return: ansatz:含参数量子线路
        '''
        if ansatz_name == 'RealAmplititudes':
            if reps == None:
                return RealAmplitudes(num_qubits, reps=5, parameter_prefix=parameter_prefix).decompose()
            else:
                return RealAmplitudes(num_qubits, reps=reps, parameter_prefix=parameter_prefix).decompose()
        elif ansatz_name == 'HardwareEfficient':
            if reps == None:
                return TwoLocal(num_qubits=num_qubits, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', entanglement='full', reps=5, parameter_prefix=parameter_prefix).decompose()
            else:
                return TwoLocal(num_qubits=num_qubits, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', entanglement='full', reps=reps, parameter_prefix=parameter_prefix).decompose()

    # 构造QCAE压缩过程的量子线路
    def QCAE_circuit(self, target_op):
        '''
        :param target_op: 需要执行的目标操作
        :return: 含参数量子线路
        '''
        # 初始化QCAE线路, 比特数目为self.num_latent + self.num_trash：
        total_qubit = self.num_latent + self.num_trash + self.pqc_ancila_num
        circuit = QuantumCircuit(total_qubit)

        # 添加encoding 和 decoding线路到输入目标操作的两侧；其中，encoding和decoding线路是含参数量子线路，但是维度要高于含参数量子线路
        circuit.append(self.ansatz(total_qubit, parameter_prefix='theta', reps=self.reps).decompose(),
                       range(0, total_qubit))
        circuit.barrier()
        circuit.append(target_op, range(self.pqc_ancila_num, self.num_qubits + self.pqc_ancila_num))
        for i in range(self.pqc_ancila_num):
            circuit.reset(i)
        circuit.barrier()
        circuit.append(self.ansatz(total_qubit, parameter_prefix='beta', reps=self.reps).decompose(),
                       range(0, total_qubit))
        circuit.barrier()
        return circuit

    def get_ME_circuit(self, num_qubits):
        qc = QuantumCircuit(num_qubits * 2)
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.cx(i, i + num_qubits)
        return qc

    def initial_state_circuit(self):
        qc = QuantumCircuit(self.num_qubits * 2)

        ME_qc1 = self.get_ME_circuit(self.num_latent)
        ME_qc2 = self.get_ME_circuit(self.num_trash)
        qc.append(ME_qc1, [i for i in range(self.num_latent * 2)])
        qc.append(ME_qc2, [i for i in range(self.num_latent * 2, self.num_qubits * 2)])

        return qc

    def hams(self):
        str = ''
        for i in range(self.num_qubits * 2):
            str += 'I'

        d = 4 * (2 ** (self.num_trash - 1))
        hams = []
        for i in range(self.num_trash):
            m, n = i, i + self.num_trash

            h0 = (str, 1 / d)

            tmp_str = list(str)
            tmp_str[m] = 'X'
            tmp_str[n] = 'X'
            h1 = (''.join(tmp_str), 1 / d)

            tmp_str = list(str)
            tmp_str[m] = 'Y'
            tmp_str[n] = 'Y'
            h2 = (''.join(tmp_str), -1 / d)

            tmp_str = list(str)
            tmp_str[m] = 'Z'
            tmp_str[n] = 'Z'
            h3 = (''.join(tmp_str), 1 / d)

            hams.append(h0)
            hams.append(h1)
            hams.append(h2)
            hams.append(h3)
            # print(hams)
        return SparsePauliOp.from_list(hams)

    def cost_func(self, params=None):
        infid_list = []
        for target_op in self.target_op_list:
            qc = self.initial_state_circuit()
            qc.barrier()
            QCAE_circuit = self.QCAE_circuit(target_op=target_op)
            qc.append(QCAE_circuit, [i for i in range(self.num_latent, self.num_latent * 2 + self.num_trash)])

            hams = self.hams()
            estimator = Estimator()
            fid1 = estimator.run(circuits=qc,
                                 observables=[hams],
                                 parameter_values=params).result().values

            infid = 1 - fid1[0]
            infid_list.append(infid)
        return np.mean(infid_list)

    def callback(self, xk):
        # print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1], 'Validation:', self.hist['validation'][-1][0], np.var(self.hist['validation'][-1][1]))
        print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1], 'Validation:', self.hist['validation'][-1])
        self.hist['x'].append(xk.tolist())
        self.hist['loss'].append(self.cost_func(xk))
        self.hist['validation'].append(self.validation(xk))

    def callback_NoValidation(self, xk):
        # print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1], 'Validation:', self.hist['validation'][-1][0], np.var(self.hist['validation'][-1][1]))
        print('Current iteration:', len(self.hist['x']), 'Loss:', self.hist['loss'][-1])
        self.hist['x'].append(xk.tolist())
        self.hist['loss'].append(self.cost_func(xk))

    def run(self, target_op_list: list, max_it=100, noValidation=False):
        self.target_op_list = target_op_list

        execute_qc = self.QCAE_circuit(target_op=target_op_list[0])

        initial_point = np.random.random(len(execute_qc.parameters))
        print(self.cost_func(params=initial_point))

        self.random_state_list = []
        for i in range(100):
            state = qi.random.random_density_matrix(2 ** (self.num_qubits + self.num_trash))
            self.random_state_list.append(state)

        if noValidation:
            self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)]}

            res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback_NoValidation, tol=1e-20,
                           options={'maxiter': max_it})
        else:
            self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)],
                         'validation': [self.validation(initial_point)]}

            res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback, tol=1e-20,
                           options={'maxiter': max_it})
        # res = minimize_parallel(self.cost_func, initial_point)
        print(res)
        print("optimal parameters:", self.hist['x'][-1])
        print("training loss:", self.hist['loss'])
        # print('Validation:', self.hist['validation'])

        return res, self.hist

        # path = '../results/exp1/noiseless' + str(len(self.target_op_list)) + '-pqc/' + str(self.num_qubits) + '-qubit/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # with open(path + str(self.num_qubits) + '-' + str(self.num_latent) + '-mu' + str(mu) + '-sigma' + str(sigma) + '.json', 'w') as file:
        #     json.dump(self.hist, file)
        #
        # # 关闭文件
        # file.close()
        #
        # validation = [data[0] for data in self.hist['validation']]
        # plt.plot(self.hist['loss'])
        # plt.plot(validation)
        # plt.show()

    def validation_swap_circuit(self, num_qubits):
        qc = QuantumCircuit(num_qubits * 2)
        for i in range(num_qubits):
            qc.swap(i, i + num_qubits)
        return qc

    def circuit_to_choi_state(self, qc):
        cir = QuantumCircuit(2 * qc.num_qubits)
        for i in range(qc.num_qubits):
            cir.h(i)
        for i in range(qc.num_qubits):
            cir.cx(i, i + qc.num_qubits)
        cir.append(qc, [i for i in range(qc.num_qubits, 2*qc.num_qubits)])
        return qi.Statevector.from_instruction(cir)

    def recover_swap_circuit(self):
        swap_circuit = QuantumCircuit(2*self.num_qubits + 2*self.pqc_ancila_num)
        a = self.num_latent + self.pqc_ancila_num
        b = self.num_trash
        for i in range(a):
            for j in range(b):
                swap_circuit.swap(2*self.num_qubits+2*self.pqc_ancila_num-2*self.num_trash - 1 - i + j, 2*self.num_qubits+2*self.pqc_ancila_num - 2*self.num_trash + j - i)
        # print(swap_circuit)
        self.swap_qc = swap_circuit
        return swap_circuit

    def validation(self, params):
        ME_circuit = QuantumCircuit(2*self.num_trash)
        for i in range(self.num_trash):
            ME_circuit.h(i)
        for i in range(self.num_trash):
            ME_circuit.cx(i, i+self.num_trash)
        ME_state = qi.DensityMatrix.from_instruction(ME_circuit)
        fid_list = []
        for target_op in self.target_op_list:
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
        return 1-np.mean(fid_list), fid_list


if __name__ == "__main__":
    num_latent, num_trash = 5, 1
    num_qubits = num_latent + num_trash
    qcae = QCAE(num_latent=num_latent, num_trash=num_trash)

    mu, sigma = 0, 0.2
    pqc = RealAmplitudes(num_qubits=num_qubits, reps=1)
    target_op_list = []
    for i in range(10):
        params = np.random.normal(mu, sigma, len(pqc.parameters))
        target_op_list.append(pqc.assign_parameters(parameters=params))
    qcae.run(target_op_list=target_op_list, noValidation=True)
    # qcae.run([QuantumCircuit(num_qubits)])

