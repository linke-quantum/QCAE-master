# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of th
# is license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2023/7/27 15:56
# @Author   : deviludier @ LINKE
# @File     : QCAE.py
# @Software : PyCharm

import numpy as np
from itertools import combinations

from mindquantum.simulator import Simulator, fidelity
from mindquantum.core.circuit import Circuit
from mindquantum.utils import random_circuit
from mindquantum.core.gates import I, X, Y, Z, H, RX, RY, RZ, CNOT, Measure, SWAP
from mindquantum.core.operators import QubitOperator, Hamiltonian

from scipy.optimize import minimize
from mindquantum.algorithm.nisq.chem.hardware_efficient_ansatz import HardwareEfficientAnsatz

from src.QCAE_state import get_choi_state, get_ME_state


class QCAE:
    def __init__(self,
                 num_qubits: int = None,
                 num_latent: int = None,
                 num_trash: int = None) -> None:
        '''
        initialize qcae model with several qubits numbers;
        :param num_qubits: the number of qubits of the target circuit(channel) needed to be compressed;
        :param num_latent: the number of qubits of the latent circuit(channel) after encoding;
        :param num_trash: the number of qubits of the ``trash" circuit(channel) after encoding;
        num_qubits = num_latent + num_trash;
        '''
        self.num_latent = num_latent
        self.num_trash = num_trash
        if num_qubits == None:
            self.num_qubits = self.num_latent + self.num_trash
        else:
            self.num_qubits = num_qubits
        try:
            if not (self.num_qubits == self.num_latent + self.num_trash):
                raise ValueError("num_qubits is the summation of num_latent and num_trash!")
        except ValueError as e:
            print("raise exception:", repr(e))
        self.pqc_ancila_num = 0

    def initial_state(self):
        '''
        prepare the initial state circuit
        :return: initial_state_circuit : numpy.array
        '''
        a = np.ones(2 ** self.num_latent)
        rho_mixed = np.diag(a)
        # rho_entangled = np.zeros(shape=(4 ** self.num_trash, 4 ** self.num_trash))
        # rho_entangled[0][0] = 1
        # rho_entangled[0][-1] = 1
        # rho_entangled[-1][0] = 1
        # rho_entangled[-1][-1] = 1

        rho_entangled = self.get_ME_state(self.num_trash)

        self.ME_mat = rho_entangled
        rho_initial = np.kron(rho_entangled, rho_mixed)
        return rho_initial

    def get_ME_state(self, num_qubits):
        qc = Circuit()
        for i in range(num_qubits):
            qc += H.on(i)
        for i in range(num_qubits):
            qc += CNOT.on(i + num_qubits, i)
        sim = Simulator('mqmatrix', qc.n_qubits)
        sim.apply_circuit(qc)
        ME_state = sim.get_qs()
        return ME_state

    def ansatz(self, num_qubits, param_prefix, reps=5):
        '''
        construct the ansatz circuit for the encoding process;
        :return: ansatz: Circuit
        '''
        # circuit = HardwareEfficientAnsatz(n_qubits=num_qubits, single_rot_gate_seq=[RY, RZ], depth=reps)
        qc = Circuit()

        for rep in range(reps):
            for i in range(num_qubits):
                qc += RY(param_prefix + str(i + num_qubits * rep)).on(i)
                qc += RZ(param_prefix + str(i + num_qubits * (rep + 1))).on(i)

            pair_list = list(combinations([i for i in range(num_qubits)], 2))
            for pair in pair_list:
                qc += X.on(pair[1], pair[0])

            for i in range(num_qubits):
                qc += RY(param_prefix + str(i + (rep + 2) * num_qubits)).on(i)
                qc += RZ(param_prefix + str(i + (rep + 3) * num_qubits)).on(i)
        return qc

    def construct_circuit(self, target_op: Circuit):
        '''
        construct the circuit in training process;
        the construction is:
            initial_state_circuit + encoding_ansatz_left + target_op + encoding_ansatz_right
        :param: target_op: the circuit(channel needed to process)
        :return: parameterized quantum circuit need to be training
        '''
        encoder_left = self.ansatz(num_qubits=target_op.n_qubits, param_prefix='theta')
        encoder_right = self.ansatz(num_qubits=target_op.n_qubits, param_prefix='beta')
        # print(encoder_left.summary())
        # print(encoder_right)

        execute_circuit = Circuit()
        execute_circuit += encoder_left
        execute_circuit += target_op
        execute_circuit += encoder_right

        return execute_circuit

    def ham(self):
        hams = []
        qubit_op = QubitOperator('', 0)
        for i in range(self.num_latent, self.num_qubits):
            m, n = i, i + self.num_trash

            # h_0 = Hamiltonian(QubitOperator(f'I{m}' + ' ' + f'I{n}', 1 / 4))
            h_1 = Hamiltonian(QubitOperator(f'Z{m}' + ' ' + f'Z{n}', 1 / 4))
            h_2 = Hamiltonian(QubitOperator(f'X{m}' + ' ' + f'X{n}', 1 / 4))
            h_3 = Hamiltonian(QubitOperator(f'Y{m}' + ' ' + f'Y{n}', -1 / 4))
            # qubit_op += h_1
            # qubit_op += h_2
            # qubit_op += h_3
            # hams.append(h_0)
            hams.append(h_1)
            hams.append(h_2)
            hams.append(h_3)

        # hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]

        self.hams = hams
        return qubit_op

    def cost_func(self, params):
        expec_sum = 0
        for target_op in self.target_op_list:
            circuit = self.construct_circuit(target_op=target_op)
            for i in range(self.num_qubits, self.num_qubits + self.num_trash):
                circuit += I.on(i)

            sim = Simulator('mqmatrix', circuit.n_qubits)

            rho_initial = self.initial_state()

            sim.set_qs(rho_initial)

            # print(sim.get_qs())
            pairs = zip(circuit.params_name, params)
            pr = {k: v for k, v in pairs}
            sim.apply_circuit(circuit, pr=pr)
            # rho_out = sim.get_partial_trace([i for i in range(self.num_latent)])
            # sim2 = Simulator('mqmatrix', 2*self.num_trash)
            # sim2.set_qs(self.ME_mat)
            # ME_rho = sim2.get_qs()
            # print(ME_rho)

            expectation = 0
            offset = int(len(self.hams) / 3) / 4
            for h in self.hams:
                expectation += sim.get_expectation(h).real
            expec = offset + expectation
            expec_sum += expec
        # return 1 - fidelity(rho_out, ME_rho)
        return 1 - expec_sum / len(self.target_op_list)
        # pass

    def callback(self, xk):
        print('Current iteration:', len(self.hist['x']))
        self.hist['x'].append(xk.tolist())
        self.hist['loss'].append(self.cost_func(xk))
        self.hist['validation'].append(self.validation(xk))

    def run(self, target_op_list):
        self.target_op_list = target_op_list
        circuit = self.construct_circuit(target_op=target_op_list[0])

        initial_point = np.random.random(len(circuit.params_name))

        self.validation(initial_point)

        self.ham()
        loss = self.cost_func(params=initial_point)
        print(loss)
        # state = circuit.get_qs(pr=dict(zip(circuit.params_name, initial_point)), ket=True)

        self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)], 'validation': [self.validation(initial_point)]}

        res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback, options={'maxiter':100})
        print(res)
        print("optimal parameters:",self.hist['x'])
        print("training loss:", self.hist['loss'])
        print('Validation:', self.hist['validation'])
        pass

    def validation(self, params):
        validation_sum = 0
        fid_list = []
        for target_op in self.target_op_list:
            circuit = self.construct_circuit(target_op=target_op)
            # print(sim.get_qs())
            pairs = zip(circuit.params_name, params)
            pr = {k: v for k, v in pairs}
            qc = circuit.apply_value(pr=pr)
            choi_state = get_choi_state(qc)

            sim = Simulator('mqmatrix', circuit.n_qubits * 2)
            sim.set_qs(choi_state)

            partial_trace_list = [i for i in range(self.num_latent, self.num_qubits)]
            for i in range(self.num_qubits + self.num_latent, self.num_qubits * 2):
                partial_trace_list.append(i)
            reduced_choi_state = sim.get_partial_trace(partial_trace_list)

            # print(reduced_choi_state)

            ME_state = get_ME_state(self.num_trash)

            half_recover_state = np.kron(ME_state, reduced_choi_state)

            self.recover_swap_circuit()
            # print(self.swap_qc)

            sim.reset()
            sim.set_qs(half_recover_state)
            sim.apply_circuit(self.swap_qc)

            recover_state = sim.get_qs()

            # print(recover_state)

            fid_list.append(fidelity(choi_state, recover_state))
        validation_sum = np.mean(fid_list)

        return validation_sum, fid_list

    def recover_swap_circuit(self):
        """

        :return:
        """
        swap_circuit = Circuit()
        for i in range(2 * self.num_qubits + 2 * self.pqc_ancila_num):
            swap_circuit += I.on(i)
        a = self.num_latent + self.pqc_ancila_num
        b = self.num_trash
        for i in range(a):
            for j in range(b):
                swap_circuit += SWAP.on([2 * self.num_qubits + 2 * self.pqc_ancila_num - 2 * self.num_trash - 1 - i + j,
                                         2 * self.num_qubits + 2 * self.pqc_ancila_num - 2 * self.num_trash + j - i])
        # print(swap_circuit)
        self.swap_qc = swap_circuit
        return swap_circuit


if __name__ == "__main__":
    print("begin test:")
    num_latent, num_trash = 1, 1
    num_qubits = num_latent + num_trash
    qcae = QCAE(num_latent=num_latent, num_trash=num_trash)

    target_op_list = []
    for i in range(2):
        target_op = random_circuit(num_latent, 10)

        for i in range(num_latent, num_qubits):
            target_op += I.on(i)

        target_op_list.append(target_op)
    qcae.run(target_op_list=target_op_list)

    # target_op = random_circuit(num_latent, 10)
    # target_op = Circuit()
    # target_op += I.on(0)
    # print(get_choi_state(target_op))
