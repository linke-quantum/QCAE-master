"""
the experiment2: anomaly detection
"""
import numpy as np
import random
import os
import json
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2

import qiskit.quantum_info as qi
from qiskit.quantum_info import Choi, random_quantum_channel, partial_trace, state_fidelity, DensityMatrix
# from qiskit.providers.aer.noise import depolarizing_error

from qiskit.providers.fake_provider import FakeNairobi
from qiskit.primitives import BackendEstimator, Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from src.QCAE import QCAE

from scipy.optimize import minimize

def get_element_with_probability(prob_list, element_list):
    return random.choices(element_list, weights=prob_list)[0]

def get_circuit_fidelity(qc1, qc2):
    state1 = DensityMatrix(Choi(qc1).data/2**qc1.num_qubits)
    state2 = DensityMatrix(Choi(qc2).data/2**qc2.num_qubits)
    return state_fidelity(state1, state2)

class Denoise_QCAE(QCAE):
    def cost_func(self, params=None):
        infid_list = []
        for target_op in self.target_op_list:
            qc = self.initial_state_circuit()
            qc.barrier()
            QCAE_circuit = self.QCAE_circuit(target_op=target_op)
            qc.append(QCAE_circuit, [i for i in range(self.num_latent, self.num_latent * 2 + self.num_trash)])

            hams = self.hams()

            # estimator = BackendEstimator(backend=FakeNairobi())
            estimator = Estimator()
            fid1 = estimator.run(circuits=qc,
                                 observables=[hams],
                                 parameter_values=params).result().values

            infid = 1 - fid1[0]
            infid_list.append(infid)
        return np.mean(infid_list)
    def validation(self, params):
        fid_sum = 0
        fid_list = []
        original_qc = self.QCAE_circuit(target_op=self.original_qc).assign_parameters(params)
        original_Choi_state = qi.DensityMatrix(Choi(original_qc).data/2**original_qc.num_qubits)

        ME_circuit = QuantumCircuit(2 * self.num_trash)
        for i in range(self.num_trash):
            ME_circuit.h(i)
        for i in range(self.num_trash):
            ME_circuit.cx(i, i + self.num_trash)
        ME_state = qi.DensityMatrix.from_instruction(ME_circuit)

        for target_op in self.target_op_list:
            qc = self.QCAE_circuit(target_op=target_op)
            qc = qc.assign_parameters(params)
            choi_execute = Choi(qc)
            choi_state = qi.DensityMatrix(choi_execute.data / (2 ** (self.num_qubits + self.pqc_ancila_num)))
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
            # print(choi_state.is_valid(), half_recovery_choi_state.is_valid())
            fid = state_fidelity(original_Choi_state, half_recovery_choi_state)
            fid_sum += fid
            fid_list.append(fid)
        return 1 - fid_sum / len(self.target_op_list), fid_list
    def run(self, target_op_list, original_qc, max_it=100):
        self.original_qc = original_qc
        self.target_op_list = target_op_list

        # self.noisy_estimator = NoiseEstimator('depolarizing')
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
        # print(self.validation(res.x))
        # validation_list = self.validation(res.x)[1]
        # print("mean:", np.mean(validation_list))
        # print("var:", np.var(validation_list))
        return res
    
if __name__ == "__main__":
    num_latent, num_trash = 4,1

    depth = 1
    num_qubit = num_latent + num_trash

    pqc = RealAmplitudes(num_qubits=num_qubit)
    mu = 0
    sigma_list = [0.1,0.2,0.3,0.4,0.5]
    # sigma_list = [0.1]
    for sigma in sigma_list:
        params = np.random.normal(mu, sigma, len(pqc.parameters))
        qc = pqc.assign_parameters(parameters=params)
        # qc = random_circuit(num_qubit, depth=depth)

        p_list = [0.01,0.02,0.03,0.04,0.05, 0.06, 0.07, 0.08, 0.09]
        # p_list = [0.2]
        for p in p_list:
            error = depolarizing_error(p, num_qubit)
            # print(error.circuits)
            # print(error.probabilities)
            target_op_list = []

            fid_list = []
            pqc_numbers = 40
            for i in range(pqc_numbers):
                err = get_element_with_probability(error.probabilities, error.circuits)
                # print(err)
            #     err = execute_events(error.probabilities, error.circuits)
            #     # print(err)
                tmp_qc = QuantumCircuit(num_qubit)
                tmp_qc.append(qc, [i for i in range(num_qubit)])
                tmp_qc.append(err, [i for i in range(num_qubit)])
                tmp_fid = get_circuit_fidelity(qc, tmp_qc)
                fid_list.append(tmp_fid)

                target_op_list.append(tmp_qc)

            print("noise:")
            noise_mean = sum(fid_list) / pqc_numbers
            noise_var = np.var(fid_list)
            print("mean:", sum(fid_list) / pqc_numbers)
            print("var:", np.var(fid_list))
            qcae = Denoise_QCAE(num_latent=num_latent, num_trash=num_trash, reps=3)
            res = qcae.run(target_op_list=target_op_list, original_qc = qc, max_it = 100)

            validation_list = qcae.validation(res.x)[1]
            print("noise:")
            noise_mean = sum(fid_list) / pqc_numbers
            noise_var = np.var(fid_list)
            print("mean:", sum(fid_list) / pqc_numbers)
            print("var:", np.var(fid_list))
            print("denoise")
            denoise_mean = np.mean(validation_list)
            denoise_var = np.var(validation_list)
            print("mean:", np.mean(validation_list))
            print("var:", np.var(validation_list))
            path = '../results/exp3/noiseless/depolarizing/' + str(pqc_numbers) + '-pqc/' + str(num_qubit) + '-qubit/' + str(num_qubit) + '-' + str(num_latent) + '-norm(' + str(mu) + "," + str(sigma) + ')/'

            tmp_dict = {"noise_mean": noise_mean, "noise_var": noise_var, "denoise_mean": denoise_mean, "denoise_var": denoise_var}
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + str(p) +'.json', 'w') as file:
                json.dump(tmp_dict, file)