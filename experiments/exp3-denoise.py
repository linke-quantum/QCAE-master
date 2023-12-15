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

from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from src.QCAE import QCAE

from scipy.optimize import minimize

def get_element_with_probability(prob_list, element_list):
    return random.choices(element_list, weights=prob_list)[0]

def get_circuit_fidelity(qc1, qc2):
    state1 = DensityMatrix(Choi(qc1).data/2**qc1.num_qubits)
    state2 = DensityMatrix(Choi(qc2).data/2**qc2.num_qubits)
    return state_fidelity(state1, state2)

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

class Denoise_QCAE(QCAE):
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

        self.noisy_estimator = NoiseEstimator('depolarizing')
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
    num_latent, num_trash = 1,1

    depth = 5
    num_qubit = num_latent + num_trash

    pqc = RealAmplitudes(num_qubits=num_qubit)
    mu = 0
    sigma_list = [0.1,0.2,0.3,0.4,0.5]
    # sigma_list = [0.1]
    for sigma in sigma_list:
        params = np.random.normal(mu, sigma, len(pqc.parameters))
        qc = pqc.assign_parameters(parameters=params)
        # qc = random_circuit(num_qubit, depth=depth)

        p_list = [0.1,0.2,0.3,0.4,0.5]
        # p_list = [0.1]
        for p in p_list:
            error = depolarizing_error(p, num_qubit)
            # print(error.circuits)
            # print(error.probabilities)
            target_op_list = []

            fid_list = []
            pqc_numbers = 10
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


            qcae = Denoise_QCAE(num_latent=num_latent, num_trash=num_trash)
            res = qcae.run(target_op_list=target_op_list, original_qc = qc, max_it = 50)

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
            path = '../results/exp3/noise/depolarizing/' + str(pqc_numbers) + '-pqc/' + str(num_qubit) + '-qubit/' + str(num_qubit) + '-' + str(num_latent) + '-norm(' + str(mu) + "," + str(sigma) + ')/'

            tmp_dict = {"noise_mean": noise_mean, "noise_var": noise_var, "denoise_mean": denoise_mean, "denoise_var": denoise_var}
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # with open(path + str(p) +'.json', 'w') as file:
            #     json.dump(tmp_dict, file)