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
# @Time     : 2023/9/27 21:13
# @Author   : deviludier @ LINKE
# @File     : QCAE_state.py
# @Software : PyCharm

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import *
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit

def get_ME_state(n_qubits):
    qc = Circuit()
    for i in range(n_qubits):
        qc += H.on(i)
    for i in range(n_qubits):
        qc += CNOT.on(i+n_qubits, i)
    sim = Simulator('mqmatrix', 2 * n_qubits)
    sim.apply_circuit(qc)
    ME_state = sim.get_qs()
    return ME_state

def get_choi_state(circuit:Circuit):
    ME_state = get_ME_state(circuit.n_qubits)

    n_qubits = circuit.n_qubits
    for i in range(n_qubits):
        circuit += I.on(i+n_qubits)
    # print(circuit)
    sim2 = Simulator('mqmatrix', circuit.n_qubits)
    sim2.set_qs(ME_state)
    sim2.apply_circuit(circuit)
    choi_state = sim2.get_qs()
    return choi_state

def get_partial_trace(state, n_qubits, obj_qubits):
    sim = Simulator('mqmatrix', n_qubits)
    sim.set_qs(state)
    sim.get_partial_trace(obj_qubits=obj_qubits)


if __name__ == "__main__":
    num_qubits = 2
    target_op = random_circuit(num_qubits, 10)
    print(get_choi_state(target_op))
    pass
