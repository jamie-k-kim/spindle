# Takes a logical circuit and computes (1) the longest interaction distance
# and (2) the two-qubit density.
def compute_interaction_metrics(circuit):
    two_qubit_gates = 0
    longest_distance = 0

    for instruction in circuit.data:

        qubits = instruction.qubits

        if len(qubits) == 2:

            two_qubit_gates += 1

            q0 = circuit.find_bit(qubits[0]).index
            q1 = circuit.find_bit(qubits[1]).index

            distance = abs(q0 - q1)

            if distance > longest_distance:
                longest_distance = distance

    depth = circuit.depth()

    two_qubit_density = (
        two_qubit_gates / depth
        if depth > 0
        else 0
    )

    return longest_distance, two_qubit_density