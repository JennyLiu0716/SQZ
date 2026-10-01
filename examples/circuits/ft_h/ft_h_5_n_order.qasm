OPENQASM 2.0;
include "qelib1.inc";
// As ft_h_5.qasm, with the CNOTs of the weight-4 checks (ancillas q[15], q[16]) in N order:
// top-left, bottom-left, top-right, bottom-right data qubit. A fault on the ancilla half way
// through then spreads onto a vertical pair of data qubits instead of a horizontal one.

qreg q[19];
creg c[35];

// reset q[14];
// reset q[15];
// reset q[16];
// reset q[17];

cx q[1], q[14];
cx q[4], q[14];
measure q[14] -> c[13];
// reset q[14];

cx q[2], q[15];
cx q[5], q[15];
cx q[3], q[15];
cx q[6], q[15];
measure q[15] -> c[14];
// reset q[15];

cx q[4], q[16];
cx q[7], q[16];
cx q[5], q[16];
cx q[8], q[16];
measure q[16] -> c[15];
// reset q[16];

cx q[6], q[17];
cx q[9], q[17];
measure q[17] -> c[16];
// reset q[17];