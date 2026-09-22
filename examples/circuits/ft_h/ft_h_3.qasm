OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
creg c[35];

// reset q[10];
// reset q[11];
// reset q[12];
// reset q[13];

cx q[2], q[10];
cx q[3], q[10];
measure q[10] -> c[6];
// reset q[10];
// h q[10];

cx q[1], q[11];
cx q[2], q[11];
cx q[4], q[11];
cx q[5], q[11];
measure q[11] -> c[5];
// reset q[11];
// h q[11];

cx q[5], q[12];
cx q[6], q[12];
cx q[8], q[12];
cx q[9], q[12];
measure q[12] -> c[7];
// reset q[12];
// h q[12];

cx q[7], q[13];
cx q[8], q[13];
measure q[13] -> c[8];
// reset q[13];
// h q[13];
