OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
creg c[35];



// reset q[10];
// reset q[11];
// reset q[12];
// reset q[13];


cx q[0], q[1];
cx q[0], q[4];
cx q[0], q[7];
h q[0];
measure q[0] -> c[0];
// reset q[0];

cx q[11], q[1];
cx q[11], q[2];
cx q[11], q[4];
cx q[11], q[5];
h q[11];
measure q[11] -> c[1];

cx q[10], q[2];
cx q[10], q[3];
h q[10];
measure q[10] -> c[2];

cx q[13], q[7];
cx q[13], q[8];
h q[13];
measure q[13] -> c[3];

cx q[12], q[5];
cx q[12], q[6];
cx q[12], q[8];
cx q[12], q[9];
h q[12];
measure q[12] -> c[4];

