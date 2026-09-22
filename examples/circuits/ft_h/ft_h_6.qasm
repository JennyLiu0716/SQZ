OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
creg c[35];

// reset q[14];
// reset q[15];
// reset q[16];
// reset q[17];
// reset q[18];

cx q[1], q[14];
cx q[4], q[14];
measure q[14] -> c[21];

cx q[2], q[15];
cx q[3], q[15];
cx q[5], q[15];
cx q[6], q[15];
measure q[15] -> c[22];

cx q[4], q[16];
cx q[5], q[16];
cx q[7], q[16];
cx q[8], q[16];
measure q[16] -> c[23];

cx q[6], q[17];
cx q[9], q[17];
measure q[17] -> c[24];



cx q[7], q[18];
cx q[8], q[18];
cx q[9], q[18];

h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];

measure q[1] -> c[26];
measure q[2] -> c[27];
measure q[3] -> c[28];
measure q[4] -> c[29];
measure q[5] -> c[30];
measure q[6] -> c[31];
measure q[7] -> c[32];
measure q[8] -> c[33];
measure q[9] -> c[34];