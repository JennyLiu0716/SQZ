OPENQASM 2.0;
include "qelib1.inc";

qreg q[19];
creg c[35];

// reset q[14];
// reset q[15];
// reset q[16];
// reset q[17];

// h q[14];
// h q[15];
// h q[16];
// h q[17];

cx q[14], q[1];
cx q[14], q[4];
h q[14];
measure q[14] -> c[9]; 
// reset q[14];

cx q[15], q[2];
cx q[15], q[3];
cx q[15], q[5];
cx q[15], q[6];
h q[15];
measure q[15] -> c[10];
// reset q[15];

cx q[16], q[4];
cx q[16], q[5];
cx q[16], q[7];
cx q[16], q[8];
h q[16];
measure q[16] -> c[11];
// reset q[16];

cx q[17], q[6];
cx q[17], q[9];
h q[17];
measure q[17] -> c[12];
// reset q[17];