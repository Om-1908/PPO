OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
y q[1];
rx(5*pi/8) q[0];
rx(-3*pi/8) q[1];
cx q[1],q[0];