#!/bin/sh
cd ~/QuantumChemistry/spindle/spiral-software
echo 'Load(quantum); Import(quantum);' > temp_test.g
echo 'Read("temp_test.g");' | bin/spiral
