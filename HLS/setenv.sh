#!/bin/bash

PYTHONPATH="\
./Pyverilog:\
./veriloggen:\
./nngen:\
$PYTHONPATH\
"
export PYTHONPATH
echo 'PYTHONPATH='$PYTHONPATH

