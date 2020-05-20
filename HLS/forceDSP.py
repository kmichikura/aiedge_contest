#!env python
###################################################################
# DSP allocation script for xilinx.
# By default vivado's int8 multipliers are generated with LUT logic.
# This script inserts a attribute into the multiplier module to make 
# the DSP even at int8.
###################################################################

import sys
import fileinput
import re

def forceDSP(file_name):

    print("python3 forceDSP.py %s" % (file_name))

    with fileinput.input(file_name, inplace=True, backup=".bdsp") as f:
        for line in f:
            line = line.strip("\n")
            if re.search(r"module madd_core_",line):
                print("(* use_dsp = \"yes\" *)")
                print(line)
            elif re.search(r"module multiplier_core_",line):
                print("(* use_dsp = \"yes\" *)")
                print(line)
            else:
                print(line)


if __name__ == '__main__':
    args = sys.argv
    file_name = args[1]
    forceDSP(file_name)
