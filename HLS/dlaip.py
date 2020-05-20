# Generate from nngen.j2

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math
import argparse
#import yaml
from collections import OrderedDict

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

import forceDSP

def get_outputs(operators):
    outputs = []
    for operator in operators.values():
        if operator.is_output is True:
            outputs.append(operator)
    return outputs


def post_process(post_param, ipname='dlaip'):
    _rtl_name = ipname + '_v1_0/hdl/' + ipname + '.v'
    if os.path.isfile(_rtl_name) is not True:
        print('{} does not exist.'.format(_rtl_name))

    x_force_dsp = post_param['X_FORCE_DSP'] if 'X_FORCE_DSP' in post_param else None
    if x_force_dsp is True:
        print('### DSP forced allocation ### ')
        forceDSP.forceDSP(_rtl_name)


user_config = {
    'maxi_datawidth': 64,
    'fsm_as_module': True,
    'offchipram_chunk_bytes': 4096,
    'header0': 192,
    'header1': 0,
    'header2': 532544,
    'header3': 264196,
}

IPNAME = 'dlaip'

operators = OrderedDict()
placeholders = OrderedDict()
variables = OrderedDict()

# graph

L002_Scale_name = 'L002_Scale'
L003_layer1_conv_cv_cbs_name = 'L003_layer1_conv_cv_cbs'
L003_layer1_conv_cv_cbs_name_filter = 'L003_layer1_conv_cv_cbs_convolution_weight'
L003_layer1_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[8, 3, 3, 3], name=L003_layer1_conv_cv_cbs_name_filter)
# - bias & scale -
L003_layer1_conv_cv_cbs_name_bias = 'L003_layer1_conv_cv_cbs_convolution_bias'
L003_layer1_conv_cv_cbs_name_scale = 'L003_layer1_conv_cv_cbs_i2f_i2f_scale'
L003_layer1_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[8], name=L003_layer1_conv_cv_cbs_name_bias)
L003_layer1_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L003_layer1_conv_cv_cbs_name_scale)
# - rshift_out -
L003_layer1_conv_cv_cbs_name_rshift_out = 'L003_layer1_conv_cv_cbs_i2f_rshift_out'
L003_layer1_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[8], name=L003_layer1_conv_cv_cbs_name_rshift_out)
variables[L003_layer1_conv_cv_cbs_name_rshift_out] = L003_layer1_conv_cv_cbs_rshift_out
# - act_func -
L003_layer1_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L003_layer1_conv_cv_cbs_name_filter] = L003_layer1_conv_cv_cbs_filter
variables[L003_layer1_conv_cv_cbs_name_bias] = L003_layer1_conv_cv_cbs_bias
variables[L003_layer1_conv_cv_cbs_name_scale] = L003_layer1_conv_cv_cbs_scale
L007_layer2_maxpool_name = 'L007_layer2_maxpool'
L008_layer3_conv_cv_cbs_name = 'L008_layer3_conv_cv_cbs'
L008_layer3_conv_cv_cbs_name_filter = 'L008_layer3_conv_cv_cbs_convolution_weight'
L008_layer3_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[16, 3, 3, 8], name=L008_layer3_conv_cv_cbs_name_filter)
# - bias & scale -
L008_layer3_conv_cv_cbs_name_bias = 'L008_layer3_conv_cv_cbs_convolution_bias'
L008_layer3_conv_cv_cbs_name_scale = 'L008_layer3_conv_cv_cbs_i2f_i2f_scale'
L008_layer3_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[16], name=L008_layer3_conv_cv_cbs_name_bias)
L008_layer3_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L008_layer3_conv_cv_cbs_name_scale)
# - rshift_out -
L008_layer3_conv_cv_cbs_name_rshift_out = 'L008_layer3_conv_cv_cbs_i2f_rshift_out'
L008_layer3_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[16], name=L008_layer3_conv_cv_cbs_name_rshift_out)
variables[L008_layer3_conv_cv_cbs_name_rshift_out] = L008_layer3_conv_cv_cbs_rshift_out
# - act_func -
L008_layer3_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L008_layer3_conv_cv_cbs_name_filter] = L008_layer3_conv_cv_cbs_filter
variables[L008_layer3_conv_cv_cbs_name_bias] = L008_layer3_conv_cv_cbs_bias
variables[L008_layer3_conv_cv_cbs_name_scale] = L008_layer3_conv_cv_cbs_scale
L012_layer4_maxpool_name = 'L012_layer4_maxpool'
L013_layer5_conv_cv_cbs_name = 'L013_layer5_conv_cv_cbs'
L013_layer5_conv_cv_cbs_name_filter = 'L013_layer5_conv_cv_cbs_convolution_weight'
L013_layer5_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[32, 3, 3, 16], name=L013_layer5_conv_cv_cbs_name_filter)
# - bias & scale -
L013_layer5_conv_cv_cbs_name_bias = 'L013_layer5_conv_cv_cbs_convolution_bias'
L013_layer5_conv_cv_cbs_name_scale = 'L013_layer5_conv_cv_cbs_i2f_i2f_scale'
L013_layer5_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[32], name=L013_layer5_conv_cv_cbs_name_bias)
L013_layer5_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L013_layer5_conv_cv_cbs_name_scale)
# - rshift_out -
L013_layer5_conv_cv_cbs_name_rshift_out = 'L013_layer5_conv_cv_cbs_i2f_rshift_out'
L013_layer5_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[32], name=L013_layer5_conv_cv_cbs_name_rshift_out)
variables[L013_layer5_conv_cv_cbs_name_rshift_out] = L013_layer5_conv_cv_cbs_rshift_out
# - act_func -
L013_layer5_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L013_layer5_conv_cv_cbs_name_filter] = L013_layer5_conv_cv_cbs_filter
variables[L013_layer5_conv_cv_cbs_name_bias] = L013_layer5_conv_cv_cbs_bias
variables[L013_layer5_conv_cv_cbs_name_scale] = L013_layer5_conv_cv_cbs_scale
L017_layer6_maxpool_name = 'L017_layer6_maxpool'
L018_layer7_conv_cv_cbs_name = 'L018_layer7_conv_cv_cbs'
L018_layer7_conv_cv_cbs_name_filter = 'L018_layer7_conv_cv_cbs_convolution_weight'
L018_layer7_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[64, 3, 3, 32], name=L018_layer7_conv_cv_cbs_name_filter)
# - bias & scale -
L018_layer7_conv_cv_cbs_name_bias = 'L018_layer7_conv_cv_cbs_convolution_bias'
L018_layer7_conv_cv_cbs_name_scale = 'L018_layer7_conv_cv_cbs_i2f_i2f_scale'
L018_layer7_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[64], name=L018_layer7_conv_cv_cbs_name_bias)
L018_layer7_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L018_layer7_conv_cv_cbs_name_scale)
# - rshift_out -
L018_layer7_conv_cv_cbs_name_rshift_out = 'L018_layer7_conv_cv_cbs_i2f_rshift_out'
L018_layer7_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[64], name=L018_layer7_conv_cv_cbs_name_rshift_out)
variables[L018_layer7_conv_cv_cbs_name_rshift_out] = L018_layer7_conv_cv_cbs_rshift_out
# - act_func -
L018_layer7_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L018_layer7_conv_cv_cbs_name_filter] = L018_layer7_conv_cv_cbs_filter
variables[L018_layer7_conv_cv_cbs_name_bias] = L018_layer7_conv_cv_cbs_bias
variables[L018_layer7_conv_cv_cbs_name_scale] = L018_layer7_conv_cv_cbs_scale
L022_layer8_maxpool_name = 'L022_layer8_maxpool'
L023_layer9_conv_cv_cbs_name = 'L023_layer9_conv_cv_cbs'
L023_layer9_conv_cv_cbs_name_filter = 'L023_layer9_conv_cv_cbs_convolution_weight'
L023_layer9_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[128, 3, 3, 64], name=L023_layer9_conv_cv_cbs_name_filter)
# - bias & scale -
L023_layer9_conv_cv_cbs_name_bias = 'L023_layer9_conv_cv_cbs_convolution_bias'
L023_layer9_conv_cv_cbs_name_scale = 'L023_layer9_conv_cv_cbs_i2f_i2f_scale'
L023_layer9_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[128], name=L023_layer9_conv_cv_cbs_name_bias)
L023_layer9_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L023_layer9_conv_cv_cbs_name_scale)
# - rshift_out -
L023_layer9_conv_cv_cbs_name_rshift_out = 'L023_layer9_conv_cv_cbs_i2f_rshift_out'
L023_layer9_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[128], name=L023_layer9_conv_cv_cbs_name_rshift_out)
variables[L023_layer9_conv_cv_cbs_name_rshift_out] = L023_layer9_conv_cv_cbs_rshift_out
# - act_func -
L023_layer9_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L023_layer9_conv_cv_cbs_name_filter] = L023_layer9_conv_cv_cbs_filter
variables[L023_layer9_conv_cv_cbs_name_bias] = L023_layer9_conv_cv_cbs_bias
variables[L023_layer9_conv_cv_cbs_name_scale] = L023_layer9_conv_cv_cbs_scale
L027_layer10_maxpool_name = 'L027_layer10_maxpool'
L028_layer11_conv_cv_cbs_name = 'L028_layer11_conv_cv_cbs'
L028_layer11_conv_cv_cbs_name_filter = 'L028_layer11_conv_cv_cbs_convolution_weight'
L028_layer11_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[248, 3, 3, 128], name=L028_layer11_conv_cv_cbs_name_filter)
# - bias & scale -
L028_layer11_conv_cv_cbs_name_bias = 'L028_layer11_conv_cv_cbs_convolution_bias'
L028_layer11_conv_cv_cbs_name_scale = 'L028_layer11_conv_cv_cbs_i2f_i2f_scale'
L028_layer11_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[248], name=L028_layer11_conv_cv_cbs_name_bias)
L028_layer11_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L028_layer11_conv_cv_cbs_name_scale)
# - rshift_out -
L028_layer11_conv_cv_cbs_name_rshift_out = 'L028_layer11_conv_cv_cbs_i2f_rshift_out'
L028_layer11_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[248], name=L028_layer11_conv_cv_cbs_name_rshift_out)
variables[L028_layer11_conv_cv_cbs_name_rshift_out] = L028_layer11_conv_cv_cbs_rshift_out
# - act_func -
L028_layer11_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L028_layer11_conv_cv_cbs_name_filter] = L028_layer11_conv_cv_cbs_filter
variables[L028_layer11_conv_cv_cbs_name_bias] = L028_layer11_conv_cv_cbs_bias
variables[L028_layer11_conv_cv_cbs_name_scale] = L028_layer11_conv_cv_cbs_scale
L032_layer13_conv_cv_cbs_name = 'L032_layer13_conv_cv_cbs'
L032_layer13_conv_cv_cbs_name_filter = 'L032_layer13_conv_cv_cbs_convolution_weight'
L032_layer13_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[508, 3, 3, 248], name=L032_layer13_conv_cv_cbs_name_filter)
# - bias & scale -
L032_layer13_conv_cv_cbs_name_bias = 'L032_layer13_conv_cv_cbs_convolution_bias'
L032_layer13_conv_cv_cbs_name_scale = 'L032_layer13_conv_cv_cbs_i2f_i2f_scale'
L032_layer13_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[508], name=L032_layer13_conv_cv_cbs_name_bias)
L032_layer13_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L032_layer13_conv_cv_cbs_name_scale)
# - rshift_out -
L032_layer13_conv_cv_cbs_name_rshift_out = 'L032_layer13_conv_cv_cbs_i2f_rshift_out'
L032_layer13_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[508], name=L032_layer13_conv_cv_cbs_name_rshift_out)
variables[L032_layer13_conv_cv_cbs_name_rshift_out] = L032_layer13_conv_cv_cbs_rshift_out
# - act_func -
L032_layer13_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L032_layer13_conv_cv_cbs_name_filter] = L032_layer13_conv_cv_cbs_filter
variables[L032_layer13_conv_cv_cbs_name_bias] = L032_layer13_conv_cv_cbs_bias
variables[L032_layer13_conv_cv_cbs_name_scale] = L032_layer13_conv_cv_cbs_scale
L036_layer14_conv_cv_cbs_name = 'L036_layer14_conv_cv_cbs'
L036_layer14_conv_cv_cbs_name_filter = 'L036_layer14_conv_cv_cbs_convolution_weight'
L036_layer14_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[128, 1, 1, 508], name=L036_layer14_conv_cv_cbs_name_filter)
# - bias & scale -
L036_layer14_conv_cv_cbs_name_bias = 'L036_layer14_conv_cv_cbs_convolution_bias'
L036_layer14_conv_cv_cbs_name_scale = 'L036_layer14_conv_cv_cbs_i2f_i2f_scale'
L036_layer14_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[128], name=L036_layer14_conv_cv_cbs_name_bias)
L036_layer14_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L036_layer14_conv_cv_cbs_name_scale)
# - rshift_out -
L036_layer14_conv_cv_cbs_name_rshift_out = 'L036_layer14_conv_cv_cbs_i2f_rshift_out'
L036_layer14_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[128], name=L036_layer14_conv_cv_cbs_name_rshift_out)
variables[L036_layer14_conv_cv_cbs_name_rshift_out] = L036_layer14_conv_cv_cbs_rshift_out
# - act_func -
L036_layer14_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L036_layer14_conv_cv_cbs_name_filter] = L036_layer14_conv_cv_cbs_filter
variables[L036_layer14_conv_cv_cbs_name_bias] = L036_layer14_conv_cv_cbs_bias
variables[L036_layer14_conv_cv_cbs_name_scale] = L036_layer14_conv_cv_cbs_scale
L040_layer15_conv_cv_cbs_name = 'L040_layer15_conv_cv_cbs'
L040_layer15_conv_cv_cbs_name_filter = 'L040_layer15_conv_cv_cbs_convolution_weight'
L040_layer15_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[140, 3, 3, 128], name=L040_layer15_conv_cv_cbs_name_filter)
# - bias & scale -
L040_layer15_conv_cv_cbs_name_bias = 'L040_layer15_conv_cv_cbs_convolution_bias'
L040_layer15_conv_cv_cbs_name_scale = 'L040_layer15_conv_cv_cbs_i2f_i2f_scale'
L040_layer15_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[140], name=L040_layer15_conv_cv_cbs_name_bias)
L040_layer15_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L040_layer15_conv_cv_cbs_name_scale)
# - rshift_out -
L040_layer15_conv_cv_cbs_name_rshift_out = 'L040_layer15_conv_cv_cbs_i2f_rshift_out'
L040_layer15_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[140], name=L040_layer15_conv_cv_cbs_name_rshift_out)
variables[L040_layer15_conv_cv_cbs_name_rshift_out] = L040_layer15_conv_cv_cbs_rshift_out
# - act_func -
L040_layer15_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L040_layer15_conv_cv_cbs_name_filter] = L040_layer15_conv_cv_cbs_filter
variables[L040_layer15_conv_cv_cbs_name_bias] = L040_layer15_conv_cv_cbs_bias
variables[L040_layer15_conv_cv_cbs_name_scale] = L040_layer15_conv_cv_cbs_scale
L044_c_layer16_conv_cv_name = 'L044_c_layer16_conv_cv'
L044_c_layer16_conv_cv_name_filter = 'L044_c_layer16_conv_cv_convolution_weight'
L044_c_layer16_conv_cv_filter = ng.variable(ng.dtype_int(width=8), shape=[33, 1, 1, 140], name=L044_c_layer16_conv_cv_name_filter)
# - bias & scale -
L044_c_layer16_conv_cv_name_bias = 'L044_c_layer16_conv_cv_convolution_bias'
L044_c_layer16_conv_cv_name_scale = 'L044_c_layer16_conv_cv_i2f_i2f_scale'
L044_c_layer16_conv_cv_bias = ng.variable(dtype=ng.int32, shape=[33], name=L044_c_layer16_conv_cv_name_bias)
L044_c_layer16_conv_cv_scale = ng.variable(dtype=ng.int16, shape=[1], name=L044_c_layer16_conv_cv_name_scale)
# - rshift_out -
L044_c_layer16_conv_cv_name_rshift_out = 'L044_c_layer16_conv_cv_i2f_rshift_out'
L044_c_layer16_conv_cv_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[33], name=L044_c_layer16_conv_cv_name_rshift_out)
variables[L044_c_layer16_conv_cv_name_rshift_out] = L044_c_layer16_conv_cv_rshift_out
# - act_func -
L044_c_layer16_conv_cv_act_func = None
variables[L044_c_layer16_conv_cv_name_filter] = L044_c_layer16_conv_cv_filter
variables[L044_c_layer16_conv_cv_name_bias] = L044_c_layer16_conv_cv_bias
variables[L044_c_layer16_conv_cv_name_scale] = L044_c_layer16_conv_cv_scale
L045_layer19_conv_cv_cbs_name = 'L045_layer19_conv_cv_cbs'
L045_layer19_conv_cv_cbs_name_filter = 'L045_layer19_conv_cv_cbs_convolution_weight'
L045_layer19_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[64, 1, 1, 128], name=L045_layer19_conv_cv_cbs_name_filter)
# - bias & scale -
L045_layer19_conv_cv_cbs_name_bias = 'L045_layer19_conv_cv_cbs_convolution_bias'
L045_layer19_conv_cv_cbs_name_scale = 'L045_layer19_conv_cv_cbs_i2f_i2f_scale'
L045_layer19_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[64], name=L045_layer19_conv_cv_cbs_name_bias)
L045_layer19_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L045_layer19_conv_cv_cbs_name_scale)
# - rshift_out -
L045_layer19_conv_cv_cbs_name_rshift_out = 'L045_layer19_conv_cv_cbs_i2f_rshift_out'
L045_layer19_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[64], name=L045_layer19_conv_cv_cbs_name_rshift_out)
variables[L045_layer19_conv_cv_cbs_name_rshift_out] = L045_layer19_conv_cv_cbs_rshift_out
# - act_func -
L045_layer19_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L045_layer19_conv_cv_cbs_name_filter] = L045_layer19_conv_cv_cbs_filter
variables[L045_layer19_conv_cv_cbs_name_bias] = L045_layer19_conv_cv_cbs_bias
variables[L045_layer19_conv_cv_cbs_name_scale] = L045_layer19_conv_cv_cbs_scale
L049_layer20_upsample_name = 'L049_layer20_upsample'
L050_layer21_route_name = 'L050_layer21_route'
# - variable -
L050_layer21_route_name_kmconcat_coefficient0 = 'L050_layer21_route_kmconcat_coefficient0'
L050_layer21_route_name_kmconcat_coefficient1 = 'L050_layer21_route_kmconcat_coefficient1'
L050_layer21_route_name_kmconcat_shift = 'L050_layer21_route_kmconcat_shift'
L050_layer21_route_v_scale0 = ng.variable(ng.dtype_int(width=8), shape=(1,), name=L050_layer21_route_name_kmconcat_coefficient0)
L050_layer21_route_v_scale1 = ng.variable(ng.dtype_int(width=8), shape=(1,), name=L050_layer21_route_name_kmconcat_coefficient1)
L050_layer21_route_v_shift  = ng.variable(ng.dtype_int(width=8), shape=(1,), name=L050_layer21_route_name_kmconcat_shift)
# - op. -
L050_layer21_route_name_scale0 = 'L050_layer21_route_scale0'
L050_layer21_route_name_shift0 = 'L050_layer21_route_shift0'
L050_layer21_route_name_clip0 = 'L050_layer21_route_clip0'
L050_layer21_route_name_scale1 = 'L050_layer21_route_scale1'
L050_layer21_route_name_shift1 = 'L050_layer21_route_shift1'
L050_layer21_route_name_clip1 = 'L050_layer21_route_clip1'
variables[L050_layer21_route_name_kmconcat_coefficient0] = L050_layer21_route_v_scale0
variables[L050_layer21_route_name_kmconcat_coefficient1] = L050_layer21_route_v_scale1
variables[L050_layer21_route_name_kmconcat_shift] = L050_layer21_route_v_shift
L051_layer22_conv_cv_cbs_name = 'L051_layer22_conv_cv_cbs'
L051_layer22_conv_cv_cbs_name_filter = 'L051_layer22_conv_cv_cbs_convolution_weight'
L051_layer22_conv_cv_cbs_filter = ng.variable(ng.dtype_int(width=8), shape=[128, 3, 3, 192], name=L051_layer22_conv_cv_cbs_name_filter)
# - bias & scale -
L051_layer22_conv_cv_cbs_name_bias = 'L051_layer22_conv_cv_cbs_convolution_bias'
L051_layer22_conv_cv_cbs_name_scale = 'L051_layer22_conv_cv_cbs_i2f_i2f_scale'
L051_layer22_conv_cv_cbs_bias = ng.variable(dtype=ng.int32, shape=[128], name=L051_layer22_conv_cv_cbs_name_bias)
L051_layer22_conv_cv_cbs_scale = ng.variable(dtype=ng.int16, shape=[1], name=L051_layer22_conv_cv_cbs_name_scale)
# - rshift_out -
L051_layer22_conv_cv_cbs_name_rshift_out = 'L051_layer22_conv_cv_cbs_i2f_rshift_out'
L051_layer22_conv_cv_cbs_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[128], name=L051_layer22_conv_cv_cbs_name_rshift_out)
variables[L051_layer22_conv_cv_cbs_name_rshift_out] = L051_layer22_conv_cv_cbs_rshift_out
# - act_func -
L051_layer22_conv_cv_cbs_act_func = ng.get_leaky_relu_op(13, rshift=None, dtype=ng.dtype_int(width=8))
variables[L051_layer22_conv_cv_cbs_name_filter] = L051_layer22_conv_cv_cbs_filter
variables[L051_layer22_conv_cv_cbs_name_bias] = L051_layer22_conv_cv_cbs_bias
variables[L051_layer22_conv_cv_cbs_name_scale] = L051_layer22_conv_cv_cbs_scale
L055_c_layer23_conv_cv_name = 'L055_c_layer23_conv_cv'
L055_c_layer23_conv_cv_name_filter = 'L055_c_layer23_conv_cv_convolution_weight'
L055_c_layer23_conv_cv_filter = ng.variable(ng.dtype_int(width=8), shape=[33, 1, 1, 128], name=L055_c_layer23_conv_cv_name_filter)
# - bias & scale -
L055_c_layer23_conv_cv_name_bias = 'L055_c_layer23_conv_cv_convolution_bias'
L055_c_layer23_conv_cv_name_scale = 'L055_c_layer23_conv_cv_i2f_i2f_scale'
L055_c_layer23_conv_cv_bias = ng.variable(dtype=ng.int32, shape=[33], name=L055_c_layer23_conv_cv_name_bias)
L055_c_layer23_conv_cv_scale = ng.variable(dtype=ng.int16, shape=[1], name=L055_c_layer23_conv_cv_name_scale)
# - rshift_out -
L055_c_layer23_conv_cv_name_rshift_out = 'L055_c_layer23_conv_cv_i2f_rshift_out'
L055_c_layer23_conv_cv_rshift_out = ng.variable(ng.dtype_int(width=8), shape=[33], name=L055_c_layer23_conv_cv_name_rshift_out)
variables[L055_c_layer23_conv_cv_name_rshift_out] = L055_c_layer23_conv_cv_rshift_out
# - act_func -
L055_c_layer23_conv_cv_act_func = None
variables[L055_c_layer23_conv_cv_name_filter] = L055_c_layer23_conv_cv_filter
variables[L055_c_layer23_conv_cv_name_bias] = L055_c_layer23_conv_cv_bias
variables[L055_c_layer23_conv_cv_name_scale] = L055_c_layer23_conv_cv_scale

# instance 
# ここから：DNNモデルの宣言

L002_Scale = ng.placeholder(ng.dtype_int(width=8), shape=[1, 288, 512, 3], name=L002_Scale_name)
placeholders[L002_Scale_name] = L002_Scale

L003_layer1_conv_cv_cbs = ng.conv2d(L002_Scale, L003_layer1_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L003_layer1_conv_cv_cbs_bias, scale=L003_layer1_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L003_layer1_conv_cv_cbs_rshift_out, act_func=L003_layer1_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L003_layer1_conv_cv_cbs_name)
operators[L003_layer1_conv_cv_cbs_name] = L003_layer1_conv_cv_cbs

L007_layer2_maxpool = ng.max_pool(L003_layer1_conv_cv_cbs, [1, 2, 2, 1], [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L007_layer2_maxpool_name)
operators[L007_layer2_maxpool_name] = L007_layer2_maxpool

L008_layer3_conv_cv_cbs = ng.conv2d(L007_layer2_maxpool, L008_layer3_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L008_layer3_conv_cv_cbs_bias, scale=L008_layer3_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L008_layer3_conv_cv_cbs_rshift_out, act_func=L008_layer3_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L008_layer3_conv_cv_cbs_name)
operators[L008_layer3_conv_cv_cbs_name] = L008_layer3_conv_cv_cbs

L012_layer4_maxpool = ng.max_pool(L008_layer3_conv_cv_cbs, [1, 2, 2, 1], [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L012_layer4_maxpool_name)
operators[L012_layer4_maxpool_name] = L012_layer4_maxpool

L013_layer5_conv_cv_cbs = ng.conv2d(L012_layer4_maxpool, L013_layer5_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L013_layer5_conv_cv_cbs_bias, scale=L013_layer5_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L013_layer5_conv_cv_cbs_rshift_out, act_func=L013_layer5_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L013_layer5_conv_cv_cbs_name)
operators[L013_layer5_conv_cv_cbs_name] = L013_layer5_conv_cv_cbs

L017_layer6_maxpool = ng.max_pool(L013_layer5_conv_cv_cbs, [1, 2, 2, 1], [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L017_layer6_maxpool_name)
operators[L017_layer6_maxpool_name] = L017_layer6_maxpool

L018_layer7_conv_cv_cbs = ng.conv2d(L017_layer6_maxpool, L018_layer7_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L018_layer7_conv_cv_cbs_bias, scale=L018_layer7_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L018_layer7_conv_cv_cbs_rshift_out, act_func=L018_layer7_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L018_layer7_conv_cv_cbs_name)
operators[L018_layer7_conv_cv_cbs_name] = L018_layer7_conv_cv_cbs

L022_layer8_maxpool = ng.max_pool(L018_layer7_conv_cv_cbs, [1, 2, 2, 1], [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L022_layer8_maxpool_name)
operators[L022_layer8_maxpool_name] = L022_layer8_maxpool

L023_layer9_conv_cv_cbs = ng.conv2d(L022_layer8_maxpool, L023_layer9_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L023_layer9_conv_cv_cbs_bias, scale=L023_layer9_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L023_layer9_conv_cv_cbs_rshift_out, act_func=L023_layer9_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L023_layer9_conv_cv_cbs_name)
operators[L023_layer9_conv_cv_cbs_name] = L023_layer9_conv_cv_cbs

L027_layer10_maxpool = ng.max_pool(L023_layer9_conv_cv_cbs, [1, 2, 2, 1], [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L027_layer10_maxpool_name)
operators[L027_layer10_maxpool_name] = L027_layer10_maxpool

L028_layer11_conv_cv_cbs = ng.conv2d(L027_layer10_maxpool, L028_layer11_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L028_layer11_conv_cv_cbs_bias, scale=L028_layer11_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L028_layer11_conv_cv_cbs_rshift_out, act_func=L028_layer11_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L028_layer11_conv_cv_cbs_name)
operators[L028_layer11_conv_cv_cbs_name] = L028_layer11_conv_cv_cbs

L032_layer13_conv_cv_cbs = ng.conv2d(L028_layer11_conv_cv_cbs, L032_layer13_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L032_layer13_conv_cv_cbs_bias, scale=L032_layer13_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L032_layer13_conv_cv_cbs_rshift_out, act_func=L032_layer13_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L032_layer13_conv_cv_cbs_name)
operators[L032_layer13_conv_cv_cbs_name] = L032_layer13_conv_cv_cbs

L036_layer14_conv_cv_cbs = ng.conv2d(L032_layer13_conv_cv_cbs, L036_layer14_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L036_layer14_conv_cv_cbs_bias, scale=L036_layer14_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L036_layer14_conv_cv_cbs_rshift_out, act_func=L036_layer14_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L036_layer14_conv_cv_cbs_name)
operators[L036_layer14_conv_cv_cbs_name] = L036_layer14_conv_cv_cbs

L040_layer15_conv_cv_cbs = ng.conv2d(L036_layer14_conv_cv_cbs, L040_layer15_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L040_layer15_conv_cv_cbs_bias, scale=L040_layer15_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L040_layer15_conv_cv_cbs_rshift_out, act_func=L040_layer15_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L040_layer15_conv_cv_cbs_name)
operators[L040_layer15_conv_cv_cbs_name] = L040_layer15_conv_cv_cbs

L044_c_layer16_conv_cv = ng.conv2d(L040_layer15_conv_cv_cbs, L044_c_layer16_conv_cv_filter, [1, 1, 1, 1], bias=L044_c_layer16_conv_cv_bias, scale=L044_c_layer16_conv_cv_scale, rshift_mul=0, rshift_sum=0, rshift_out=L044_c_layer16_conv_cv_rshift_out, act_func=L044_c_layer16_conv_cv_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L044_c_layer16_conv_cv_name)
operators[L044_c_layer16_conv_cv_name] = L044_c_layer16_conv_cv
L044_c_layer16_conv_cv.set_output()

L045_layer19_conv_cv_cbs = ng.conv2d(L036_layer14_conv_cv_cbs, L045_layer19_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L045_layer19_conv_cv_cbs_bias, scale=L045_layer19_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L045_layer19_conv_cv_cbs_rshift_out, act_func=L045_layer19_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L045_layer19_conv_cv_cbs_name)
operators[L045_layer19_conv_cv_cbs_name] = L045_layer19_conv_cv_cbs

L049_layer20_upsample = ng.upsampling2d(L045_layer19_conv_cv_cbs, [1, 2, 2, 1], dtype=ng.dtype_int(width=8), name=L049_layer20_upsample_name)
operators[L049_layer20_upsample_name] = L049_layer20_upsample

L050_layer21_route_values = [L049_layer20_upsample,L023_layer9_conv_cv_cbs,]
L050_layer21_route_op_sc0 = ng.multiply(L050_layer21_route_values[0], L050_layer21_route_v_scale0, dtype=ng.dtype_int(width=16), name=L050_layer21_route_name_scale0)
L050_layer21_route_op_sft0= ng.rshift_round(L050_layer21_route_op_sc0, L050_layer21_route_v_shift, dtype=ng.dtype_int(width=16), name=L050_layer21_route_name_shift0)
L050_layer21_route_op_clp0= ng.clip(L050_layer21_route_op_sft0, dtype=ng.dtype_int(width=8), name=L050_layer21_route_name_clip0)
L050_layer21_route_op_sc1 = ng.multiply(L050_layer21_route_values[1], L050_layer21_route_v_scale1, dtype=ng.dtype_int(width=16), name=L050_layer21_route_name_scale1)
L050_layer21_route_op_sft1= ng.rshift_round(L050_layer21_route_op_sc1, L050_layer21_route_v_shift, dtype=ng.dtype_int(width=16), name=L050_layer21_route_name_shift1)
L050_layer21_route_op_clp1= ng.clip(L050_layer21_route_op_sft1, dtype=ng.dtype_int(width=8), name=L050_layer21_route_name_clip1)
L050_layer21_route = ng.concat([L050_layer21_route_op_clp0, L050_layer21_route_op_clp1], axis=3, dtype=ng.dtype_int(width=8), name=L050_layer21_route_name)
operators[L050_layer21_route_name_scale0] = L050_layer21_route_op_sc0
operators[L050_layer21_route_name_shift0] = L050_layer21_route_op_sft0
operators[L050_layer21_route_name_clip0] = L050_layer21_route_op_clp0
operators[L050_layer21_route_name_scale1] = L050_layer21_route_op_sc1
operators[L050_layer21_route_name_shift1] = L050_layer21_route_op_sft1
operators[L050_layer21_route_name_clip1] = L050_layer21_route_op_clp1
operators[L050_layer21_route_name] = L050_layer21_route

L051_layer22_conv_cv_cbs = ng.conv2d(L050_layer21_route, L051_layer22_conv_cv_cbs_filter, [1, 1, 1, 1], bias=L051_layer22_conv_cv_cbs_bias, scale=L051_layer22_conv_cv_cbs_scale, rshift_mul=0, rshift_sum=0, rshift_out=L051_layer22_conv_cv_cbs_rshift_out, act_func=L051_layer22_conv_cv_cbs_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L051_layer22_conv_cv_cbs_name)
operators[L051_layer22_conv_cv_cbs_name] = L051_layer22_conv_cv_cbs

L055_c_layer23_conv_cv = ng.conv2d(L051_layer22_conv_cv_cbs, L055_c_layer23_conv_cv_filter, [1, 1, 1, 1], bias=L055_c_layer23_conv_cv_bias, scale=L055_c_layer23_conv_cv_scale, rshift_mul=0, rshift_sum=0, rshift_out=L055_c_layer23_conv_cv_rshift_out, act_func=L055_c_layer23_conv_cv_act_func, dtype=ng.dtype_int(width=8), mul_dtype=ng.dtype_int(width=16), sum_dtype=ng.dtype_int(width=32), name=L055_c_layer23_conv_cv_name)
operators[L055_c_layer23_conv_cv_name] = L055_c_layer23_conv_cv
L055_c_layer23_conv_cv.set_output()

# ここまで：DNNモデル宣言

# attribute(par)
# --- Generate from conv2d_attribute.j2 ---
L003_layer1_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from max_pool_attribute.j2 ---
L007_layer2_maxpool.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L008_layer3_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from max_pool_attribute.j2 ---
L012_layer4_maxpool.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L013_layer5_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from max_pool_attribute.j2 ---
L017_layer6_maxpool.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L018_layer7_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from max_pool_attribute.j2 ---
L022_layer8_maxpool.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L023_layer9_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from max_pool_attribute.j2 ---
L027_layer10_maxpool.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L028_layer11_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L032_layer13_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L036_layer14_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=2, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L040_layer15_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L044_c_layer16_conv_cv.attribute(par_ich=8, par_och=4, par_col=2, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L045_layer19_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=2, par_row=None)
# --- Generate from upsampling2d_attribute.j2 ---
L049_layer20_upsample.attribute(par=4)
# --- Generate from concat_attribute.j2 ---
L050_layer21_route_op_sc0.attribute(par=4)
L050_layer21_route_op_sft0.attribute(par=4)
L050_layer21_route_op_clp0.attribute(par=4)
L050_layer21_route_op_sc1.attribute(par=4)
L050_layer21_route_op_sft1.attribute(par=4)
L050_layer21_route_op_clp1.attribute(par=4)
# --- Generate from conv2d_attribute.j2 ---
L051_layer22_conv_cv_cbs.attribute(par_ich=8, par_och=4, par_col=None, par_row=None)
# --- Generate from conv2d_attribute.j2 ---
L055_c_layer23_conv_cv.attribute(par_ich=8, par_och=4, par_col=2, par_row=None)

# attribute(ram)
# --- Generate from conv2d_attribute_ram.j2 ---
L003_layer1_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L008_layer3_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L013_layer5_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L018_layer7_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L023_layer9_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L028_layer11_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L032_layer13_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L036_layer14_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L040_layer15_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L044_c_layer16_conv_cv.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L045_layer19_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L051_layer22_conv_cv_cbs.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)
# --- Generate from conv2d_attribute_ram.j2 ---
L055_c_layer23_conv_cv.attribute(input_ram_size=1024, filter_ram_size=1024, bias_ram_size=1024, scale_ram_size=1024, out_ram_size=4096)

_outputs = get_outputs(operators)

m = ng.to_ipxact(_outputs, IPNAME, config=user_config)

post_process({'X_FORCE_DSP': True}, IPNAME)

#print(placeholders)
#print(variables)
#print(operators)
