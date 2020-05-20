from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.storage as storage
import nngen.dtype_list as dtype_list

from . import util
from . import basic
from . import conv
from . import gemm
from . import pool
from . import relu
from . import batchnormalization
from . import squeeze
from . import pad
from . import shape
from . import reshape
from . import concat
from . import gather
from . import flatten


# describe custom ONNX converting methods here
func_map = {
    'Add': basic.Add,
    'Sub': basic.Sub,
    'Mul': basic.Mul,
    'Div': basic.Div,
    'Conv': conv.Conv,
    'Gemm': gemm.Gemm,
    'AveragePool': pool.AveragePool,
    'GlobalAveragePool': pool.GlobalAveragePool,
    'MaxPool': pool.MaxPool,
    'Relu': relu.Relu,
    'LeakyRelu': relu.LeakyRelu,
    'BatchNormalization': batchnormalization.BatchNormalization,
    'Squeeze': squeeze.Squeeze,
    'Unsqueeze': squeeze.Unsqueeze,
    'Pad': pad.Pad,
    'Shape': shape.Shape,
    'Reshape': reshape.Reshape,
    'Concat': concat.Concat,
    'Gather': gather.Gather,
    'Flatten': flatten.Flatten,
}


def _get_func(op_type):
    return func_map[op_type]


class _OperatorVisitor(object):

    def __init__(self, model,
                 placeholders, variables, constants, operators,
                 producers, consumers,
                 value_dtypes,
                 default_placeholder_dtype, default_variable_dtype,
                 default_constant_dtype, default_operator_dtype,
                 default_scale_dtype, default_bias_dtype,
                 onnx_input_layout='NCHW', onnx_filter_layout='OIHW',
                 disable_fusion=False):

        self.model = model

        self.placeholders = placeholders
        self.variables = variables
        self.constants = constants
        self.operators = operators

        self.producers = producers
        self.consumers = consumers

        self.value_dtypes = value_dtypes

        self.default_placeholder_dtype = default_placeholder_dtype
        self.default_variable_dtype = default_variable_dtype
        self.default_constant_dtype = default_constant_dtype
        self.default_operator_dtype = default_operator_dtype
        self.default_scale_dtype = default_scale_dtype
        self.default_bias_dtype = default_bias_dtype

        self.onnx_input_layout = onnx_input_layout
        self.onnx_filter_layout = onnx_filter_layout

        self.disable_fusion = disable_fusion

    def visit(self, name):
        if name in self.placeholders:
            return self.placeholders[name]

        if name in self.variables:
            return self.variables[name]

        if name in self.constants:
            return self.constants[name]

        if name in self.operators:
            return self.operators[name]

        node = util.search_node_from_model(self.model, name)

        node_name = util.get_name(node)
        node_func = _get_func(node.op_type)

        node_op = node_func(self, node)

        self.operators[node_name] = node_op

        return node_op


def from_onnx(filename,
              value_dtypes=None,
              default_placeholder_dtype=dtype_list.int32,
              default_variable_dtype=dtype_list.int32,
              default_constant_dtype=dtype_list.int32,
              default_operator_dtype=dtype_list.int32,
              default_scale_dtype=dtype_list.int32,
              default_bias_dtype=dtype_list.int32,
              onnx_input_layout='NCHW',
              onnx_filter_layout='OIHW',
              disable_fusion=False):
    """
    Convert ONNX model to NNgen model

    Parameters
    ----------
    filename : str
        File name of ONNX model

    value_dtypes : dict
        dtype_info dictionary by name

    default_placeholder_dtype : nngen.dtype_info
        Default dtype for placeholder

    default_variable_dtype : nngen.dtype_info
        Default dtype for variable

    default_constant_dtype : nngen.dtype_info
        Default dtype for constant

    default_operator_dtype : nngen.dtype_info
        Default dtype for operator

    default_scale_dtype : nngen.dtype_info
        Default dtype for scale

    default_bias_dtype : nngen.dtype_info
        Default dtype for bias

    onnx_input_layout : str
        Layout of ONNX input values

    onnx_filter_layout : str
        Layout of ONNX filter (weight) values

    disable_fusion : bool
        Disable operator fusion

    Returns
    -------
    outputs : collections.OrderedDict
        Dict of output values

    placeholders : collections.OrderedDict
        Dictionary of placeholders

    variables : collections.OrderedDict
        Dictionary of variables

    constants : collections.OrderedDict
        Dictionary of constants

    operators : collections.OrderedDict
        Dictionary of operators
    """

    try:
        import onnx
        from onnx import numpy_helper
    except:
        raise ImportError('onnx is required.')

    if value_dtypes is None:
        value_dtypes = {}

    # load model
    model = onnx.load(filename)

    # input/output node dict
    input_nodes = collections.OrderedDict()
    output_nodes = collections.OrderedDict()

    for input_var in model.graph.input:
        input_nodes[input_var.name] = input_var

    for output_var in model.graph.output:
        output_nodes[output_var.name] = output_var

    # variable ndarray dict
    variable_values = collections.OrderedDict()

    for weight in model.graph.initializer:
        name = weight.name
        np_weight = numpy_helper.to_array(weight)
        variable_values[name] = np_weight

    # constant ndarray dict
    constant_values = collections.OrderedDict()

    for node in model.graph.node:
        if node.op_type == 'Constant':
            name = util.get_name(node)
            value = numpy_helper.to_array(node.attribute[0].t)
            constant_values[name] = value

    # placeholders
    placeholders = _to_placeholders(input_nodes, output_nodes,
                                    variable_values, constant_values,
                                    value_dtypes,
                                    default_placeholder_dtype,
                                    default_variable_dtype,
                                    default_constant_dtype,
                                    default_operator_dtype)

    # variables
    variables = _to_variables(input_nodes, output_nodes,
                              variable_values, constant_values,
                              value_dtypes,
                              default_placeholder_dtype,
                              default_variable_dtype,
                              default_constant_dtype,
                              default_operator_dtype)

    # constants
    # constants = _to_constants(input_nodes, output_nodes,
    #                          variable_values, constant_values,
    #                          value_dtypes,
    #                          default_placeholder_dtype,
    #                          default_variable_dtype,
    #                          default_constant_dtype,
    #                          default_operator_dtype)
    constants = constant_values

    # producer/consumer table
    producers = collections.defaultdict(list)
    consumers = collections.defaultdict(list)

    for node in model.graph.node:
        node_name = util.get_name(node)
        for arg in node.input:
            if arg not in producers[node_name]:
                producers[node_name].append(arg)
            if node_name not in consumers[arg]:
                consumers[arg].append(node_name)

    # operators
    operators = collections.OrderedDict()
    visitor = _OperatorVisitor(model,
                               placeholders, variables, constants, operators,
                               producers, consumers,
                               value_dtypes,
                               default_placeholder_dtype, default_variable_dtype,
                               default_constant_dtype, default_operator_dtype,
                               default_scale_dtype, default_bias_dtype,
                               onnx_input_layout, onnx_filter_layout,
                               disable_fusion)

    placeholders = visitor.placeholders
    variables = visitor.variables
    constants = visitor.constants
    operators = visitor.operators

    for name, output_node in output_nodes.items():
        visitor.visit(name)

    # outputs
    outputs = collections.OrderedDict()

    for name, node in output_nodes.items():
        if name in operators:
            outputs[name] = operators[name]
        elif name in placeholders:
            outputs[name] = placeholders[name]
        elif name in variables:
            outputs[name] = variables[name]
        elif name in constants:
            outputs[name] = constants[name]

    return outputs, placeholders, variables, constants, operators


def _to_placeholders(input_nodes, output_nodes, variable_values, constant_values,
                     value_dtypes,
                     default_placeholder_dtype, default_variable_dtype,
                     default_constant_dtype, default_operator_dtype):

    placeholders = collections.OrderedDict()

    for name, node in input_nodes.items():
        # exclude variables
        if name in variable_values:
            continue

        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_placeholder_dtype

        shape = util.to_shape(node)
        p = storage.placeholder(dtype=dtype, shape=shape, name=name)
        placeholders[name] = p

    return placeholders


def _to_variables(input_nodes, output_nodes, variable_values, constant_values,
                  value_dtypes,
                  default_placeholder_dtype, default_variable_dtype,
                  default_constant_dtype, default_operator_dtype):

    variables = collections.OrderedDict()

    for name, node in variable_values.items():
        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_variable_dtype

        shape = node.shape
        v = storage.variable(dtype=dtype, shape=shape, name=name)
        v.set_value(node)
        variables[name] = v

    return variables


def _to_constants(input_nodes, output_nodes, variable_values, constant_values,
                  value_dtypes,
                  default_placeholder_dtype, default_variable_dtype,
                  default_constant_dtype, default_operator_dtype):

    constants = collections.OrderedDict()

    for name, node in constant_values.items():
        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_constant_dtype

        shape = node.shape
        c = storage.constant(value=node,
                             dtype=dtype, shape=shape, name=name)
        constants[name] = c

    return constants
