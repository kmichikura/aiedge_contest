from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from . import util


def normalize(visitor, node):
    input = node.args[0]
    scale = node.args[1]
    bias = node.args[2]
    shamt = node.args[3]

    visitor.visit(input)
    visitor.visit(scale)
    visitor.visit(bias)
    visitor.visit(shamt)

    scale_value = scale.value
    if isinstance(scale_value, (tuple, list)):
        scale_value = np.array(scale_value)

    q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value,
                                                                   scale.dtype.width)
    scale.set_value(q_scale_value)
    scale.scale_factor = scale_scale_factor

    bias_value = bias.value
    if isinstance(bias_value, (tuple, list)):
        bias_value = np.array(bias_value)

    q_bias_value = util.quantize_linear_by_scale_factor(
        bias_value, bias.dtype.width, input.scale_factor * scale_scale_factor)
    bias.set_value(q_bias_value)
    bias.scale_factor = input.scale_factor * scale_scale_factor

    q_shamt = find_optimal_shamt_normalize(visitor, node, q_scale_value, q_bias_value)
    shamt.fill_value = q_shamt
    node.scale_factor = input.scale_factor * scale.scale_factor / (2 ** q_shamt)


def find_optimal_shamt_normalize(visitor, node, scale, bias,
                                 allowed_rate=0.0, range_rate=0.95,
                                 init_shamt=0):

    shamt = init_shamt

    input = node.args[0].eval(visitor.memo, visitor.input_dict)

    if node.dtype.signed:
        _range = round((2 ** (node.dtype.width - 1)) * range_rate)
    else:
        _range = round((2 ** node.dtype.width) * range_rate)

    while True:
        rslt = try_shamt_normalize(node, input, scale, bias, shamt)
        neg_overflow = np.where(rslt <= - _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        pos_overflow = np.where(rslt >= _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        num_overflow = np.sum(neg_overflow + pos_overflow)

        rate = num_overflow / rslt.size
        if rate <= allowed_rate:
            break

        shamt += 1

    visitor.memo[id(node)] = rslt

    return shamt


def try_shamt_normalize(node, x, y, z, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['dtype'] = node.dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par'] = node.par

    return method(x, y, z, shamt, **kwargs)


def scaled_add(visitor, node):
    a = node.args[0]
    b = node.args[1]

    visitor.visit(a)
    visitor.visit(b)

    max_scale_factor = max(a.scale_factor, b.scale_factor)
    a_scale_value = max_scale_factor / a.scale_factor
    b_scale_value = max_scale_factor / b.scale_factor

    if max_scale_factor == a.scale_factor:
        q_b_scale_value, b_scale_scale_factor = util.quantize_linear_scale(b_scale_value, 32,
                                                                           allowed_rate=0.001)
        a_scale_value = np.round(b_scale_scale_factor).astype(np.int64)
        q_a_scale_value = a_scale_value
        a_scale_scale_factor = a_scale_value
    elif max_scale_factor == b.scale_factor:
        q_a_scale_value, a_scale_scale_factor = util.quantize_linear_scale(a_scale_value, 32,
                                                                           allowed_rate=0.001)
        b_scale_value = np.round(a_scale_scale_factor).astype(np.int64)
        q_b_scale_value = b_scale_value
        b_scale_scale_factor = b_scale_value

    node.a_scale = int(q_a_scale_value)
    node.b_scale = int(q_b_scale_value)

    q_shamt = find_optimal_shamt_scaled_add(visitor, node, q_a_scale_value, q_b_scale_value)
    node.shamt = q_shamt
    node.scale_factor = max(a.scale_factor * a_scale_scale_factor,
                            b.scale_factor * b_scale_scale_factor) / (2 ** q_shamt)


def find_optimal_shamt_scaled_add(visitor, node, a_scale, b_scale,
                                  allowed_rate=0.0, range_rate=0.95,
                                  init_shamt=0):

    shamt = init_shamt

    a_input = node.args[0].eval(visitor.memo, visitor.input_dict)
    b_input = node.args[1].eval(visitor.memo, visitor.input_dict)

    if node.dtype.signed:
        _range = round((2 ** (node.dtype.width - 1)) * range_rate)
    else:
        _range = round((2 ** node.dtype.width) * range_rate)

    while True:
        rslt = try_shamt_scaled_add(node, a_input, a_scale,
                                    b_input, b_scale, shamt)
        neg_overflow = np.where(rslt <= - _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        pos_overflow = np.where(rslt >= _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        num_overflow = np.sum(neg_overflow + pos_overflow)

        rate = num_overflow / rslt.size
        if rate <= allowed_rate:
            break

        shamt += 1

    visitor.memo[id(node)] = rslt

    return shamt


def try_shamt_scaled_add(node, a, a_scale, b, b_scale, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['dtype'] = node.dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par'] = node.par

    return method(a, b, a_scale, b_scale, shamt, **kwargs)


def scaled_concat(visitor, node):
    values = node.args

    for value in values:
        visitor.visit(value)

    max_scale_factor = max(*[value.scale_factor for value in values])
    scale_values = [max_scale_factor / value.scale_factor for value in values]

    new_scale_values = []
    scale_scale_factors = []

    for value, scale_value in zip(values, scale_values):
        if max_scale_factor != value.scale_factor:
            q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value, 32,
                                                                           allowed_rate=0.001)
            new_scale_values.append(q_scale_value)
            scale_scale_factors.append(scale_scale_factor)
        else:
            new_scale_values.append(1)
            scale_scale_factors.append(1.0)

    max_scale_scale_factor = max(*scale_scale_factors)

    new_scales = []
    new_scale_scale_factors = []

    for value, new_scale_value, scale_scale_factor in zip(values,
                                                          new_scale_values, scale_scale_factors):

        mag = np.round(max_scale_scale_factor / scale_scale_factor).astype(np.int64)
        scale_value = new_scale_value * mag
        q_scale_value = scale_value
        scale_scale_factor = scale_scale_factor * mag
        new_scales.append(int(scale_value))
        new_scale_scale_factors.append(scale_scale_factor)

    node.scales = new_scales

    q_shamt = find_optimal_shamt_scaled_concat(visitor, node, new_scales)
    node.shamt = q_shamt
    node.scale_factor = max(*[value.scale_factor * scale_scale_factor
                              for value, scale_scale_factor in zip(
                                  values, new_scale_scale_factors)]) / (2 ** q_shamt)


def find_optimal_shamt_scaled_concat(visitor, node, scales,
                                     allowed_rate=0.0, range_rate=0.95,
                                     init_shamt=0):

    shamt = init_shamt

    inputs = [arg.eval(visitor.memo, visitor.input_dict) for arg in node.args]

    if node.dtype.signed:
        _range = round((2 ** (node.dtype.width - 1)) * range_rate)
    else:
        _range = round((2 ** node.dtype.width) * range_rate)

    while True:
        rslt = try_shamt_scaled_concat(node, inputs, scales, shamt)
        neg_overflow = np.where(rslt <= - _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        pos_overflow = np.where(rslt >= _range,
                                np.ones_like(rslt), np.zeros_like(rslt))
        num_overflow = np.sum(neg_overflow + pos_overflow)

        rate = num_overflow / rslt.size
        if rate <= allowed_rate:
            break

        shamt += 1

    visitor.memo[id(node)] = rslt

    return shamt


def try_shamt_scaled_concat(node, values, scales, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['axis'] = node.axis
    kwargs['dtype'] = node.dtype
    kwargs['mul_dtype'] = node.mul_dtype
    kwargs['name'] = node.name

    return method(values, scales, shamt, **kwargs)
