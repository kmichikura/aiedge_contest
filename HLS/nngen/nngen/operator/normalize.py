from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import functools
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
from . import basic
from . import concat


class normalize(basic.multiply_add_rshift_clip):
    """ for Batchnorm """
    pass


class scaled_add(bt._ElementwiseOperator):
    """ for ResBlock """
    input_chainable = True
    output_chainable = True

    def __sub_str__(self):
        a_scale = ' a_scale:%d' % self.a_scale
        b_scale = ' b_scale:%d' % self.b_scale
        shamt = ' shamt:%d' % self.shamt
        ret = [a_scale, b_scale, shamt]
        return ''.join(ret)

    def get_local_control_param_values(self):
        return OrderedDict([('a_scale_cparam', self.a_scale),
                            ('b_scale_cparam', self.b_scale),
                            ('shamt_cparam', self.shamt)])

    def op(self, strm, *args, **kwargs):
        a_datawidth = self.args[0].get_op_width()
        a_point = self.args[0].get_op_point()
        a_signed = self.args[0].get_signed()
        b_datawidth = self.args[1].get_op_width()
        b_point = self.args[1].get_op_point()
        b_signed = self.args[1].get_signed()

        a_scale = strm.ReinterpretCast(self.a_scale_cparam,
                                       width=a_datawidth,
                                       signed=a_signed)
        b_scale = strm.ReinterpretCast(self.b_scale_cparam,
                                       width=b_datawidth,
                                       signed=b_signed)

        mul = strm.Times(args[0], a_scale)

        if self.sum_dtype is not None:
            mul.width = self.sum_dtype.width
            mul.signed = self.sum_dtype.signed
        else:
            mul.width = a_datawidth + self.a_scale_cparam.bit_length()
            mul.signed = self.dtype.signed

        if self.sum_dtype is not None and mul.point != self.sum_dtype.point:
            mul = strm.Cast(mul, point=self.sum_dtype.point)

        madd = strm.Madd(args[1], b_scale, mul)

        if self.sum_dtype is not None:
            madd.width = self.sum_dtype.width
            madd.signed = self.sum_dtype.signed
        else:
            madd.width = max(b_datawidth + self.b_scale_cparam.bit_length(), mul.width)
            madd.signed = self.dtype.signed

        if self.sum_dtype is not None and madd.point != self.sum_dtype.point:
            madd = strm.Cast(madd, point=self.sum_dtype.point)

        shamt = strm.ReinterpretCast(self.shamt_cparam,
                                     width=self.shamt_cparam.width,
                                     signed=False)
        sra = strm.Sra(madd, shamt)

        width = self.dtype.width
        p_th = (1 << (width - 1)) - 1
        n_th = -1 * p_th

        p = strm.Mux(sra > p_th, p_th, sra)
        n = strm.Mux(sra < n_th, n_th, sra)
        return strm.Mux(sra >= 0, p, n)

    def __init__(self, a, b, a_scale, b_scale, shamt,
                 dtype=None, sum_dtype=None, name=None, par=1):

        shape = None
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.shamt = shamt
        bt._ElementwiseOperator.__init__(self, a, b,
                                         dtype=dtype, shape=shape, name=name, par=par)
        self.sum_dtype = sum_dtype

    def eval(self, memo, input_dict, **kwargs):
        kwargs['a_scale'] = self.a_scale
        kwargs['b_scale'] = self.b_scale
        kwargs['shamt'] = self.shamt
        kwargs['a_dtype'] = self.args[0].dtype
        kwargs['b_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class scaled_concat(concat):

    def __sub_str__(self):
        base = concat.__sub_str__(self)
        scales = ' scales:%s' % str(tuple(self.scales))
        shamt = ' shamt:%d' % self.shamt
        return ''.join([base, scales, shamt])

    def get_control_param_values(self):
        ret = concat.get_control_param_values(self)

        buffered = True
        ret['buffered'] = buffered

        # for __str__
        self.buffered_value = buffered

        return ret

    def get_local_control_param_values(self):
        return OrderedDict([('scale_cparams', self.scales),
                            ('shamt_cparam', self.shamt)])

    def __init__(self, values, scales, shamt, axis,
                 dtype=None, mul_dtype=None, name=None):
        if not isinstance(scales, (tuple, list)):
            raise TypeError('scales must be tuple or list.')

        # for quantization, scales must be list
        self.scales = list(scales)
        self.shamt = shamt
        concat.__init__(self, values, axis, dtype, name)
        self.mul_dtype = mul_dtype

    def get_stream_func(self):
        def func(strm):
            datawidth = self.args[0].get_op_width()

            src = strm.source(datawidth=datawidth)
            sel = strm.constant(datawidth=bt.log_width(len(self.scale_cparams)),
                                signed=False)

            scale_width = max(*[scale.width for scale in self.scale_cparams])
            scale_signed = max(*[1 if scale.signed else 0 for scale in self.scale_cparams]) == 1
            selected_scale = strm.ReinterpretCast(self.scale_cparams[-1],
                                                  width=scale_width,
                                                  signed=scale_signed)

            for i, scale in enumerate(reversed(self.scale_cparams[:-1])):
                scale = strm.ReinterpretCast(scale,
                                             width=scale_width,
                                             signed=scale_signed)
                selected_scale = strm.Mux(sel == len(self.scale_cparams) - 2 - i,
                                          scale, selected_scale)
                selected_scale.width = scale_width
                selected_scale.signed = scale_signed
                selected_scale.latency = 0

            mul = strm.Times(src, selected_scale)

            if self.mul_dtype is not None:
                mul.width = self.mul_dtype.width
                mul.signed = self.mul_dtype.signed
            else:
                mul.width = datawidth + scale_width
                mul.signed = self.dtype.signed

            if self.mul_dtype is not None and mul.point != self.mul_dtype.point:
                mul = strm.Cast(mul, point=self.mul_dtype.point)

            shamt = strm.ReinterpretCast(self.shamt_cparam,
                                         width=self.shamt_cparam.width,
                                         signed=False)
            sra = strm.Sra(mul, shamt)

            width = self.dtype.width
            p_th = (1 << (width - 1)) - 1
            n_th = -1 * p_th

            p = strm.Mux(sra > p_th, p_th, sra)
            n = strm.Mux(sra < n_th, n_th, sra)
            dst = strm.Mux(sra >= 0, p, n)

            strm.sink(dst)

        return func

    def control_sequence(self, fsm):
        arg_gaddrs = [self.m.Reg(self._name('arg_gaddr_%d' % i),
                                 self.maxi.addrwidth, initval=0)
                      for i, _ in enumerate(self.arg_objaddrs)]
        out_gaddr = self.m.Reg(self._name('out_gaddr'),
                               self.maxi.addrwidth, initval=0)

        arg_laddr = self.m.Reg(self._name('arg_laddr'),
                               self.maxi.addrwidth, initval=0)
        copy_laddr = self.m.Reg(self._name('copy_laddr'),
                                self.maxi.addrwidth, initval=0)
        copy_size = self.m.Reg(self._name('copy_size'),
                               self.maxi.addrwidth, initval=0)
        sum_read_sizes = self.m.Wire(self._name('sum_read_sizes'),
                                     max([i.width for i in self.arg_read_sizes])
                                     + int(math.ceil(math.log2(len(self.arg_read_sizes)))))
        sum_read_sizes.assign(vg.Add(*self.arg_read_sizes))

        out_addr_inc_unbuffered = self.m.Reg(self._name('out_addr_inc_unbuffered'),
                                             self.maxi.addrwidth, initval=0)

        arg_select = self.m.Reg(self._name('arg_select'),
                                int(max(math.ceil(math.log(len(self.args), 2)), 1)),
                                initval=0)
        prev_arg_select = self.m.Reg(self._name('prev_arg_select'),
                                     int(max(math.ceil(math.log(len(self.args), 2)), 1)),
                                     initval=0)
        arg_chunk_count = self.m.Reg(self._name('arg_chunk_count'),
                                     self.maxi.addrwidth + 1, initval=0)
        out_count = self.m.Reg(self._name('out_count'),
                               self.maxi.addrwidth + 1, initval=0)

        # --------------------
        # initialization phase
        # --------------------
        fsm(
            [arg_gaddr(0) for arg_gaddr in arg_gaddrs],
            out_gaddr(0),
            arg_laddr(0),
            copy_laddr(0),
            copy_size(0),
            out_addr_inc_unbuffered(0),
            arg_select(0),
            prev_arg_select(0),
            arg_chunk_count(0),
            out_count(0)
        )

        fsm.goto_next()

        # --------------------
        # Read phase
        # --------------------
        state_read = fsm.current

        fsm.inc()

        state_read_begin_list = []
        state_read_end_list = []

        for (arg,
             arg_objaddr, arg_gaddr, arg_addr_inc,
             arg_read_size, arg_chunk_size) in zip(
                 self.args,
                 self.arg_objaddrs, arg_gaddrs, self.arg_addr_incs,
                 self.arg_read_sizes, self.arg_chunk_sizes):

            b = fsm.current
            state_read_begin_list.append(b)

            # normal
            laddr = arg_laddr
            gaddr = arg_objaddr + arg_gaddr

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, self.input_rams[0], laddr, gaddr, arg_read_size)
            bt.bus_unlock(self.maxi, fsm)

            fsm(
                arg_gaddr.add(arg_addr_inc),
                arg_laddr.add(arg_read_size),
                arg_chunk_count.inc(),
                copy_size(arg_read_size),
                out_addr_inc_unbuffered(arg_addr_inc)
            )
            fsm.If(arg_chunk_count == arg_chunk_size - 1)(
                arg_chunk_count(0),
                arg_select.inc()
            )
            fsm.If(arg_chunk_count == arg_chunk_size - 1,
                   arg_select == len(self.args) - 1)(
                arg_select(0)
            )
            fsm(
                prev_arg_select(arg_select)
            )

            e = fsm.current
            state_read_end_list.append(e)

            fsm.inc()

        state_read_end = fsm.current

        for i, b in enumerate(state_read_begin_list):
            fsm.If(arg_select == i).goto_from(state_read, b)

        for i, e in enumerate(state_read_end_list):
            fsm.goto_from(e, state_read_end)

        # --------------------
        # Copy phase
        # --------------------
        state_copy = fsm.current

        name = list(self.stream.sources.keys())[0]
        self.stream.set_source(fsm, name, self.input_rams[0], 0, copy_size)
        fsm.set_index(fsm.current - 1)

        name = list(self.stream.constants.keys())[0]
        self.stream.set_constant(fsm, name, prev_arg_select)
        fsm.set_index(fsm.current - 1)

        name = list(self.stream.sinks.keys())[0]
        self.stream.set_sink(fsm, name, self.output_rams[0], copy_laddr, copy_size)
        self.stream.run(fsm)
        self.stream.join(fsm)

        fsm(
            arg_laddr(0),
            copy_laddr.add(copy_size)
        )
        fsm.goto_next()

        fsm.If(copy_laddr < sum_read_sizes).goto(state_read)
        fsm.If(copy_laddr >= sum_read_sizes).goto_next()

        state_copy_end = fsm.current
        fsm.If(vg.Not(self.buffered)).goto_from(state_copy, state_copy_end)

        # --------------------
        # Write phase
        # --------------------
        state_write = fsm.current
        fsm.inc()

        # Case with Copy
        state_write_buffered = fsm.current

        laddr = 0
        gaddr = self.objaddr + out_gaddr
        bt.bus_lock(self.maxi, fsm)
        bt.dma_write(self.maxi, fsm,
                     self.output_rams[0], laddr, gaddr, self.out_write_size)
        bt.bus_unlock(self.maxi, fsm)

        fsm(
            copy_laddr(0),
            out_gaddr.add(self.out_addr_inc),
            out_count.inc()
        )

        state_write_end_buffered = fsm.current
        fsm.inc()

        # Case without Copy
        state_write_unbuffered = fsm.current

        laddr = 0
        gaddr = self.objaddr + out_gaddr
        bt.bus_lock(self.maxi, fsm)
        bt.dma_write(self.maxi, fsm,
                     self.input_rams[0], laddr, gaddr, copy_size)
        bt.bus_unlock(self.maxi, fsm)

        fsm(
            arg_laddr(0),
            out_gaddr.add(out_addr_inc_unbuffered),
            out_count.inc()
        )

        state_write_end_unbuffered = fsm.current
        fsm.inc()

        state_write_end = fsm.current

        fsm.If(self.buffered).goto_from(state_write, state_write_buffered)
        fsm.If(vg.Not(self.buffered)).goto_from(state_write, state_write_unbuffered)

        fsm.goto_from(state_write_end_buffered, state_write_end)
        fsm.goto_from(state_write_end_unbuffered, state_write_end)

        # --------------------
        # update for next iteration
        # --------------------
        fsm.If(out_count < self.num_steps).goto(state_read)
        fsm.If(out_count == self.num_steps).goto_next()

    def eval(self, memo, input_dict, **kwargs):
        kwargs['scales'] = self.scales
        kwargs['shamt'] = self.shamt
        kwargs['mul_dtype'] = self.mul_dtype
        return concat.eval(self, memo, input_dict, **kwargs)
