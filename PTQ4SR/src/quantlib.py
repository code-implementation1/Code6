# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
''' quantization library'''
import numpy as np
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter, ops

class QuantActivation(nn.Cell):
    '''
        qiantize activation with given bit-width
    '''
    def __init__(self, bit=8):
        super(QuantActivation, self).__init__()
        self.min_val, self.max_val = -2**(bit-1), 2**(bit-1)-1
        self.scale_a = Parameter(Tensor(1., dtype=mstype.float32), name="alpha_a", requires_grad=True)
        self.offset_a = Parameter(Tensor(0., dtype=mstype.float32), name="beta_a", requires_grad=True)

    def construct(self, x):
        x_int = ops.round((x-self.offset_a)/self.scale_a)
        x_int = ops.clip_by_value(x_int, self.min_val, self.max_val)
        x_deq = x_int * self.scale_a + self.offset_a
        return x_deq

class QuantWeight(nn.Cell):
    '''
        qiantize weight with given bit-width
    '''
    def __init__(self, channels, bit=8):
        super(QuantWeight, self).__init__()
        self.min_val, self.max_val = -2**(bit-1), 2**(bit-1)-1
        self.scale_w = Parameter(Tensor(np.ones((1, channels, 1, 1)), \
                                   dtype=mstype.float32), name="scale_w", requires_grad=True)

    def construct(self, x):
        x_int = ops.round(x/self.scale_w)
        x_int = ops.clip_by_value(x_int, self.min_val, self.max_val)
        x_deq = x_int * self.scale_w
        return x_deq

class QuantConv2d(nn.Conv2d):
    '''
        Quantization Convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, \
                 group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW", \
                    a_bit=1, w_bit=1):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, pad_mode, \
                                          padding, dilation, group, has_bias, weight_init, bias_init, \
                                            data_format)
        self.a_bit = a_bit
        self.w_bit = w_bit

        self.act_quantizer = QuantActivation(bit=a_bit)
        self.wgt_quantizer = QuantWeight(out_channels, bit=w_bit)

        self.conv2d = P.Conv2D(out_channel=out_channels,
                               kernel_size=kernel_size,
                               mode=1,
                               pad_mode=pad_mode,
                               pad=padding,
                               stride=stride,
                               dilation=dilation,
                               group=group)

    def construct(self, x):
        if self.a_bit != 32:
            x = self.act_quantizer(x)
        if self.w_bit != 1:
            weight = self.wgt_quantizer(self.weight)
        else:
            weight = self.weight
        output = self.conv2d(x, weight)
        return output
