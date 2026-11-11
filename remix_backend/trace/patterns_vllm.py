#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false
import torch
from tads.ops.utils import register_tads_op
from tads.ops.torch_fx_patterns import mirror_signature, BasePattern, PassReplace
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from tads.torch_compile_backend.dshape_prop import dshape_prop, is_input
from typing import List, Tuple


@torch.library.custom_op("tads_ops::llama_rms", mutates_args=(), device_types=["cpu", "cuda"])
def llama_rms_impl(
    x: Tensor,
    dtype_0: torch.dtype,
    exponent: int,
    mean_dim: List[int],
    keep_dim: bool,
    norm: float,
    dtype_1: torch.dtype,
    m: Tensor,
) -> Tensor:
    x = torch.ops.aten._to_copy.default(x, dtype=dtype_0)
    pow = torch.ops.aten.pow.Tensor_Scalar(x, exponent)
    mean = torch.ops.aten.mean.dim(pow, mean_dim, keep_dim)
    normalized_mean = torch.ops.aten.add.Tensor(mean, norm)
    sqrt = torch.ops.aten.rsqrt.default(normalized_mean)
    mul = torch.ops.aten.mul.Tensor(x, sqrt)
    to_copy = torch.ops.aten._to_copy.default(mul, dtype=dtype_1)
    r = torch.ops.aten.mul.Tensor(m, to_copy)
    return r

class StageDense(BasePattern):
    @staticmethod
    def pattern_to_match(x, w1):
        return torch.ops.tads_ops.dense_layer.default(x, w1)

    @staticmethod
    def replacement(x, w1):
        x_stage = torch.ops.tads_ops.staging.default(x)
        x_stage.node.meta["mapping"] = "staging_macro"
        out = torch.ops.tads_ops.dense_layer.default(x_stage, w1, None)
        # out_staged = torch.ops.tads_ops.staging.default(out)
        # out_staged.node.meta["mapping"] = "staging_macro"
        return out

class LastBambaDense(BasePattern):
    @staticmethod
    def pattern_to_match(staging_default_7, arg8_1, s0, s1, s2):
        dense_layer_default_7 = torch.ops.tads_ops.dense_layer.default(staging_default_7, arg8_1, None)
        slice_tensor_2 = torch.ops.aten.slice.Tensor(dense_layer_default_7, s0, s1, s2)
        return dense_layer_default_7,slice_tensor_2

    @staticmethod
    def replacement(staging_default_7, arg8_1, s0, s1, s2):
      dense_layer_default_7 = torch.ops.tads_ops.dense_layer.default(staging_default_7, arg8_1, None)
      d_stage = torch.ops.tads_ops.staging.default(dense_layer_default_7)
      d_stage.node.meta["mapping"] = "staging_macro"
      slice_tensor_2 = torch.ops.aten.slice.Tensor(d_stage,  s0, s1, s2)
      return d_stage,slice_tensor_2

@torch.library.register_fake("tads_ops::llama_rms")
def llama_rms_abstract(x, dtype_0, exponent, mean_dim, keep_dim, norm, dtype_1, m):
    return torch.empty_like(x)


def vllm_rms_pattern_impl(
    x: Tensor,
    weight: Tensor,
    exponent: int,
    mean_dim: List[int],
    keep_dim: bool,
    eps: float,
    to_copy_norm_dtype: torch.dtype,
) -> Tensor:
    pow_v = torch.ops.aten.pow.Tensor_Scalar(x, exponent)
    mean_v = torch.ops.aten.mean.dim(pow_v, mean_dim, keep_dim)
    add_eps = torch.ops.aten.add.Tensor(mean_v, eps)
    rsqrt_v = torch.ops.aten.rsqrt.default(add_eps)
    mul_v = torch.ops.aten.mul.Tensor(x, rsqrt_v)
    x_norm = torch.ops.aten._to_copy.default(mul_v, dtype=to_copy_norm_dtype)
    return torch.ops.aten.mul.Tensor(x_norm, weight)


@torch.library.custom_op("tads_ops::vllm_rms", mutates_args=(), device_types=["cpu", "cuda"])
@mirror_signature(vllm_rms_pattern_impl)
def vllm_rms(*args, **kwargs):
    return vllm_rms_pattern_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::vllm_rms")
@mirror_signature(vllm_rms_pattern_impl)
def vllm_rms_abstract(*args, **kwargs):
    return vllm_rms_pattern_impl(*args, **kwargs)

@register_tads_op("grouped_mm")
def grouped_mm_impl(
    self: Tensor,
    mat2: Tensor,
    offs: Tensor,
    transpose_5_arg_0: int,
    transpose_5_arg_1: int,
) -> Tensor:
    transpose_5 = torch.ops.aten.transpose.int(mat2, transpose_5_arg_0, transpose_5_arg_1)
    return torch.ops.aten._grouped_mm.default(self, transpose_5, offs)


# @torch.library.custom_op("tads_ops::grouped_mm", mutates_args=(), device_types=["cuda"])
# @mirror_signature(grouped_mm_impl)
# def grouped_mm(*args, **kwargs):
#     return grouped_mm_impl(*args, **kwargs)


# @torch.library.register_fake("tads_ops::grouped_mm")
# @mirror_signature(grouped_mm_impl)
# def grouped_mm_abstract(*args, **kwargs):
#     return grouped_mm_impl(*args, **kwargs)


class GroupedMMPattern(BasePattern):
    
    @staticmethod
    def pattern_to_match(act, mat2, t0, t1, offs):
        t = torch.ops.aten.transpose.int(mat2, t0, t1)
        return torch.ops.aten._grouped_mm.default(act, t, offs)
    # pattern_to_match = staticmethod(grouped_mm_impl)

    @staticmethod
    def replacement(act, mat2, t0, t1, offs):
        act_staged = torch.ops.tads_ops.staging.default(act)
        act_staged.node.meta["mapping"] = "staging_macro"
        return torch.ops.tads_ops.grouped_mm.default(act_staged, mat2, offs, t0, t1)


class GroupedMMMul(BasePattern):
    @staticmethod
    def pattern_to_match(
        arg15_1,
        mul_10,
        cumsum_default,
        unsqueeze_14,
        grouped_mm_default_1_arg_0,
        grouped_mm_default_1_arg_1,
    ):
        grouped_mm_default_1 = torch.ops.tads_ops.grouped_mm.default(
            arg15_1, mul_10, cumsum_default, grouped_mm_default_1_arg_0, grouped_mm_default_1_arg_1
        )
        mul_11 = torch.ops.aten.mul.Tensor(grouped_mm_default_1, unsqueeze_14)
        return mul_11

    @staticmethod
    def replacement(
        arg15_1,
        mul_10,
        cumsum_default,
        unsqueeze_14,
        grouped_mm_default_1_arg_0,
        grouped_mm_default_1_arg_1,
    ):
        grouped_mm_default_1 = torch.ops.tads_ops.grouped_mm.default(
            arg15_1,
            mul_10,
            cumsum_default,
            grouped_mm_default_1_arg_0,
            grouped_mm_default_1_arg_1,
        )
        mm_staged = torch.ops.tads_ops.staging.default(grouped_mm_default_1)
        mm_staged.node.meta["mapping"] = "staging_macro"
        mul_11 = torch.ops.aten.mul.Tensor(mm_staged, unsqueeze_14)
        mul_11.node.meta["mapping"] = "macro_mul"
        return mul_11

class AliasToInput(PassReplace):
    """Replace aten.alias(x) → x (alias is a no-op view used by vLLM for weights)."""

    def transform(self, graph_module: torch.fx.GraphModule):
        nodes_to_remove = []
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and node.target == torch.ops.aten.alias.default:
                node.replace_all_uses_with(node.args[0])
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            graph_module.graph.erase_node(node)
        graph_module.graph.lint()
        graph_module.recompile()


class VllmRMSNorm(BasePattern):
    """vLLM RMSNorm: fuses RMSNorm into one op (residual add and residual cast stay in graph).
    Must run after AliasToInput (which strips aten.alias from weight tensors)."""

    pattern_to_match = staticmethod(vllm_rms_pattern_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        return torch.ops.tads_ops.vllm_rms.default(*args, **kwargs)


class FuseKQV(BasePattern):
    @staticmethod
    def pattern_to_match(x, shape_1, shape_2, shape_3, w1, w2, w3):
        s1 = torch.ops.aten.reshape.default(x, shape_1)
        s1 = torch.ops.tads_ops.dense_layer.default(s1, w1)
        s2 = torch.ops.aten.reshape.default(x, shape_2)
        s2 = torch.ops.tads_ops.dense_layer.default(s2, w2)
        s3 = torch.ops.aten.reshape.default(x, shape_3)
        s3 = torch.ops.tads_ops.dense_layer.default(s3, w3)
        return (s1, s2, s3)

    @staticmethod
    def replacement(x, shape_1, shape_2, shape_3, w1, w2, w3):
        x = torch.ops.tads_ops.staging.default(x)
        x.node.meta["mapping"] = "staging_macro"
        w_fused = torch.ops.aten.cat.default((w1, w2, w3))
        w_fused.node.meta["mapping"] = "dont_map"
        m_fused = torch.ops.tads_ops.dense_layer.default(x, w_fused, None)
        m_staged = torch.ops.tads_ops.staging.default(m_fused)
        m_staged.node.meta["mapping"] = "staging_macro"
        m = torch.ops.aten.split_with_sizes.default(
            m_staged, [w1.shape[0], w2.shape[0], w3.shape[0]], -1
        )
        m1, m2, m3 = m[0], m[1], m[2]
        return (m1, m2, m3)


class SiluFP32Cast(BasePattern):
    """Normalize vLLM's gate-MLP SiLU:
    slice(x,dim,0,half) → _to_copy(fp32) → sigmoid → x*sigmoid → _to_copy(bf16) + slice(x,dim,half,end) → mul
    into: split(x, half, dim) → silu(gate) → mul(silu, up).
    Must run before FuseGM."""

    @staticmethod
    def pattern_to_match(
        x,
        gate_dim: int,
        gate_end: int,
        up_start: int,
        up_end: int,
        dtype_in: torch.dtype,
        dtype_out: torch.dtype,
    ):
        gate = torch.ops.aten.slice.Tensor(x, gate_dim, 0, gate_end)
        gate_fp32 = torch.ops.aten._to_copy.default(gate, dtype=dtype_in)
        sig = torch.ops.aten.sigmoid.default(gate_fp32)
        silu_val = torch.ops.aten.mul.Tensor(gate_fp32, sig)
        gate_out = torch.ops.aten._to_copy.default(silu_val, dtype=dtype_out)
        up = torch.ops.aten.slice.Tensor(x, gate_dim, up_start, up_end)
        return torch.ops.aten.mul.Tensor(gate_out, up)

    @staticmethod
    def replacement(
        x,
        gate_dim: int,
        gate_end: int,
        up_start: int,
        up_end: int,
        dtype_in: torch.dtype,
        dtype_out: torch.dtype,
    ):
        split_t = torch.ops.aten.split.Tensor(x, gate_end, gate_dim)
        gate = split_t[0]
        up = split_t[1]
        silu_val = torch.ops.aten.silu.default(gate)
        return torch.ops.aten.mul.Tensor(silu_val, up)


class FuseGM(BasePattern):
    """vLLM gate-MLP: weights are pre-fused (gate+up in one tensor).
    Pattern: dense_layer → split(gate_end, dim) → silu(gate) → mul(silu, up).
    Requires SiluFP32Cast to run first (normalizes slice pairs to split)."""

    @staticmethod
    def pattern_to_match(
        x,
        w_gate_up,
        split_dim: int,
        split_size: int,
    ):
        dl = torch.ops.tads_ops.dense_layer.default(x, w_gate_up, None)
        split_t = torch.ops.aten.split.Tensor(dl, split_size, split_dim)
        gate = split_t[0]
        up = split_t[1]
        silu_gate = torch.ops.aten.silu.default(gate)
        gate_up = torch.ops.aten.mul.Tensor(silu_gate, up)
        return gate_up

    @staticmethod
    def replacement(
        x,
        w_gate_up,
        split_dim: int,
        split_size: int,
    ):
        x = torch.ops.tads_ops.staging.default(x)
        x.node.meta["mapping"] = "staging_macro"
        dl = torch.ops.tads_ops.dense_layer.default(x, w_gate_up, None)
        split_t = torch.ops.aten.split.Tensor(dl, split_size, split_dim)
        gate = split_t[0]
        up = split_t[1]
        silu_gate = torch.ops.aten.silu.default(gate)
        gate_up = torch.ops.aten.mul.Tensor(silu_gate, up)
        ret = torch.ops.tads_ops.staging.default(gate_up)
        ret.node.meta["mapping"] = "staging_macro"
        return ret


class IgnoreMulinRoPE(BasePattern):
    @staticmethod
    def pattern_to_match(transpose_2, unsqueeze_3, cat_2, unsqueeze_4, transpose_1, cat_1):
        mul_4 = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze_3)
        mul_5 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_4)
        mul_6 = torch.ops.aten.mul.Tensor(transpose_2, unsqueeze_3)
        mul_7 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_4)
        return mul_4, mul_5, mul_6, mul_7

    @staticmethod
    def replacement(transpose_2, unsqueeze_3, cat_2, unsqueeze_4, transpose_1, cat_1):
        mul_4 = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze_3)
        mul_5 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_4)
        mul_6 = torch.ops.aten.mul.Tensor(transpose_2, unsqueeze_3)
        mul_7 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_4)
        mul_4.node.meta["mapping"] = "dont_map"
        mul_5.node.meta["mapping"] = "dont_map"
        mul_6.node.meta["mapping"] = "dont_map"
        mul_7.node.meta["mapping"] = "dont_map"
        return mul_4, mul_5, mul_6, mul_7


class IgnoreSymAdd(PassReplace):
    def transform(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if (
                "add" in node.name
                and "val" in node.meta
                and isinstance(node.meta["val"], torch.SymInt)
            ):
                node.meta["mapping"] = "dont_map"


class IgnoreLMHead(PassReplace):
    def __init__(self, input_names, inputs, vocab_size):
        super().__init__(input_names, inputs)
        self.vocab_size = vocab_size

    def requires(self, graph_module: torch.fx.GraphModule) -> None:
        # Determine the shapes and dtypes of all nodes.
        dshape_prop(graph_module, inputs=self.inputs)
        pass

    def transform(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if (
                is_input(node)
                and "val" in node.meta
                and isinstance(node.meta["val"], FakeTensor)
                and len(node.meta["val"].size()) == 2
                and node.meta["val"].size()[0] == self.vocab_size
            ):
                user = list(node.users.keys())[0]
                if "dense_layer" in user.name:
                    user.meta["mapping"] = "dont_map"


def positional_encoding_impl(
    arg8_1: Tensor,
    arg3_1: Tensor,
    unsqueeze_arg_0: int,
    unsqueeze_1_arg_0: int,
    unsqueeze_2_arg_0: int,
    _to_copy_arg_0: torch.dtype,
    transpose_arg_0: int,
    transpose_arg_1: int,
    cat_arg_0: int,
    mul_19_arg_0: float,
    mul_20_arg_0: float,
    _to_copy_1_arg_0: torch.dtype,
    _to_copy_2_arg_0: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    unsqueeze = torch.ops.aten.unsqueeze.default(arg8_1, unsqueeze_arg_0)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, unsqueeze_1_arg_0)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg3_1, unsqueeze_2_arg_0)
    _to_copy = torch.ops.aten._to_copy.default(unsqueeze_2, dtype=_to_copy_arg_0)
    bmm = torch.ops.aten.bmm.default(unsqueeze_1, _to_copy)
    transpose = torch.ops.aten.transpose.int(bmm, transpose_arg_0, transpose_arg_1)
    cat = torch.ops.aten.cat.default([transpose, transpose], cat_arg_0)
    cos = torch.ops.aten.cos.default(cat)
    mul_19 = torch.ops.aten.mul.Tensor(cos, mul_19_arg_0)
    sin = torch.ops.aten.sin.default(cat)
    mul_20 = torch.ops.aten.mul.Tensor(sin, mul_20_arg_0)
    _to_copy_1 = torch.ops.aten._to_copy.default(mul_19, dtype=_to_copy_1_arg_0)
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_20, dtype=_to_copy_2_arg_0)
    return _to_copy_1, _to_copy_2


@torch.library.custom_op(
    "tads_ops::positional_encoding", mutates_args=(), device_types=["cpu", "cuda"]
)
@mirror_signature(positional_encoding_impl)
def positional_encoding(*args, **kwargs):
    return positional_encoding_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::positional_encoding")
def positional_encoding_abstract(*args, **kwargs):
    return positional_encoding_impl(*args, **kwargs)


class PositionalEncoding(BasePattern):
    pattern_to_match = staticmethod(positional_encoding_impl)

    @staticmethod
    # @mirror_signature(positional_encoding_impl)
    def replacement_impl(*args, **kwargs):
        ret = torch.ops.tads_ops.positional_encoding.default(*args, **kwargs)
        ret0, ret1 = ret[0], ret[1]
        return ret0, ret1


def positional_encoding_2_impl(
    arg8_1: Tensor,
    arg3_1: Tensor,
    unsqueeze_arg_0: int,
    unsqueeze_1_arg_0: int,
    expand_1_arg_0: List[int],
    unsqueeze_2_arg_0: int,
    _to_copy_arg_0: torch.dtype,
    transpose_arg_0: int,
    transpose_arg_1: int,
    cat_arg_0: int,
    mul_19_arg_0: float,
    mul_20_arg_0: float,
    _to_copy_1_arg_0: torch.dtype,
    _to_copy_2_arg_0: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    unsqueeze = torch.ops.aten.unsqueeze.default(arg8_1, unsqueeze_arg_0)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, unsqueeze_1_arg_0)
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, expand_1_arg_0)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg3_1, unsqueeze_2_arg_0)
    _to_copy = torch.ops.aten._to_copy.default(unsqueeze_2, dtype=_to_copy_arg_0)
    bmm = torch.ops.aten.bmm.default(expand_1, _to_copy)
    transpose = torch.ops.aten.transpose.int(bmm, transpose_arg_0, transpose_arg_1)
    cat = torch.ops.aten.cat.default([transpose, transpose], cat_arg_0)
    cos = torch.ops.aten.cos.default(cat)
    mul_19 = torch.ops.aten.mul.Tensor(cos, mul_19_arg_0)
    sin = torch.ops.aten.sin.default(cat)
    mul_20 = torch.ops.aten.mul.Tensor(sin, mul_20_arg_0)
    _to_copy_1 = torch.ops.aten._to_copy.default(mul_19, dtype=_to_copy_1_arg_0)
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_20, dtype=_to_copy_2_arg_0)
    return _to_copy_1, _to_copy_2


@torch.library.custom_op(
    "tads_ops::positional_encoding_2", mutates_args=(), device_types=["cpu", "cuda"]
)
@mirror_signature(positional_encoding_2_impl)
def positional_encoding_2(*args, **kwargs):
    return positional_encoding_2_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::positional_encoding_2")
def positional_encoding_2_abstract(*args, **kwargs):
    return positional_encoding_2_impl(*args, **kwargs)


class PositionalEncoding2(BasePattern):
    pattern_to_match = staticmethod(positional_encoding_2_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        ret = torch.ops.tads_ops.positional_encoding_2.default(*args, **kwargs)
        ret0, ret1 = ret[0], ret[1]
        return ret0, ret1


PositionalEncodings = [PositionalEncoding, PositionalEncoding2]


def attention_mask_impl(
    arg2_1: Tensor,
    arange_2_arg_0: int,
    arange_2_arg_1: torch.device,
    arange_2_arg_2: bool,
    unsqueeze_arg_0: int,
    unsqueeze_1_arg_0: int,
    unsqueeze_2_arg_0: int,
    add_tensor_4_arg_0: int,
    unsqueeze_3_arg_0: int,
    unsqueeze_4_arg_0: int,
    unsqueeze_5_arg_0: int,
) -> Tensor:

    arange_2 = torch.ops.aten.arange.default(
        arange_2_arg_0, device=arange_2_arg_1, pin_memory=arange_2_arg_2
    )
    unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, unsqueeze_arg_0)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, unsqueeze_1_arg_0)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, unsqueeze_2_arg_0)
    add_tensor_4 = torch.ops.aten.add.Tensor(arange_2, add_tensor_4_arg_0)
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_tensor_4, unsqueeze_3_arg_0)
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, unsqueeze_4_arg_0)
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, unsqueeze_5_arg_0)
    return torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2)


@torch.library.custom_op("tads_ops::attention_mask", mutates_args=(), device_types=["cpu", "cuda"])
@mirror_signature(attention_mask_impl)
def attention_mask(*args, **kwargs) -> Tensor:
    return attention_mask_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::attention_mask")
def attention_mask_abstract(*args, **kwargs):
    return attention_mask_impl(*args, **kwargs)


class AttentionMask(BasePattern):

    pattern_to_match = staticmethod(attention_mask_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        ret = torch.ops.tads_ops.attention_mask.default(*args, **kwargs)
        ret.node.meta["hints"] = {"batching": "dont_split"}
        return ret


def attention_mask_2_impl(
    add_4: int,
    arg7_1: int,
    constant_pad_nd_arg_0: int,
    constant_pad_nd_arg_1: float,
    arg8_1: Tensor,
    arange_arg_0: torch.device,
    arange_arg_1: bool,
    arg0_1: int,
    arange_1_arg_0: torch.device,
    arange_1_arg_1: bool,
    arg3_1: Tensor,
    new_ones_arg_0: List[int],
    new_ones_arg_1: torch.dtype,
    new_ones_arg_2: bool,
    reshape_default_arg_0: List[int],
    reshape_default_1_arg_0: List[int],
    reshape_default_2_arg_0: int,
    reshape_default_3_arg_0: int,
    reshape_default_4_arg_0: int,
    reshape_default_4_arg_1: int,
    arg4_1: int,
) -> Tensor:
    sub_1 = add_4 - arg7_1
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
        arg8_1, [constant_pad_nd_arg_0, sub_1], constant_pad_nd_arg_1
    )
    arange = torch.ops.aten.arange.default(add_4, device=arange_arg_0, pin_memory=arange_arg_1)
    add_12 = torch.ops.aten.add.Tensor(arange, 0)
    arange_1 = torch.ops.aten.arange.default(
        arg0_1, device=arange_1_arg_0, pin_memory=arange_1_arg_1
    )
    new_ones = torch.ops.aten.new_ones.default(
        arg3_1, new_ones_arg_0, dtype=new_ones_arg_1, pin_memory=new_ones_arg_2
    )
    reshape_default = torch.ops.aten.reshape.default(arg3_1, reshape_default_arg_0)
    le = torch.ops.aten.le.Tensor(add_12, reshape_default)
    reshape_default_1 = torch.ops.aten.reshape.default(new_ones, reshape_default_1_arg_0)
    bitwise_and = torch.ops.aten.bitwise_and.Tensor(reshape_default_1, le)
    reshape_default_2 = torch.ops.aten.reshape.default(arange_1, [arg4_1, reshape_default_2_arg_0])
    index = torch.ops.aten.index.Tensor(constant_pad_nd, [reshape_default_2, add_12])
    reshape_default_3 = torch.ops.aten.reshape.default(
        index, [arg4_1, reshape_default_3_arg_0, add_4]
    )
    bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, reshape_default_3)
    reshape_default_4 = torch.ops.aten.reshape.default(
        bitwise_and_1, [arg4_1, reshape_default_4_arg_0, reshape_default_4_arg_1, add_4]
    )
    return reshape_default_4


@torch.library.custom_op(
    "tads_ops::attention_mask_2", mutates_args=(), device_types=["cpu", "cuda"]
)
@mirror_signature(attention_mask_2_impl)
def attention_mask_2(*args, **kwargs):
    return attention_mask_2_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::attention_mask_2")
def attention_mask_2_abstract(*args, **kwargs):
    return attention_mask_2_impl(*args, **kwargs)


class AttentionMask2(BasePattern):

    pattern_to_match = staticmethod(attention_mask_2_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        ret = torch.ops.tads_ops.attention_mask_2.default(*args, **kwargs)
        ret.node.meta["hints"] = {"batching": "root"}
        return ret


AttentionMasks = [AttentionMask, AttentionMask2]


def attention_impl(
    staging_default_1: Tensor,
    arg5_1: Tensor,
    arg11_1: Tensor,
    arg8_1: Tensor,
    arg9_1: Tensor,
    arg10_1: Tensor,
    getitem_20: Tensor,
    getitem_21: Tensor,
    attention_mask_default: Tensor,
    split_with_sizes_default_arg_0: int,
    reshape_default_50_arg_0: List[int],
    transpose_1_arg_0: int,
    transpose_1_arg_1: int,
    reshape_default_51_arg_0: List[int],
    transpose_2_arg_0: int,
    transpose_2_arg_1: int,
    reshape_default_52_arg_0: List[int],
    transpose_3_arg_0: int,
    transpose_3_arg_1: int,
    unsqueeze_3_arg_0: int,
    unsqueeze_4_arg_0: int,
    slice_4_arg_0: int,
    slice_4_arg_1: int,
    slice_4_arg_2: int,
    slice_5_arg_0: int,
    slice_5_arg_1: int,
    slice_5_arg_2: int,
    cat_1_arg_0: int,
    slice_6_arg_0: int,
    slice_6_arg_1: int,
    slice_6_arg_2: int,
    slice_7_arg_0: int,
    slice_7_arg_1: int,
    slice_7_arg_2: int,
    cat_2_arg_0: int,
    cat_3_arg_0: int,
    cat_4_arg_0: int,
    unsqueeze_5_arg_0: int,
    expand_4_arg_0: List[int],
    unsqueeze_6_arg_0: int,
    expand_5_arg_0: List[int],
    scalar_tensor_arg_0: torch.dtype,
    scalar_tensor_arg_1: torch.device,
    scalar_tensor_1_arg_0: float,
    scalar_tensor_1_arg_1: torch.dtype,
    scalar_tensor_1_arg_3: torch.device,
    constant_pad_nd_arg_0: List[int],
    constant_pad_nd_arg_1: float,
    slice_19_arg_0: int,
    slice_19_arg_1: int,
    slice_19_arg_2: int,
    expand_6_arg_0: List[int],
    reshape_default_34_arg_0: List[int],
    reshape_default_35_arg_0: List[int],
    _scaled_dot_product_efficient_attention_arg_0: bool,
    _scaled_dot_product_efficient_attention_arg_1: float,
    getitem_arg_0: int,
    transpose_4_arg_0: int,
    transpose_4_arg_1: int,
    reshape_default_53_arg_0: List[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    getattr_1 = arg8_1.shape
    getitem_8 = getattr_1[0]
    getattr_2 = arg9_1.shape
    getitem_9 = getattr_2[0]
    getattr_3 = arg10_1.shape
    getitem_10 = getattr_3[0]
    split_with_sizes_default = torch.ops.aten.split_with_sizes.default(
        staging_default_1, [getitem_8, getitem_9, getitem_10], split_with_sizes_default_arg_0
    )
    getitem_11 = split_with_sizes_default[0]
    getitem_12 = split_with_sizes_default[1]
    getitem_13 = split_with_sizes_default[2]
    reshape_default_50 = torch.ops.aten.reshape.default(getitem_11, reshape_default_50_arg_0)
    transpose_1 = torch.ops.aten.transpose.int(
        reshape_default_50, transpose_1_arg_0, transpose_1_arg_1
    )
    reshape_default_51 = torch.ops.aten.reshape.default(getitem_12, reshape_default_51_arg_0)
    transpose_2 = torch.ops.aten.transpose.int(
        reshape_default_51, transpose_2_arg_0, transpose_2_arg_1
    )
    reshape_default_52 = torch.ops.aten.reshape.default(getitem_13, reshape_default_52_arg_0)
    transpose_3 = torch.ops.aten.transpose.int(
        reshape_default_52, transpose_3_arg_0, transpose_3_arg_1
    )
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_20, unsqueeze_3_arg_0)
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(getitem_21, unsqueeze_4_arg_0)
    mul_4 = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze_3)
    slice_4 = torch.ops.aten.slice.Tensor(transpose_1, slice_4_arg_0, slice_4_arg_1, slice_4_arg_2)
    slice_5 = torch.ops.aten.slice.Tensor(transpose_1, slice_5_arg_0, slice_5_arg_1, slice_5_arg_2)
    neg = torch.ops.aten.neg.default(slice_5)
    cat_1 = torch.ops.aten.cat.default([neg, slice_4], cat_1_arg_0)
    mul_5 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_4)
    add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5)
    mul_6 = torch.ops.aten.mul.Tensor(transpose_2, unsqueeze_3)
    slice_6 = torch.ops.aten.slice.Tensor(transpose_2, slice_6_arg_0, slice_6_arg_1, slice_6_arg_2)
    slice_7 = torch.ops.aten.slice.Tensor(transpose_2, slice_7_arg_0, slice_7_arg_1, slice_7_arg_2)
    neg_1 = torch.ops.aten.neg.default(slice_7)
    cat_2 = torch.ops.aten.cat.default([neg_1, slice_6], cat_2_arg_0)
    mul_7 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_4)
    add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7)
    cat_3 = torch.ops.aten.cat.default([arg5_1, add_3], cat_3_arg_0)
    cat_4 = torch.ops.aten.cat.default([arg11_1, transpose_3], cat_4_arg_0)
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(cat_3, unsqueeze_5_arg_0)
    expand_4 = torch.ops.aten.expand.default(unsqueeze_5, expand_4_arg_0)
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(cat_4, unsqueeze_6_arg_0)
    expand_5 = torch.ops.aten.expand.default(unsqueeze_6, expand_5_arg_0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(
        -torch.inf, dtype=scalar_tensor_arg_0, device=scalar_tensor_arg_1
    )
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
        scalar_tensor_1_arg_0,
        dtype=scalar_tensor_1_arg_1,
        layout=torch.strided,
        device=scalar_tensor_1_arg_3,
    )
    where = torch.ops.aten.where.self(attention_mask_default, scalar_tensor_1, scalar_tensor)
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
        where, constant_pad_nd_arg_0, constant_pad_nd_arg_1
    )
    slice_19 = torch.ops.aten.slice.Tensor(
        constant_pad_nd, slice_19_arg_0, slice_19_arg_1, slice_19_arg_2
    )
    expand_6 = torch.ops.aten.expand.default(slice_19, expand_6_arg_0)
    reshape_default_34 = torch.ops.aten.reshape.default(expand_4, reshape_default_34_arg_0)
    reshape_default_35 = torch.ops.aten.reshape.default(expand_5, reshape_default_35_arg_0)
    _scaled_dot_product_efficient_attention = (
        torch.ops.aten._scaled_dot_product_efficient_attention.default(
            add_2,
            reshape_default_34,
            reshape_default_35,
            expand_6,
            _scaled_dot_product_efficient_attention_arg_0,
            scale=_scaled_dot_product_efficient_attention_arg_1,
        )
    )
    getitem = _scaled_dot_product_efficient_attention[getitem_arg_0]
    transpose_4 = torch.ops.aten.transpose.int(getitem, transpose_4_arg_0, transpose_4_arg_1)
    reshape_default_53 = torch.ops.aten.reshape.default(transpose_4, reshape_default_53_arg_0)
    return cat_3, cat_4, reshape_default_53


@torch.library.custom_op("tads_ops::attention", mutates_args=(), device_types=["cpu", "cuda"])
@mirror_signature(attention_impl)
def attention(*args, **kwargs):
    return attention_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::attention")
def attention_abstract(*args, **kwargs):
    return attention_impl(*args, **kwargs)


class AttentionBlock(BasePattern):

    pattern_to_match = staticmethod(attention_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        a0 = torch.ops.tads_ops.staging.default(args[0])
        a0.node.meta["mapping"] = "staging_macro"
        a0.node.meta["hints"] = {"batching": "dont_split"}
        a0 = torch.ops.tads_ops.staging.default(a0)
        a0.node.meta["mapping"] = "staging_npu"
        a0.node.meta["hints"] = {"batching": "dont_split"}
        a1 = torch.ops.tads_ops.staging.default(args[1])
        a1.node.meta["mapping"] = "staging_dram"
        a1.node.meta["hints"] = {"batching": "dont_split"}
        a2 = torch.ops.tads_ops.staging.default(args[2])
        a2.node.meta["mapping"] = "staging_dram_pipelined"
        a2.node.meta["hints"] = {"batching": "dont_split"}
        ret = torch.ops.tads_ops.attention.default(a0, a1, a2, *args[3:], **kwargs)
        ret0, ret1, ret2 = ret[0], ret[1], ret[2]
        ret.node.meta["hints"] = {"batching": "dont_split"}
        ret0.node.meta["hints"] = {"batching": "dont_split"}
        ret1.node.meta["hints"] = {"batching": "dont_split"}
        ret2.node.meta["hints"] = {"batching": "dont_split"}
        ret2 = torch.ops.tads_ops.staging.default(ret2)
        ret2.node.meta["mapping"] = "staging_macro"
        ret2.node.meta["hints"] = {"batching": "dont_split"}
        return ret0, ret1, ret2


def attention_2_impl(
    staging_default_1: Tensor,
    arg10_1: Tensor,
    arg17_1: Tensor,
    arg13_1: Tensor,
    arg14_1: Tensor,
    arg15_1: Tensor,
    getitem_22: Tensor,
    getitem_23: Tensor,
    attention_mask_default: Tensor,
    split_with_sizes_default_arg_0: int,
    reshape_default_50_arg_0: List[int],
    transpose_1_arg_0: int,
    transpose_1_arg_1: int,
    reshape_default_51_arg_0: List[int],
    transpose_2_arg_0: int,
    transpose_2_arg_1: int,
    reshape_default_52_arg_0: List[int],
    transpose_3_arg_0: int,
    transpose_3_arg_1: int,
    unsqueeze_3_arg_0: int,
    unsqueeze_4_arg_0: int,
    slice_4_arg_0: int,
    slice_4_arg_1: int,
    slice_4_arg_2: int,
    slice_5_arg_0: int,
    slice_5_arg_1: int,
    slice_5_arg_2: int,
    cat_1_arg_0: int,
    slice_6_arg_0: int,
    slice_6_arg_1: int,
    slice_6_arg_2: int,
    slice_7_arg_0: int,
    slice_7_arg_1: int,
    slice_7_arg_2: int,
    cat_2_arg_0: int,
    cat_3_arg_0: int,
    cat_4_arg_0: int,
    unsqueeze_5_arg_0: int,
    expand_4_arg_0: List[int],
    unsqueeze_6_arg_0: int,
    expand_5_arg_0: List[int],
    scalar_tensor_arg_0: torch.dtype,
    scalar_tensor_arg_1: torch.device,
    scalar_tensor_1_arg_0: float,
    scalar_tensor_1_arg_1: torch.dtype,
    scalar_tensor_1_arg_3: torch.device,
    reshape_default_34_arg_0: List[int],
    reshape_default_35_arg_0: List[int],
    _scaled_dot_product_flash_attention_for_cpu_arg_0: float,
    getitem_arg_0: int,
    transpose_4_arg_0: int,
    transpose_4_arg_1: int,
    reshape_default_53_arg_0: List[int],
) -> Tuple[Tensor, Tensor, Tensor]:
    getattr_1 = arg13_1.shape
    getitem_4 = getattr_1[0]
    getattr_2 = arg14_1.shape
    getitem_5 = getattr_2[0]
    getattr_3 = arg15_1.shape
    getitem_6 = getattr_3[0]
    split_with_sizes_default = torch.ops.aten.split_with_sizes.default(
        staging_default_1, [getitem_4, getitem_5, getitem_6], split_with_sizes_default_arg_0
    )
    getitem_7 = split_with_sizes_default[0]
    getitem_8 = split_with_sizes_default[1]
    getitem_9 = split_with_sizes_default[2]
    reshape_default_50 = torch.ops.aten.reshape.default(getitem_7, reshape_default_50_arg_0)
    transpose_1 = torch.ops.aten.transpose.int(
        reshape_default_50, transpose_1_arg_0, transpose_1_arg_1
    )
    reshape_default_51 = torch.ops.aten.reshape.default(getitem_8, reshape_default_51_arg_0)
    transpose_2 = torch.ops.aten.transpose.int(
        reshape_default_51, transpose_2_arg_0, transpose_2_arg_1
    )
    reshape_default_52 = torch.ops.aten.reshape.default(getitem_9, reshape_default_52_arg_0)
    transpose_3 = torch.ops.aten.transpose.int(
        reshape_default_52, transpose_3_arg_0, transpose_3_arg_1
    )
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_22, unsqueeze_3_arg_0)
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(getitem_23, unsqueeze_4_arg_0)
    mul_125 = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze_3)
    slice_4 = torch.ops.aten.slice.Tensor(transpose_1, slice_4_arg_0, slice_4_arg_1, slice_4_arg_2)
    slice_5 = torch.ops.aten.slice.Tensor(transpose_1, slice_5_arg_0, slice_5_arg_1, slice_5_arg_2)
    neg = torch.ops.aten.neg.default(slice_5)
    cat_1 = torch.ops.aten.cat.default([neg, slice_4], cat_1_arg_0)
    mul_136 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_4)
    add_241 = torch.ops.aten.add.Tensor(mul_125, mul_136)
    mul_141 = torch.ops.aten.mul.Tensor(transpose_2, unsqueeze_3)
    slice_6 = torch.ops.aten.slice.Tensor(transpose_2, slice_6_arg_0, slice_6_arg_1, slice_6_arg_2)
    slice_7 = torch.ops.aten.slice.Tensor(transpose_2, slice_7_arg_0, slice_7_arg_1, slice_7_arg_2)
    neg_1 = torch.ops.aten.neg.default(slice_7)
    cat_2 = torch.ops.aten.cat.default([neg_1, slice_6], cat_2_arg_0)
    mul_152 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_4)
    add_277 = torch.ops.aten.add.Tensor(mul_141, mul_152)
    cat_3 = torch.ops.aten.cat.default([arg10_1, add_277], cat_3_arg_0)
    cat_4 = torch.ops.aten.cat.default([arg17_1, transpose_3], cat_4_arg_0)
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(cat_3, unsqueeze_5_arg_0)
    expand_4 = torch.ops.aten.expand.default(unsqueeze_5, expand_4_arg_0)
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(cat_4, unsqueeze_6_arg_0)
    expand_5 = torch.ops.aten.expand.default(unsqueeze_6, expand_5_arg_0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(
        -torch.inf, dtype=scalar_tensor_arg_0, device=scalar_tensor_arg_1
    )
    scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
        scalar_tensor_1_arg_0,
        dtype=scalar_tensor_1_arg_1,
        layout=torch.strided,
        device=scalar_tensor_1_arg_3,
    )
    where = torch.ops.aten.where.self(attention_mask_default, scalar_tensor_1, scalar_tensor)
    reshape_default_34 = torch.ops.aten.reshape.default(expand_4, reshape_default_34_arg_0)
    reshape_default_35 = torch.ops.aten.reshape.default(expand_5, reshape_default_35_arg_0)
    _scaled_dot_product_flash_attention_for_cpu = (
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            add_241,
            reshape_default_34,
            reshape_default_35,
            attn_mask=where,
            scale=_scaled_dot_product_flash_attention_for_cpu_arg_0,
        )
    )
    getitem = _scaled_dot_product_flash_attention_for_cpu[getitem_arg_0]
    transpose_4 = torch.ops.aten.transpose.int(getitem, transpose_4_arg_0, transpose_4_arg_1)
    reshape_default_53 = torch.ops.aten.reshape.default(transpose_4, reshape_default_53_arg_0)
    return cat_3, cat_4, reshape_default_53


@torch.library.custom_op("tads_ops::attention_2", mutates_args=(), device_types=["cpu", "cuda"])
@mirror_signature(attention_2_impl)
def attention_2(*args, **kwargs):
    return attention_2_impl(*args, **kwargs)


@torch.library.register_fake("tads_ops::attention_2")
def attention_abstract_2(*args, **kwargs):
    return attention_2_impl(*args, **kwargs)


class AttentionBlock2(BasePattern):

    pattern_to_match = staticmethod(attention_2_impl)

    @staticmethod
    def replacement_impl(*args, **kwargs):
        a0 = torch.ops.tads_ops.staging.default(args[0])
        a0.node.meta["mapping"] = "staging_macro"
        a0.node.meta["hints"] = {"batching": "dont_split"}
        a0 = torch.ops.tads_ops.staging.default(a0)
        a0.node.meta["mapping"] = "staging_npu"
        a0.node.meta["hints"] = {"batching": "dont_split"}
        a1 = torch.ops.tads_ops.staging.default(args[1])
        a1.node.meta["mapping"] = "staging_dram"
        a1.node.meta["hints"] = {"batching": "dont_split"}
        a2 = torch.ops.tads_ops.staging.default(args[2])
        a2.node.meta["mapping"] = "staging_dram_pipelined"
        a2.node.meta["hints"] = {"batching": "dont_split"}
        ret = torch.ops.tads_ops.attention_2.default(a0, a1, a2, *args[3:], **kwargs)
        ret0, ret1, ret2 = ret[0], ret[1], ret[2]
        ret.node.meta["hints"] = {"batching": "dont_split"}
        ret0.node.meta["hints"] = {"batching": "dont_split"}
        ret1.node.meta["hints"] = {"batching": "dont_split"}
        ret2.node.meta["hints"] = {"batching": "dont_split"}
        ret2 = torch.ops.tads_ops.staging.default(ret2)
        ret2.node.meta["mapping"] = "staging_macro"
        ret2.node.meta["hints"] = {"batching": "dont_split"}
        return ret0, ret1, ret2


AttentionBlocks = [AttentionBlock, AttentionBlock2]
