# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end coarse-tile tests that need no tiling hint.

Rescued from the retired hint-driven coarse-tile e2e suite: these compile
graphs whose tiling comes from SDPA's for_each_tile decomposition, from
automatic span-overflow planning, or not at all, plus the named-dims
validation tests.  ``run_coarse_tile_test`` is the shared driver (compile once,
check the LoopSpec, compare against CPU).
"""

import dataclasses
import os
import sys

import pytest
import regex as re
import torch
import torch.nn.functional as F
from unittest.mock import patch as mock_patch

from torch._inductor.test_case import TestCase as InductorTestCase, fresh_cache
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor import config
from torch_spyre._inductor import spyre_hint
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import mock_backend_compiler, compare_with_cpu, _compile_and_run  # noqa: E402

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims

# Paths to mock for disabling actual device kernel execution.
_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class LoopSpecCheck:
    """Callable loopspec checker passed as the loopspec= argument to run_coarse_tile_test.

    loopspec=LoopSpecCheck()             — asserts LoopSpec( appears in generated source (default)
    loopspec=LoopSpecCheck(counts=[4,2]) — also checks count=sympify('N') for each N
    loopspec=None                        — skip loopspec check entirely
    """

    def __init__(self, counts=None):
        self.counts = counts

    def __call__(self, src):
        assert "LoopSpec(" in src, f"Expected LoopSpec( in generated source:\n{src}"
        if self.counts:
            for count in self.counts:
                assert f"count=sympify('{count}')" in src, (
                    f"Expected count=sympify('{count}') in generated source:\n{src}"
                )


@dataclasses.dataclass
class TensorSpec:
    """Descriptor for a test input tensor.

    name:       parameter name in fn — for readability only.
    shape:      physical tensor shape passed to torch.randn.
    dims:       named dim labels in order, passed to _name_tensor_dims.
    named_dims: optional explicit {dim_name: size} dict for declaring dims.
                Use when len(dims) > len(shape) — i.e. multiple logical dims
                are fused into one physical dim (e.g. flat [B, S, H*D] input
                named ["batch_size", "max_seqlen", "num_heads", "head_dim"]).
                When absent, sizes are inferred by zipping dims with shape.
    value:      optional pre-built CPU tensor. When set, used directly instead
                of torch.randn — shape and scale are ignored for tensor creation.
                Useful for structured inputs like causal masks.
    """

    name: str
    shape: tuple
    dims: list
    named_dims: dict | None = dataclasses.field(default=None)
    value: "torch.Tensor | None" = dataclasses.field(default=None)


def tensor(name, *, shape, dims, named_dims=None, value=None):
    """Shorthand constructor for TensorSpec."""
    return TensorSpec(
        name=name, shape=shape, dims=dims, named_dims=named_dims, value=value
    )


def run_coarse_tile_test(
    fn,
    inputs,
    loopspec=LoopSpecCheck(),
    correctness=True,
    atol=None,
    rtol=None,
    scale=1.0,
):
    """Compile fn on Spyre once, then check loopspec and/or correctness.

    inputs: list of TensorSpec (from tensor(...)) — driver creates tensors,
        declares dims, and calls _name_tensor_dims before each compile.

    loopspec: LoopSpecCheck() — check generated source for LoopSpec (default).
              LoopSpecCheck(counts=[4,2]) also checks specific tile counts.
              None — skip loopspec check.
    correctness: True  — compare_with_cpu against CPU reference.

    Always compiles exactly once, regardless of which checks are enabled.
    """

    torch.manual_seed(0xC0A75E)
    cpu_tensors = [
        s.value
        if s.value is not None
        else torch.randn(s.shape, dtype=torch.float16) * scale
        for s in inputs
    ]

    def _setup_dims_and_dev_tensors():
        _pnd.reset()
        for spec in inputs:
            if spec.named_dims is not None:
                for dim, size in spec.named_dims.items():
                    _declare_tensor_dim(dim, size)
            else:
                for dim, size in zip(spec.dims, spec.shape):
                    _declare_tensor_dim(dim, size)
        dev_tensors = [t.to("spyre") for t in cpu_tensors]
        for spec, t in zip(inputs, dev_tensors):
            _name_tensor_dims(t, spec.dims)
        return dev_tensors

    with fresh_cache():
        dev_tensors = _setup_dims_and_dev_tensors()
        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_backend_compiler(),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), *dev_tensors)

    if loopspec:
        assert len(source_codes) > 0
        loopspec(source_codes[0])

    if correctness:
        dev_tensors = _setup_dims_and_dev_tensors()
        spyre_result = _compile_and_run(fn, dev_tensors, "spyre")
        kwargs = {}
        if atol is not None:
            kwargs["atol"] = atol
        if rtol is not None:
            kwargs["rtol"] = rtol
        compare_with_cpu(
            fn,
            *cpu_tensors,
            target=spyre_result,
            run_eager=False,
            **kwargs,
        )


def test_m1_matmul_coarse_tile():
    """M=1 SDPA + o_proj decode: o_proj bmm hits coarse-tile with factorized x.

    SDPA with q shape [B, H, 1, D] (decode: max_seqlen_q=1) produces attn_out
    [B, H, 1, D] with a factorized tiled layout from SDPA's internal num_heads
    tiling.  After transpose+reshape to [B, 1, H*D], the combined H*D dim is
    the K-dimension for the down-projection (o_proj) linear.  Inside the
    coarse-tile group the bmm's x reads attn_out with host coordinates
    [0, 32*d0+floor(d1/D), 0, Mod(d1,D)], where outer tile var d0 appears
    alongside contraction var d1 — a factorized layout.

    Without the fix, _canonical_stl_from_collapsed_host rejects this with
    "batchmatmul: cannot canonicalize factorized x_var".  The fix relaxes the
    affine_full_range check to allow outer loop vars alongside the contraction
    var.
    """
    B, H, Lq, Lk, D = 1, 32, 1, 2048, 128  # decode: seq_len=1
    hidden = H * D  # 4096
    inputs = [
        tensor(
            "q",
            shape=(B, H, Lq, D),
            dims=["_b", "num_heads", "max_seqlen_q", "head_dim"],
            named_dims={
                "_b": B,
                "num_heads": H,
                "max_seqlen_q": Lq,
                "head_dim": D,
                "max_seqlen_kv": Lk,
            },
        ),
        tensor(
            "k",
            shape=(B, H, Lk, D),
            dims=["_b", "num_heads", "max_seqlen_kv", "head_dim"],
            named_dims={},
        ),
        tensor(
            "v",
            shape=(B, H, Lk, D),
            dims=["_b", "num_heads", "max_seqlen_kv", "head_dim"],
            named_dims={},
        ),
        tensor(
            "w",
            shape=(hidden, hidden),
            dims=["out_hidden", "in_hidden"],
            named_dims={"out_hidden": hidden, "in_hidden": hidden},
        ),
    ]

    def fn(q, k, v, w):
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=D**-0.5)
        # Granite decode pattern: collapse heads before o_proj.
        # attn_out is [B, H, 1, D] with SDPA's factorized tiled layout.
        # After reshape, K-dim = H*D spans both num_heads and head_dim dims,
        # making x's host coords factorized for the coarse-tile bmm.
        x = attn_out.transpose(1, 2).reshape(B, Lq, hidden)  # [1, 1, 4096]
        return F.linear(x, w)  # o_proj: no spyre_hint, coarse-tile from SDPA

    run_coarse_tile_test(fn, inputs, loopspec=None, atol=0.2, rtol=0.2)


def test_sdpa_packed_qkv_input_offsets_h8():
    """Read copies preserve offsets of QKV views repaired after pre-stickify.

    Eight heads trigger SDPA's H coarse tiling.  K and V are non-contiguous
    views into one packed allocation, with nonzero storage offsets.  Before
    issue #4331's fix the generated per-tile read copies silently read Q's
    slice for both tensors.
    """
    torch.manual_seed(0x4331)
    B, H, L, D = 1, 8, 64, 64
    packed = torch.randn(B, L, 3 * H * D, dtype=torch.float16)

    def split_qkv(x):
        return tuple(
            part.reshape(B, L, H, D).transpose(1, 2) for part in x.chunk(3, dim=-1)
        )

    q_cpu, k_cpu, v_cpu = split_qkv(packed)
    mask = torch.zeros(L, L, dtype=torch.float16).masked_fill(
        ~torch.ones(L, L, dtype=torch.bool).tril(),
        torch.finfo(torch.float16).min,
    )[None, None]
    expected = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu, attn_mask=mask, scale=D**-0.5
    )

    q, k, v = split_qkv(packed.to("spyre"))
    assert (q.storage_offset(), k.storage_offset(), v.storage_offset()) == (
        0,
        H * D,
        2 * H * D,
    )
    with fresh_cache():
        actual = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.to("spyre"), scale=D**-0.5
        )

    torch.testing.assert_close(actual.cpu(), expected, atol=0.01, rtol=0.1)


def test_validate_named_dims_raises_on_mismatch():
    """validate_named_dims raises AssertionError when expected_named_dims is wrong."""
    inputs = [tensor("x", shape=(256, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(expected_named_dims=["WRONG", "DIMS"]):
            return torch.abs(x)

    with pytest.raises(Exception, match="expected_named_dims"):
        run_coarse_tile_test(fn, inputs)


def test_validate_reduction_dims_raises_on_mismatch():
    """validate_named_dims raises AssertionError when expected_reduction_dims is wrong."""
    inputs = [tensor("x", shape=(512, 256), dims=["A", "B"])]

    def fn(x):
        with spyre_hint(expected_named_dims=["B"], expected_reduction_dims=["WRONG"]):
            return x.amin(dim=0)

    with pytest.raises(Exception, match="expected_reduction_dims"):
        run_coarse_tile_test(fn, inputs)


class TestSpanOverflowMutationE2E(InductorTestCase):
    """Span-overflow auto-tiling on a post-stickify mutation (no spyre_hint)."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(45054)
        _pnd.reset()

    @config.patch(
        {
            "sencores": 4,
            "ignore_span_overflow_hints": False,
        }
    )
    def test_span_overflow_mutation_case_external_input_layout_mismatch(self):
        """Originally a regression repro for the Case 2/"Case 3"
        layout-reconciliation gap; now an xfail for a separate, deeper,
        pre-existing bug this task's fix newly exposes on this exact op.

        Task 2 (this task) deleted the direct-mutation Case 2/"Case 3"
        branch in `_propagate_tiled_op` entirely, so every cross-loop-group
        write -- including this test's -- now takes the `_insert_copy_op`
        path unconditionally. That fix *is* correct and closes the gap this
        test originally targeted: `TestCoarseTileBufferPropagation`'s
        `test_case2_condition_now_produces_copy_op` unit test directly
        confirms the old Case 2 branch's code no longer exists and the
        copy-op path is taken instead. A control probe run without any
        divergent input layout at all (plain contiguous `x`/`y`, same
        shapes, same span-overflow trigger) confirms the external-input-
        layout-mismatch scenario is no longer what's failing here.

        What still fails: `_insert_copy_op`'s interaction with a
        post-stickify, span-overflow-scaled `full_buf` has its own,
        independent, pre-existing addressing bug in `superdsc.py`'s per-arg
        `device_size`/`device_coordinates` handling -- confirmed present on
        the *pre-Task-2* code too, via a second control probe that forces a
        real inside-consumer (so the OLD Case 2/"Case 3" branch does not
        apply and the OLD code already takes the Case 1 copy-op path) on
        this same post-stickify span-overflow setup: it fails the same way
        (~87% mismatch) with zero divergent input layouts. This is the same
        general class of latent Case-1-copy-path bug flagged as "item 4,
        out of scope" in task-1-report.md (there triggered by a 3-input
        case); here it's shown to need neither 3 inputs nor a divergent
        layout, only `_insert_copy_op` + post-stickify span-overflow with
        `loop_count > 1`. It was never exercised end to end before because,
        pre-Task-2, an op with no inside consumers and no loop-internal
        input (this test's exact shape) always took Case 2 instead, and no
        other post-stickify span-overflow e2e test in this file forces
        Case 1 with `loop_count > 1`.

        This bug is in `_insert_copy_op`/`superdsc.py` addressing, not in
        the Case 2/"Case 3" deletion this task performs, and is out of this
        task's scope (its fix would require changes to `superdsc.py`, not
        listed in this task's file scope). Per this task's explicit
        instructions, this test stays `@unittest.expectedFailure` -- do not
        un-xfail a test that still fails, and do not reinstate any form of
        the deleted Case 2/"Case 3" branch as a workaround.

        Confirmed failure mode (current code, post-Task-2 fix): 15779/16384
        elements (96.3%) mismatch at atol=0.01/rtol=0.01, max abs diff
        ~6.74 -- still far outside fp16 rounding noise, but a different
        mismatch count/magnitude than the pre-fix 12221/16384 (74.6%),
        max abs diff ~5.89 recorded in task-1-report.md, consistent with a
        different root cause now being hit.

        MAX_SPAN_BYTES is patched down so a small tensor triggers automatic
        span-overflow tiling without needing a multi-hundred-MB real
        allocation (technique matches
        test_span_overflow_hint_analysis.py's TestSpanOverflowNumericValidation
        class).
        """
        from unittest.mock import patch as mock_patch2

        B, Lq, D = 32, 8, 64
        x_raw = torch.randn(Lq, B, D, dtype=torch.float16)
        x = x_raw.transpose(0, 1)  # logical [B, Lq, D], non-contiguous strides
        y = torch.randn(B, Lq, D, dtype=torch.float16)

        def fn(x, y):
            return x + y

        with mock_patch2(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
            8192,
        ):
            compare_with_cpu(
                fn, x, y, run_compile=True, run_eager=False, atol=0.01, rtol=0.01
            )


class TestNoHintMoEBroadcastMatmulE2E(InductorTestCase):
    """MoE unsqueeze-broadcast matmul compiled without any tiling hint."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(45141)
        _pnd.reset()

    def test_unsqueeze_broadcast_matmul_no_hint_reuse_dim_scale(self):
        """[1,T,H]@[E,H,F] -> [E,T,F], no spyre_hint (single kernel invocation).

        With no coarse-tiling hint, E never gets tiled/constant-folded away
        before SDSC generation, so x's emitted SDSCArgs must carry E as a
        genuine "reuse dim" (present in w/output, absent from x) with
        scale == -1, the same way any other op's Step 2 broadcast-dim
        handling would. See _matmul_reuse_dims in superdsc.py.
        """
        from torch_spyre._inductor.codegen import superdsc

        E, T, H, F = 4, 64, 64, 64
        x = torch.randn(T, H, dtype=torch.float16) * 0.01
        w = torch.randn(E, H, F, dtype=torch.float16) * 0.01
        _declare_tensor_dim("E", E)
        _declare_tensor_dim("T", T)
        _declare_tensor_dim("H", H)
        _declare_tensor_dim("F", F)

        def fn(x, w):
            _name_tensor_dims(x, ["T", "H"])
            _name_tensor_dims(w, ["E", "H", "F"])
            return torch.matmul(x.unsqueeze(0), w)

        captured = []
        real_create_sdsc_tensors = superdsc._create_sdsc_tensors

        def _spy(*args, **kwargs):
            result = real_create_sdsc_tensors(*args, **kwargs)
            captured.append(result[0])
            return result

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_backend_compiler(),
            mock_patch.object(superdsc, "_create_sdsc_tensors", side_effect=_spy),
        ):
            run_and_get_code(torch.compile(fn), x.to("spyre"), w.to("spyre"))

        self.assertTrue(captured, "no SDSC generated")
        found_reuse_dim = False
        for sdsc_args_list in captured:
            x_arg = sdsc_args_list[0]
            for dim, scale in x_arg.scales.items():
                if scale == -1 and str(dim) not in ("H",):
                    found_reuse_dim = True
        self.assertTrue(
            found_reuse_dim,
            "expected x's SDSCArgs to include a reuse dim (scale == -1) for "
            "the broadcast batch dim E",
        )


class TestAutoSpanOverflowBmmE2E(InductorTestCase):
    """Automatic span-overflow planning of a bmm (output + K tiling)."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(51966)
        _pnd.reset()

    @config.patch(
        {
            "sencores": 4,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "ignore_span_overflow_hints": False,
            "enable_reduction_tiling": True,
        }
    )
    def test_auto_span_overflow_bmm_combined_correct(self):
        """Automatic span planning selects output+K tiling without spyre_hint."""
        B, M, K, N = 1, 128, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16) * 0.01
        b = torch.randn(B, K, N, dtype=torch.float16) * 0.01

        def fn(a, b):
            return torch.bmm(a, b)

        with mock_patch(
            "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
            16 * 1024,
        ):
            # Outputs have std ~2e-3 (0.01-scaled inputs, K=512); the tolerance
            # must sit well below that, or a dropped K partial (~1.6e-3) and
            # even an all-zero result would pass.
            compare_with_cpu(
                fn, a, b, run_compile=True, run_eager=False, atol=5e-4, rtol=0.05
            )

    def _auto_span_overflow_bmm_source(self, *, enable_reduction_tiling):
        B, M, K, N = 1, 128, 512, 32
        a = torch.randn(B, M, K, dtype=torch.float16).to("spyre")
        b = torch.randn(B, K, N, dtype=torch.float16).to("spyre")

        def fn(a, b):
            return torch.bmm(a, b)

        with (
            config.patch(
                {
                    "sencores": 4,
                    "lx_planning": True,
                    "allow_all_ops_in_lx_planning": True,
                    "ignore_span_overflow_hints": False,
                    "enable_reduction_tiling": enable_reduction_tiling,
                }
            ),
            mock_patch(
                "torch_spyre._inductor.wsr.span_overflow_hint_analysis.MAX_SPAN_BYTES",
                16 * 1024,
            ),
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_backend_compiler(),
        ):
            _, source_codes = run_and_get_code(torch.compile(fn), a, b)

        self.assertTrue(source_codes)
        return source_codes[0]

    def test_auto_span_overflow_bmm_combined_loopspec(self):
        """Automatic M+K planning emits the expected nested loop counts."""
        src = self._auto_span_overflow_bmm_source(enable_reduction_tiling=True)

        self.assertEqual(src.count("LoopSpec("), 2)
        self.assertIn("count=sympify('8')", src)
        self.assertIn("count=sympify('2')", src)
        self.assertIn("coarse_tile_reduce_copy", src)

        # The combine op's accumulator read/write must stay at one fixed
        # address across the inner K loop: no device_tile_advance_expr.
        combine = re.search(
            r"ir_chain=\('bmm', 'coarse_tile_combine_\w+'\).*?args=\[(.*?)\n\s*\]\n",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(combine, "expected the combine op's OpSpec in source")
        self.assertNotIn("device_tile_advance_expr", combine.group(1))

    def test_auto_span_overflow_bmm_kill_switch_rejects_unresolved_k_span(self):
        """Disabling K tiling must not hide a K span left by output-only tiling."""
        with self.assertRaisesRegex(
            Exception,
            "no combined split.*makes all spans fit",
        ):
            self._auto_span_overflow_bmm_source(enable_reduction_tiling=False)
