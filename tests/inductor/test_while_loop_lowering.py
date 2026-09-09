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

"""End-to-end Spyre-device tests for WhileLoop -> OpSpec/LoopSpec lowering.

Minimum coverage per docs/superpowers/specs/2026-09-09-while-loop-lowering-design.md:
1. Single carry (this file: test_carry_mode_split_k) -- currently XFAIL on a
   read-copy/stick-layout gap; see that test's own docstring.
2 (carry + Kind.SLICE tile-advancing input): covered implicitly by
   test_carry_mode_split_k, whose X/Y operands are both Kind.SLICE.
Cases 3-6 (Kind.GATHER, multiple carries, nested for_each_tile, and the
deliberate-decline case) are follow-on work once cases 1/2 are green --
tracked as open items rather than duplicated here, since each needs its own
fixture beyond what Task 1 vendored.

test_map_mode_split_m (map mode: Kind.SLICE + Kind.INVARIANT operands, a
stacking carry, no user carry) passes end to end with verified numerics and
is the case that exercises the full splice -> DimHint synthesis ->
coarse-tile -> single scf.for pipeline.
"""

import unittest

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre.constants import DEVICE_NAME

from tests.inductor.test_for_each_tile_fixtures import (
    matmul_inputs,
    split_k_fn,
    split_m_fn,
)


class TestWhileLoopLowering(unittest.TestCase):
    # Spyre's matmul runs in fp16, so the reference has to be an fp16-faithful
    # one: cast the operands first, then accumulate in fp32 on CPU. Comparing
    # against the fp32 product of fp32 operands would fail on rounding alone,
    # independently of anything this test is meant to check.
    #
    # Operands must also be cast to fp16 BEFORE the host->device transfer, not
    # after: `t.to(DEVICE_NAME).half()` (transfer fp32, cast on device)
    # currently produces garbage on this backend for reasons unrelated to
    # while_loop lowering -- a plain `torch.compile`d `a @ b` reproduces it
    # without any for_each_tile involved. `t.half().to(DEVICE_NAME)` is the
    # idiom the rest of the compiled-op suite uses (see
    # tests/inductor/test_inductor_matmul.py, whose inputs are constructed
    # `dtype=torch.float16` up front).
    ATOL = 0.1
    RTOL = 0.1

    @staticmethod
    def _operands():
        (X, Y), _ = matmul_inputs()
        ref = (X.half().float()) @ (Y.half().float())
        return X.half().to(DEVICE_NAME), Y.half().to(DEVICE_NAME), ref

    def test_map_mode_split_m(self):
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_m_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    @unittest.expectedFailure
    def test_carry_mode_split_k(self):
        """Carry mode: accumulate a split-K matmul across tiles.

        XFAIL on a gap in read-copy layout reconciliation, downstream of and
        distinct from everything this test's own lowering path needs -- the
        accumulator carry itself is wired correctly and WSR's own tiled-
        reduction accumulator (coarse_tile_fill/combine on the K level) picks
        the K accumulation up as intended.

        The gap: ``for_each_tile``'s ``xs`` leaves for ``dims=(-1, 0)`` are
        3-D, transposed, ``movedim``-derived views of the operands
        (``[4, 3, 8]`` stride ``[3, 1, 12]`` for X, ``[4, 3, 6]`` stride
        ``[18, 6, 1]`` for Y). The K-advancing reads of those leaves route
        through ``coarse_tile.py``'s read-copy machinery, which builds tile
        buffers whose own layouts (e.g. ``[8, 6, 3]`` stride ``[0, 1, 6]`` --
        a broadcast leading dim over transposed inner dims) then fail stick
        reconciliation in ``optimize_restickify.py``/``propagate_layouts.py``
        ("No mechanism to scatter elements from one stick to multiple
        sticks"). The equivalent HINT-driven K-tiled matmul (same M/K/N,
        ``spyre_hint(num_tiles_per_dim={"K": 4})``) compiles and is
        numerically correct, and needs no read copies at all -- it reads the
        2-D operands directly. So this is a read-copy/stick-layout gap
        surfaced by the 3-D stacked-leaf shape, not a while_loop-lowering
        one, and it needs the same kind of layout work that the map-mode
        carry's own ``[4, 2, 6] -> [8, 6]`` fold needed (see
        ``while_loop_bridge.fold_stacked_carry_layout``) applied to the
        read side.
        """
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_k_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )


if __name__ == "__main__":
    unittest.main()
