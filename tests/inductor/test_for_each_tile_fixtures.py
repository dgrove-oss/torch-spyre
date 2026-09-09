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

"""Minimal for_each_tile fixtures, vendored from torch-spyre#4136.

torch-spyre#4136 (the for_each_tile frontend) is unmerged. These two cases --
a pure map and a pure carry -- are trimmed from that PR's own test suite so
tests/inductor/test_while_loop_lowering.py has a real `while_loop` FX node to
drive through GraphLowering, without depending on the unmerged branch.
"""

import contextlib
from collections.abc import Callable

import torch
from torch._inductor.utils import run_and_get_code

from torch_spyre._inductor.wsr import for_each_tile


M, K, N = 8, 12, 6


def matmul_inputs() -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    X = torch.randn(M, K)
    Y = torch.randn(K, N)
    return (X, Y), X @ Y


def split_m_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case A: tile M as a map. Y is invariant; the result tile lays along dim 0."""

    def body(_, ops):
        x_tile, y_whole = ops
        return None, x_tile @ y_whole

    _, out = for_each_tile(body, (X, Y), dims=(0, None), tile_size=2, out_dim=0)
    return out


def split_k_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Case C: co-indexed split-K matmul; carry accumulates the partial product."""

    def body(acc, ops):
        x_tile, y_tile = ops
        return acc + x_tile @ y_tile, None

    final, _ = for_each_tile(
        body,
        (X, Y),
        dims=(-1, 0),
        tile_size=3,
        init=torch.zeros(M, N, device=X.device, dtype=X.dtype),
    )
    return final


@contextlib.contextmanager
def _post_grad_graphs():
    """Capture each post-grad graph right after decompose_scan_to_while_loop runs.

    post_grad_custom_post_pass fires BEFORE that decomposition, so a custom
    pass cannot see the while_loop node; wrapping the decomposition itself
    can.
    """
    import torch._inductor.fx_passes.post_grad as pg

    seen: list[torch.fx.GraphModule] = []
    original = pg.decompose_scan_to_while_loop

    def wrapper(gm):
        out = original(gm)
        seen.append(gm)
        return out

    pg.decompose_scan_to_while_loop = wrapper
    try:
        yield seen
    finally:
        pg.decompose_scan_to_while_loop = original


def capture_post_grad_while_loop(
    fn: Callable[..., torch.Tensor], args: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.fx.GraphModule]:
    """Compile fn(*args); return (output, the post-grad graph module).

    Asserts the returned graph module contains a real `while_loop` node
    once fully lowered by decompose_scan_to_while_loop -- these fixtures
    exist specifically to drive that node into torch-spyre's
    CustomPreSchedulingPasses.
    """
    torch._dynamo.reset()
    with _post_grad_graphs() as graphs:
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out, _code = run_and_get_code(compiled, *args)
    assert graphs, "no post-grad graph captured (FX graph cache hit?)"
    gm = graphs[-1]
    found = any(
        "while_loop" in str(node.target)
        for node in gm.graph.nodes
        if node.op == "call_function"
    )
    assert found, "expected a while_loop node in the post-grad graph"
    return out, gm
