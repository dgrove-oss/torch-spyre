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

"""for_each_tile-specific WhileLoop prover.

Recognizes the exact WhileLoop shape torch-spyre#4136's for_each_tile
frontend (via decompose_scan_to_while_loop) produces, derives a provable
trip count, and -- once accepted -- hands off to the generic bridge
(while_loop_bridge.py) plus DimHint synthesis to actually splice and
coarse-tile the body. This module owns every for_each_tile-specific
assumption; while_loop_bridge.py knows none of them.

Real cond-graph shape (confirmed empirically against a live compiled graph
for both split_m_fn (map mode) and split_k_fn (carry mode) -- see the task-4
report for the full investigation): decompose_scan_to_while_loop always
lowers for_each_tile's cond_fn to a cond_subgraph.graph with exactly one
ir.Operation -- a scalar (size=[]) bool ComputedBuffer -- whose inner_fn
does exactly:

    tmp0 = ops.load(<cond graph's own first placeholder>, 0)
    tmp1 = ops.constant(N, torch.int64)
    tmp2 = tmp0 < tmp1
    return tmp2

i.e. `lt(iteration_sym, N)` with N a plain Python int/constant baked in by
the tracer (for_each_tile.py's `_step_counter`/`count_mode` logic always
carries the trip counter as carried_inputs[0], and the cond subgraph's own
first placeholder is that same carry positionally). `N` is not exposed as a
separate symbolic node anywhere reachable from the IR level -- it only shows
up as the literal second operand of the `<` -- so rather than parse
inner_fn's closure cells (an internal, unstable implementation detail of
torch._inductor.ir.make_pointwise/ops_wrapper), this module *runs* inner_fn
once under a small recording ops handler that intercepts `load`/`constant`
and returns opaque placeholders for everything else. This is the same "wrap
the ops handler, don't reconstruct index expressions" pattern CLAUDE.md
mandates for ComputedBuffer.inner_fn elsewhere in this codebase, applied
here for read-only shape recognition rather than mutation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import sympy
import torch

from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._inductor import ir


@dataclasses.dataclass(frozen=True)
class ProverResult:
    """Outcome of trying to recognize a WhileLoop as a for_each_tile loop."""

    accepted: bool
    trip_count: sympy.Expr | None = None
    reason: str = ""


class _CondInnerFnRecorder(DefaultHandler):
    """Records the loads/constants/comparison op a cond inner_fn issues.

    Every other ops call (there should be none for the shape this prover
    recognizes) is routed through `_default` and answered with an opaque
    placeholder string so `inner_fn` can run to completion without needing a
    real kernel-codegen context.
    """

    def __init__(self) -> None:
        self.loads: list[tuple[str, Any]] = []
        self.constants: list[Any] = []
        self.compare_ops: list[str] = []

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name == "load":
            self.loads.append((args[0], args[1]))
            return f"__load_{len(self.loads) - 1}__"
        if name == "constant":
            self.constants.append(args[0])
            return f"__constant_{len(self.constants) - 1}__"
        if name in ("lt", "le", "gt", "ge", "eq", "ne"):
            self.compare_ops.append(name)
            return f"__cmp_{name}__"
        # Anything else means this cond graph does not match the known
        # for_each_tile shape (a single load-vs-constant comparison); record
        # the op name so the caller can decline with a useful reason.
        self.compare_ops.append(f"unexpected:{name}")
        return f"__unexpected_{name}__"


def _first_placeholder_name(cond_graph) -> str | None:
    """The cond subgraph's own first graph input -- the iteration carry."""
    graph_inputs = getattr(cond_graph, "graph_inputs", None)
    if not graph_inputs:
        return None
    return next(iter(graph_inputs), None)


def _extract_trip_count(cond_graph) -> sympy.Expr | None:
    """Find the single lt(iteration_sym, N)-shaped comparison cond_graph computes.

    for_each_tile's cond_fn (after decompose_scan_to_while_loop) reduces to
    exactly one boolean scalar ComputedBuffer computing
    `ops.load(<first placeholder>, 0) < ops.constant(N, ...)`. Returns N as a
    sympy.Expr, or None if the shape does not match.
    """
    graph_outputs = getattr(cond_graph, "graph_outputs", None)
    if not graph_outputs or len(graph_outputs) != 1:
        return None
    operations = getattr(cond_graph, "operations", None)
    if not operations or len(operations) != 1:
        return None

    op = operations[0]
    data = getattr(op, "data", None)
    inner_fn = getattr(data, "inner_fn", None)
    if inner_fn is None:
        return None
    # The comparison is a scalar bool -- no output ranges to index over.
    get_size = getattr(data, "get_size", None)
    if get_size is None or list(get_size()) != []:
        return None
    if getattr(data, "dtype", None) != torch.bool:
        return None

    first_placeholder = _first_placeholder_name(cond_graph)
    if first_placeholder is None:
        return None

    recorder = _CondInnerFnRecorder()
    with V.set_ops_handler(recorder):
        inner_fn(())

    if recorder.compare_ops != ["lt"]:
        return None
    if len(recorder.loads) != 1 or len(recorder.constants) != 1:
        return None

    (loaded_name, loaded_index) = recorder.loads[0]
    if loaded_name != first_placeholder:
        return None
    if loaded_index != 0:
        return None

    bound = recorder.constants[0]
    if isinstance(bound, bool):
        return None
    if not isinstance(bound, (int, sympy.Expr)):
        return None
    return sympy.sympify(bound)


def try_prove_for_each_tile(while_op: "ir.WhileLoop") -> ProverResult:
    """Decide whether while_op matches for_each_tile's known WhileLoop shape."""
    cond_subgraph = getattr(while_op, "cond_subgraph", None)
    cond_graph = getattr(cond_subgraph, "graph", None) if cond_subgraph else None
    if cond_graph is None:
        return ProverResult(accepted=False, reason="no cond_subgraph.graph to inspect")

    trip_count = _extract_trip_count(cond_graph)
    if trip_count is None:
        return ProverResult(
            accepted=False,
            reason=(
                "cond_subgraph did not reduce to a single provable "
                "lt(iteration_sym, N) comparison"
            ),
        )
    return ProverResult(accepted=True, trip_count=trip_count)
