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

"""Generic, for_each_tile-independent bridge from a WhileLoop op to a
coarse-tile-shaped group of spliced ir.Operations.

Deliberately knows nothing about for_each_tile's own frontend contract
(Kind.SLICE/GATHER/INVARIANT, TileSpec, etc.) -- that classification lives in
wsr/for_each_tile_lowering.py, which is this module's only caller. Anything
here should stay reusable by a future, differently-shaped while_loop
producer.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch._inductor import ir


@dataclasses.dataclass(frozen=True)
class CarryBinding:
    """One WhileLoop carry: initial value, per-iteration scratch identity, final consumers.

    Attributes
    ----------
    carry_index:
        Position in ``while_op.carried_inputs`` / the body subgraph's
        ``graph_outputs`` -- carries are matched purely positionally on the
        raw IR (see ``ir.py``'s ``WhileLoop.create``, which asserts
        ``len(carried_inputs) == len(body_outputs)``).
    initial:
        ``while_op.carried_inputs[carry_index]`` -- the pre-loop value that
        seeds the carry's scratch buffer (the "fill" step).
    body_output:
        ``while_op.body_subgraph.graph.graph_outputs[carry_index]`` -- the
        body's own per-iteration result for this carry position (what the
        "rewrite" step redirects the body's write to, and what "drain"
        reads after the last iteration).
    scratch_name:
        Persistent buffer identity threaded through every iteration.
    """

    carry_index: int
    initial: Any
    body_output: Any
    scratch_name: str


def carry_bindings_for(while_op: "ir.WhileLoop") -> list[CarryBinding]:
    """Build one CarryBinding per position in while_op.carried_inputs."""
    carried_inputs = while_op.carried_inputs
    body_outputs = while_op.body_subgraph.graph.graph_outputs
    return [
        CarryBinding(
            carry_index=i,
            initial=initial,
            body_output=body_outputs[i],
            scratch_name=f"while_carry_{id(while_op)}_{i}",
        )
        for i, initial in enumerate(carried_inputs)
    ]
