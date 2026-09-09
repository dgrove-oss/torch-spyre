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
    from torch._inductor.graph import GraphLowering


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


def _transplant_buffer_registrations(
    graph: "GraphLowering",
    body_graph: "GraphLowering",
    body_ops: list["ir.Operation"],
) -> None:
    """Register body_ops' own output buffers into graph, not just body_graph.

    Every ComputedBuffer/Buffer an op produces self-registers into whichever
    GraphLowering instance is V.graph at construction time (see
    GraphLowering.register_buffer/register_operation) -- for a while_loop
    body that is the *inner* SubgraphLowering built for the body subgraph,
    never the outer graph this bridge splices ops into. SubgraphLowering
    does not delegate lookups to its .parent, so leaving this unpatched
    means every downstream V.graph.get_buffer(name)/get_operation(name) call
    against the spliced ops (e.g. coarse_tile.py's read-copy planning) fails
    with "Failed to find buffer/operation matching name ...", confirmed
    empirically: splicing alone (without this) raises exactly that
    RuntimeError from _full_buffer_read_deps the first time
    coarse_tile_pre_stickify inspects a spliced op's reads.

    Only copies the dict/list entries this bridge and its callers are known
    to read (name_to_buffer/buffers/name_to_op) -- not a general graph
    merge.
    """
    for op in body_ops:
        op_name = getattr(op, "get_operation_name", lambda: None)()
        if op_name is not None and op_name in body_graph.name_to_op:
            graph.name_to_op[op_name] = op
        for buf in op.get_outputs() if hasattr(op, "get_outputs") else ():
            buf_name = buf.get_name()
            if buf_name in body_graph.name_to_buffer:
                graph.name_to_buffer[buf_name] = buf
                if buf not in graph.buffers:
                    graph.buffers.append(buf)


def _substitute_direct_input_refs(
    body_ops: list["ir.Operation"],
    ref_map: dict[str, Any],
) -> None:
    """Rewrite direct object references to a spliced-away placeholder.

    Ops without an `inner_fn` (DynamicScalar, ExternKernelOut, ...) hold
    their reads as direct Python object references in `.inputs`, not as
    named index-expression loads -- confirmed empirically for both
    split_m_fn and split_k_fn: DynamicScalar.inputs[0] is always the
    body's own iteration-carry placeholder object, and ExternKernelOut's
    per-tile operand input is either the placeholder object itself or a
    frozen ReinterpretView/mutable StorageBox wrapping it. Renaming a dict
    entry (as redirect_computed_buffer_reads does for inner_fn loads) does
    nothing for these -- the object reference itself must be replaced.

    ref_map maps a placeholder buffer's own name to the real object it
    should resolve to (a carry's scratch buffer, or the real outer-graph
    value for a non-carry per-tile operand). Mutates each op's `.inputs`
    list in place (`ExternKernel.inputs` is an ordinary mutable list) and,
    for the one observed wrapped case (StorageBox, itself mutable) rewrites
    the wrapper's `.data` in place rather than fighting ReinterpretView's
    frozen dataclass -- ReinterpretView.data is looked up dynamically by
    every consumer, so the wrapped StorageBox's identity does not need to
    change, only what it points at.
    """
    from torch._inductor import ir

    def resolve(node):
        name = getattr(node, "get_name", lambda: None)()
        return ref_map.get(name)

    for op in body_ops:
        inputs = getattr(op, "inputs", None)
        if not inputs:
            continue
        for i, inp in enumerate(inputs):
            replacement = resolve(inp)
            if replacement is not None:
                inputs[i] = replacement
                continue
            # One level of unwrapping: StorageBox is mutable, so patch its
            # .data in place; other wrapper kinds are left alone (none seen
            # in practice -- see docstring).
            base = getattr(inp, "data", None)
            if isinstance(base, ir.StorageBox):
                inner_replacement = resolve(base.data)
                if inner_replacement is not None:
                    base.data = inner_replacement


def splice_while_loop(
    graph: "GraphLowering",
    while_op: "ir.WhileLoop",
    carries: list[CarryBinding],
) -> list["ir.Operation"]:
    """Replace while_op in graph.operations with its body subgraph's ops.

    Carry rewiring (fill/rewrite/drain): each CarryBinding's body_output is
    redirected, via redirect_computed_buffer_reads, so any op inside the
    spliced body that wrote the body subgraph's own carry placeholder now
    writes carries[i].scratch_name instead -- the persistent buffer a future
    fill/drain step will seed and read across iterations. No such fill step
    exists yet anywhere in this codebase (carry_bindings_for only mints the
    scratch_name identity; nothing materializes a real buffer under it) --
    confirmed empirically, so redirecting a *read* to scratch_name today
    would just trade one nonexistent-buffer failure for another.

    So the read side is handled per carry shape, distinguishing two cases by
    identity (confirmed against both split_m_fn and split_k_fn):

    - Pass-through carry: body_output IS the placeholder object itself
      (the body never rewrites this carry -- e.g. a per-tile xs leaf
      threaded through unmodified). Its read is aliased straight to the
      real while_op.carried_inputs[i] object, already live in the outer
      graph -- correct and available today, no scratch buffer needed.
    - Mutated carry: body_output is a different, real op-produced buffer
      (the body computes a new value each iteration -- e.g. an
      accumulator). Splicing the body in once (as this bridge does; the
      loop structure itself is discarded, with iteration folded into
      DimHints/levels by the caller) means this single copy of the body
      reads the carry's pre-loop initial value, exactly as the first real
      iteration would -- so its read is aliased to
      while_op.carried_inputs[i] too, same as the pass-through case. Only
      the *write* side (body_output) is redirected to scratch_name, per
      the existing contract above, so a later fill/drain task has a stable
      name to build on.

    Two distinct read shapes exist in the body and both need rewiring:
    ComputedBuffer.inner_fn issues ops.load(name, index) calls (name-based
    -- handled by redirect_computed_buffer_reads/NameSwapHandler, per
    CLAUDE.md's "wrap, never reconstruct" rule); DynamicScalar/
    ExternKernelOut hold direct Python object references to the
    placeholder buffer in `.inputs` (object-based -- handled by
    _substitute_direct_input_refs, since no name_map rename can reach a
    held object reference). Confirmed empirically (both fixtures): every
    mutated-carry placeholder is read exclusively via inner_fn/ops.load;
    direct .inputs references only ever target pass-through carries (and,
    for split_k_fn's counter carry, both shapes read the same placeholder
    simultaneously) -- so both maps are populated for every carry
    regardless of shape, rather than assuming shape predicts read kind.

    Returns the spliced body ops (graph.operations, still in topological
    order) so the caller can build a coarse-tile (ops, levels) group from
    them.
    """
    from torch_spyre._inductor.pass_utils import redirect_computed_buffer_reads

    body_ops = list(while_op.body_subgraph.graph.operations)
    body_graph_input_names = list(while_op.body_subgraph.graph.graph_inputs.keys())

    name_map: dict[str, str] = {}
    ref_map: dict[str, Any] = {}

    for binding in carries:
        placeholder_name = body_graph_input_names[binding.carry_index]
        real_input = while_op.carried_inputs[binding.carry_index]
        real_name = real_input.get_name()

        body_output_name = getattr(binding.body_output, "get_name", lambda: None)()
        is_passthrough = body_output_name == placeholder_name
        if body_output_name is not None and not is_passthrough:
            # Real per-iteration rewrite: redirect the write side to the
            # persistent scratch identity a future fill/drain step will own.
            name_map[body_output_name] = binding.scratch_name

        # Read side always resolves to the real, already-registered initial
        # value -- see docstring above for why this holds for both
        # pass-through and mutated carries at this stage of the pipeline.
        name_map[placeholder_name] = real_name
        ref_map[placeholder_name] = real_input

    for i in range(len(carries), len(body_graph_input_names)):
        placeholder_name = body_graph_input_names[i]
        real_input = while_op.inputs[i]
        real_name = real_input.get_name()
        name_map[placeholder_name] = real_name
        ref_map[placeholder_name] = real_input

    if name_map:
        body_ops = [
            redirect_computed_buffer_reads(
                op,
                name_map,
                body_ops,
                pass_name="splice_while_loops",
                reason="redirect while_loop carry/tile reads to persistent scratch",
            )
            if hasattr(op, "data")
            else op
            for op in body_ops
        ]

    if ref_map:
        _substitute_direct_input_refs(body_ops, ref_map)

    _transplant_buffer_registrations(graph, while_op.body_subgraph.graph, body_ops)

    idx = graph.operations.index(while_op)
    graph.operations[idx : idx + 1] = body_ops

    # Drop this while_op's MultiOutput children -- they read while_op's own
    # (now-removed) buffer positionally; the caller redirects any real
    # outside consumer to the relevant carry's scratch_name via the same
    # name_map before/while removing them.
    #
    # Verified against a live compiled graph (split_k_fn, which has a real
    # carry): each such child is an ir.MultiOutput ExternKernel whose own
    # `.inputs[0] is while_op` -- that is the real linkage, confirmed by
    # object identity, not by any `_while_loop_parent` attribute (no such
    # attribute exists anywhere on the real IR; the placeholder predicate
    # this replaced was therefore a no-op that removed nothing). MutationOutput
    # is a Buffer, not an Operation, and so can never appear in
    # graph.operations at all -- only MultiOutput needs handling here.
    graph.operations = [
        op
        for op in graph.operations
        if op is while_op or while_op not in (getattr(op, "inputs", None) or ())
    ]

    return body_ops
