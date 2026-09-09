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
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering

logger = logging.getLogger(__name__)


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
    stacking:
        Whether this carry is a STACKING carry rather than an ACCUMULATOR.
        The raw IR models the two identically -- both are just a position in
        ``carried_inputs``/``body_outputs`` -- but they need opposite
        treatment, and only the caller (which understands the producing
        frontend's contract) can tell them apart:

        - **Accumulator** (``stacking=False``, the default): the same
          logical value, same shape, updated in place every iteration and
          read once at the end. This is the pattern the existing
          fill/rewrite/drain ``scratch_name`` redirect is built for, and it
          is left completely untouched by the stacking machinery below.
        - **Stacking** (``stacking=True``): each iteration writes a
          DIFFERENT SLICE of one larger result -- ``scan``'s ``ys``
          accumulation, which upstream materializes as a
          ``[trip_count, *tile]`` buffer that the caller then folds back to
          ``[trip_count * tile[0], *tile[1:]]``. There is no intermediate
          per-iteration state to thread, so it needs no scratch buffer at
          all; what it needs instead is for its own destination buffer's
          layout to BE the folded shape, so the per-iteration write is an
          ordinary tile-advancing write into it. See
          ``fold_stacked_carry_layout``.
    """

    carry_index: int
    initial: Any
    body_output: Any
    scratch_name: str
    stacking: bool = False


def carry_bindings_for(
    while_op: "ir.WhileLoop",
    stacking_indices: "frozenset[int] | None" = None,
) -> list[CarryBinding]:
    """Build one CarryBinding per position in while_op.carried_inputs.

    ``stacking_indices`` names the carry positions the caller has classified
    as stacking rather than accumulator carries (see ``CarryBinding``'s
    ``stacking`` docstring); omitting it treats every carry as an
    accumulator, the pre-existing behaviour.
    """
    stacking_indices = stacking_indices or frozenset()
    carried_inputs = while_op.carried_inputs
    body_outputs = while_op.body_subgraph.graph.graph_outputs
    return [
        CarryBinding(
            carry_index=i,
            initial=initial,
            body_output=body_outputs[i],
            scratch_name=f"while_carry_{id(while_op)}_{i}",
            stacking=i in stacking_indices,
        )
        for i, initial in enumerate(carried_inputs)
    ]


def fold_stacked_carry_layout(node: Any, trip_count: Any) -> bool:
    """Collapse a stacking carry's buffer from [trip, *tile] to the folded shape.

    Upstream's ``scan`` lowering allocates a stacking carry (see
    ``CarryBinding.stacking``) as a genuinely rank-(n+1)
    ``[trip_count, *tile]`` "stack of tiles" buffer, and returns the real
    result as a ``ReinterpretView`` of it with the leading two axes merged --
    a pure view, since merging two contiguous leading axes is expressible in
    strides.

    This bridge's splice discards the loop structure and folds iteration
    into DimHints, so the per-iteration write becomes an ordinary
    tile-advancing write into one flat destination. That destination's own
    layout must therefore be the FOLDED shape: leaving the rank-(n+1)
    stacked layout in place is what produced the
    ``AssertionError: size=[8, 6], stride=[12, 6, 1]`` in
    ``coarse_tile.py``'s ``_allocate_full_buffer`` -- the write's planned
    full ranges are 2-D while the buffer it targets is still 3-D.

    Rewrites ``node``'s layout in place to
    ``[trip_count * tile[0], *tile[1:]]`` with the corresponding contiguous
    strides, and drops the matching leading axis from its ``data.ranges``
    when it has one. The two are the same storage with the same element
    order, so nothing about what the buffer holds changes -- only how many
    axes name it.

    Mutating in place (rather than substituting a fresh buffer object
    everywhere, as ``_substitute_direct_input_refs`` must do for a
    placeholder) is safe because a ``Layout`` object belongs to exactly one
    buffer: verified for this shape that the carry buffer, the body's own
    placeholder, and the graph output's ``ReinterpretView`` each hold a
    distinct ``FixedLayout`` instance. The graph output's view is unaffected
    -- it already describes the folded shape, so after this it is an exact
    identity view of the buffer instead of a reshape of it.

    Returns True if a fold was applied, False if the layout was not a
    foldable stacked shape (already folded, non-contiguous leading axes, or
    a symbolic trip count that does not match the leading extent). Declining
    is not an error: a caller that mis-classifies a carry as stacking simply
    gets the pre-existing behaviour rather than a corrupted layout.
    """
    import sympy
    from torch._inductor import ir
    from torch._inductor.ir import FixedLayout

    while isinstance(node, ir.MutableBox):
        node = node.data
    layout = getattr(node, "layout", None)
    if not isinstance(layout, FixedLayout):
        return False

    size = list(layout.size)
    stride = list(layout.stride)
    if len(size) < 2:
        return False
    # The stacked axis must be the leading one, with exactly the trip count
    # as its extent, and the fold must be a pure reshape: axis 0's stride has
    # to be exactly axis 1's extent times axis 1's stride, or the merged axis
    # cannot be described by a single stride.
    if sympy.simplify(sympy.sympify(size[0]) - sympy.sympify(trip_count)) != 0:
        return False
    if sympy.simplify(stride[0] - size[1] * stride[1]) != 0:
        return False

    new_size = [size[0] * size[1], *size[2:]]
    new_stride = [stride[1], *stride[2:]]
    node.layout = FixedLayout(
        layout.device, layout.dtype, new_size, new_stride, layout.offset
    )

    data: Any = getattr(node, "data", None)
    ranges = getattr(data, "ranges", None)
    if data is not None and ranges is not None and len(ranges) == len(size):
        # An `aten.empty_strided`-origin allocation has all-zero ranges (it
        # iterates over nothing); a real fill has the buffer's own extents.
        # Merging the two leading entries is correct either way, and keeps
        # data.ranges' rank in step with the layout's -- Inductor's
        # `indexer` asserts the two agree when it extracts this buffer's
        # own write MemoryDep.
        try:
            node.data = dataclasses.replace(
                data, ranges=[ranges[0] * ranges[1], *ranges[2:]]
            )
        except TypeError:
            # Not a dataclass with a `ranges` field we can rebuild; leave it
            # (and the layout fold above) rather than half-applying.
            node.layout = layout
            return False
    return True


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

    A third read shape exists alongside the two documented above: a
    ComputedBuffer with a MutationLayoutSHOULDREMOVE layout holds its
    mutation target as a direct object reference on `op.layout.target`
    (set once in Layout.__init__ and never renamed) -- not in `.inputs`,
    not a named ops.load. Confirmed via test_map_mode_split_m
    (issue #3965): a body-internal constant_pad_nd/copy pair targets the
    body's own per-tile-invariant operand placeholder
    (while_loop_body_graph_0_0_arg1_1, Y's map-mode carry), which after
    splicing is never registered in the outer graph's buffer namespace --
    V.graph.get_buffer(target_name) in propagate_layouts.py's mutation-op
    handling then raises "Failed to find buffer matching name ...". Patch
    op.layout.target in place the same way as an `.inputs` entry (one level
    of StorageBox unwrapping included) so it resolves to the real
    outer-graph object splice_while_loop already computed, exactly as
    `.inputs` entries do.

    ref_map maps a placeholder buffer's own name to the real object it
    should resolve to (a carry's scratch buffer, or the real outer-graph
    value for a non-carry per-tile operand). Mutates each op's `.inputs`
    list in place (`ExternKernel.inputs` is an ordinary mutable list) and,
    for the one observed wrapped case (StorageBox, itself mutable) rewrites
    the wrapper's `.data` in place rather than fighting ReinterpretView's
    frozen dataclass -- ReinterpretView.data is looked up dynamically by
    every consumer, so the wrapped StorageBox's identity does not need to
    change, only what it points at.

    Whole-object substitution (rebinding a list slot or `.layout.target`
    directly to the resolved replacement) is only correct when the node
    being replaced is a bare identity wrapper for the placeholder -- i.e. it
    contributes no reslicing/offset/layout information of its own beyond
    forwarding to the placeholder. `ReinterpretView` is NOT such a wrapper:
    `ReinterpretView.get_name()` delegates to `self.data.get_name()` (see
    ir.py), so `resolve()` happily resolves a `ReinterpretView` by the name
    of the placeholder it wraps -- but the `ReinterpretView` itself carries
    its own real `FixedLayout`/offset describing a *slice* of that
    placeholder (e.g. one per-iteration tile of an invariant operand).
    Substituting the whole `ReinterpretView` object away, as a naive
    `resolve(node) is not None` check would do, silently discards that
    slice/offset and hands consumers the placeholder's raw, untiled layout
    instead -- confirmed empirically via test_map_mode_split_m: doing so
    produces a MutationLayoutSHOULDREMOVE target whose real_layout() delivers
    the untiled buffer's layout while the mutation op's own store still
    computes indices against the tile's small `[2, 6]` iteration domain,
    which resolves handles fine (real_layout()/stride still work -- the
    layout is well-formed) but is simply the wrong slice; the *actual*
    crash this produces downstream is even more direct: some other
    call reads `layout.target`'s stride expecting the tile's own FixedLayout
    and instead unwraps all the way down to the InputBuffer, whose
    FixedLayout's stride/size mismatches what `_fixed_indexer` was closed
    over. So: only replace whole-node when the node has no distinguishing
    layout of its own (a bare StorageBox/InputBuffer/TensorBox pass-through);
    for a ReinterpretView (or any node whose own `.data` is a StorageBox
    still pointing at the placeholder), unwrap one level and patch the
    inner StorageBox's `.data` in place instead, leaving the ReinterpretView
    node itself (and its own layout/offset) completely untouched.
    """
    from torch._inductor import ir

    def resolve(node):
        name = getattr(node, "get_name", lambda: None)()
        return ref_map.get(name)

    def substitute(node, setter):
        """Rewire one reference to node to the real object in ref_map.

        Returns True if a substitution was made. Only rebinds the whole
        node via `setter` when node is a bare pass-through wrapper (no
        layout/offset of its own to lose); a ReinterpretView (or anything
        else wrapping a StorageBox that itself resolves) is instead patched
        one level down, in place, preserving the outer node's identity and
        its own layout/offset.
        """
        base = getattr(node, "data", None)
        if isinstance(base, ir.StorageBox):
            inner_replacement = resolve(base.data)
            if inner_replacement is not None:
                # inner_replacement resolves from ref_map, whose values are
                # whatever object shape while_op.carried_inputs/inputs held
                # (confirmed empirically: a TensorBox(StorageBox(...))
                # MutableBox, same as any other real graph value) -- but
                # StorageBox.data must hold the innermost real node (a
                # Buffer/View/Loops), never another MutableBox, or
                # consumers that expect exactly one level of box
                # (unwrap_views's own MutableBox arm recurses fine, but
                # make_indexer()/get_layout() chains built before this
                # point were not written expecting a double-boxed shape).
                # Unwrap down to the innermost non-MutableBox object, same
                # as MutationLayoutSHOULDREMOVE.get_buffer()'s own
                # unwrap_views helper does.
                while isinstance(inner_replacement, ir.MutableBox):
                    inner_replacement = inner_replacement.data
                base.data = inner_replacement
                return True
            return False
        replacement = resolve(node)
        if replacement is not None:
            setter(replacement)
            return True
        return False

    for op in body_ops:
        inputs = getattr(op, "inputs", None)
        if inputs:
            for i, inp in enumerate(inputs):
                substitute(inp, lambda r, i=i: inputs.__setitem__(i, r))

        layout = getattr(op, "layout", None)
        if not isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            continue
        target = layout.target

        def _set_target(r, layout=layout):
            layout.target = r

        substitute(target, _set_target)


def _assert_body_output_not_read_elsewhere(
    body_output_name: str,
    body_ops: list["ir.Operation"],
) -> None:
    """Raise if any spliced op besides body_output_name's own producer reads it.

    A mutated carry's write side is redirected to scratch_name, which never
    resolves to a real buffer today (see splice_while_loop's caller
    comment) -- that is only safe because, in both fixtures this bridge is
    validated against, nothing reads a mutated carry's per-iteration output
    a second time within the same body pass. This function makes that
    invariant an explicit, checked precondition rather than a silent
    assumption: if a future for_each_tile body violates it, this raises
    immediately here, rather than letting the second read silently resolve
    to a nonexistent (or, worse, stale/aliased) scratch_name buffer
    downstream in codegen.

    Uses op.get_read_writes() rather than hand-parsing inner_fn/.inputs --
    confirmed to correctly surface both read shapes (named ops.load calls
    and DynamicScalar/ExternKernelOut's direct object-reference inputs)
    uniformly for every op kind seen in either fixture.
    """
    for op in body_ops:
        op_name = getattr(op, "get_operation_name", lambda: None)()
        if op_name is not None and op_name == body_output_name:
            continue  # the producer itself; not a foreign read
        try:
            rw = op.get_read_writes()
        except Exception as e:  # noqa: BLE001 -- best-effort; see docstring
            logger.debug(
                "_assert_body_output_not_read_elsewhere: get_read_writes() "
                "raised for %s: %s",
                op_name,
                e,
            )
            continue
        read_names = {getattr(d, "name", None) for d in rw.reads}
        if body_output_name in read_names:
            raise RuntimeError(
                f"splice_while_loop: op {op_name!r} reads {body_output_name!r}, "
                "a mutated carry's per-iteration output, a second time within "
                "the same while_loop body. This carry's write side is "
                "redirected to a scratch_name that no fill/drain mechanism "
                "backs with a real buffer yet, so this read cannot be "
                "satisfied. See _assert_body_output_not_read_elsewhere's "
                "docstring in while_loop_bridge.py."
            )


def _rewire_accumulator_output(
    graph: "GraphLowering",
    while_op: "ir.WhileLoop",
    binding: CarryBinding,
    body_ops: list["ir.Operation"],
    real_input: Any,
) -> list["ir.Operation"]:
    """Make an accumulator carry's body write land in the carry's own buffer.

    Turns the body op that produces ``binding.body_output`` into an in-place
    mutation of ``real_input`` (the carry's pre-loop initial buffer) by
    swapping its layout for ``MutationLayoutSHOULDREMOVE(real_input)``, and
    repoints any outside consumer of the loop's result for this carry
    position at that same buffer.

    Why the initial buffer and not a fresh one: the accumulator's fill is
    already there. ``for_each_tile(..., init=torch.zeros(M, N))`` lowers to a
    real zeros-filling ``ComputedBuffer`` that runs before the loop, which is
    exactly the "fill" step WSR's ``CarriedReductionRecord`` pattern wants;
    reusing it avoids synthesizing a second fill (and a second buffer) that
    would then have to be kept consistent with it.

    The outside consumer is reached through the ``WhileLoop``'s
    ``MultiOutput`` child for this carry index -- the object
    ``graph.graph_outputs`` (or any downstream op) actually holds, since the
    ``WhileLoop`` itself has a ``MultiOutputLayout`` and is never read
    directly. ``splice_while_loop`` drops those children right after this
    runs, so anything still pointing at one would dangle: for
    ``split_k_fn``, that is precisely how ``graph_outputs`` ended up naming
    a removed ``buf9``, producing a wrapper with an empty body and a
    ``NameError`` at runtime.

    Returns ``body_ops`` (possibly with the rewritten op substituted in
    place, preserving order).
    """
    from torch._inductor import ir

    while_out_name = None
    for child in graph.operations:
        if not isinstance(child, ir.MultiOutput):
            continue
        if (getattr(child, "inputs", None) or [None])[0] is not while_op:
            continue
        # MultiOutput.indices is a list of (type, index) accessor steps; for
        # a WhileLoop's positional outputs it is a single tuple whose second
        # element is the carry index.
        indices: list[Any] = list(getattr(child, "indices", None) or ())
        if len(indices) == 1 and indices[0][1] == binding.carry_index:
            while_out_name = child.get_name()
            break

    body_output_name = binding.body_output.get_name()
    producer = None
    for op in body_ops:
        if isinstance(op, ir.ComputedBuffer) and op.get_name() == body_output_name:
            producer = op
            break
    if producer is None:
        logger.debug(
            "_rewire_accumulator_output: no ComputedBuffer produces carry %d's "
            "body output %r; leaving the write untouched",
            binding.carry_index,
            body_output_name,
        )
        return body_ops

    target = real_input
    while isinstance(target, ir.MutableBox):
        target = target.data
    producer.layout = ir.MutationLayoutSHOULDREMOVE(target)

    if while_out_name is not None:
        _repoint_refs_to_buffer(graph, while_out_name, target)
    return body_ops


def _repoint_refs_to_buffer(
    graph: "GraphLowering", old_name: str, new_buf: Any
) -> None:
    """Point graph outputs (and any op input) naming old_name at new_buf.

    Mirrors coarse_tile.py's ``_patch_graph_outputs``: a graph output is
    often a ``StorageBox``/``ReinterpretView`` wrapper rather than the buffer
    itself, and a ``ReinterpretView``'s own layout must be preserved (it
    describes a reshape of the result), so its ``.data`` is repointed in
    place rather than the whole node being replaced.
    """
    from torch._inductor import ir

    new_tb = ir.TensorBox(ir.StorageBox(new_buf))

    outputs = getattr(graph, "graph_outputs", None) or []
    for i, out in enumerate(outputs):
        candidate = out
        last_view = None
        while isinstance(candidate, (ir.StorageBox, ir.ReinterpretView)):
            if isinstance(candidate, ir.ReinterpretView):
                last_view = candidate
            candidate = candidate.data
        if getattr(candidate, "get_name", lambda: None)() != old_name:
            continue
        if last_view is not None:
            object.__setattr__(last_view, "data", ir.StorageBox(new_buf))
        else:
            outputs[i] = new_tb

    for op in graph.operations:
        inputs = getattr(op, "inputs", None)
        if not inputs:
            continue
        for i, inp in enumerate(inputs):
            if getattr(inp, "get_name", lambda: None)() == old_name:
                inputs[i] = new_tb


def splice_while_loop(
    graph: "GraphLowering",
    while_op: "ir.WhileLoop",
    carries: list[CarryBinding],
    trip_count: Any = None,
) -> list["ir.Operation"]:
    """Replace while_op in graph.operations with its body subgraph's ops.

    A carry the caller marked ``stacking=True`` (see ``CarryBinding``) is
    handled entirely differently from the accumulator path described below:
    its destination buffer's layout is folded from ``[trip_count, *tile]``
    to the flat result shape via ``fold_stacked_carry_layout`` (which needs
    ``trip_count``; passing None disables the fold), and NO ``scratch_name``
    redirect is applied to it. There is no intermediate per-iteration state
    to thread for a stacking carry -- each iteration writes a different
    slice of the final buffer and nothing reads a previous iteration's
    slice back -- so the write stays pointed at the real buffer and becomes
    an ordinary tile-advancing write, driven by the DimHints the caller
    stamps. That also leaves the outside consumer of the loop's result
    (a ``ReinterpretView`` of the same buffer, already describing the folded
    shape) correct with no patching.

    Carry rewiring (fill/rewrite/drain) for an ACCUMULATOR carry: each
    CarryBinding's body_output is
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

        if binding.stacking:
            # Stacking carry: fold its destination to the flat result shape
            # and leave the write pointed at it (no scratch redirect) -- see
            # this function's docstring. The read side still resolves to the
            # real buffer below, exactly as for a pass-through carry, since
            # the spliced body reads the same object it writes.
            if trip_count is not None and not fold_stacked_carry_layout(
                real_input, trip_count
            ):
                logger.debug(
                    "splice_while_loop: carry %d (%s) marked stacking but its "
                    "layout is not a foldable [trip, *tile] shape; leaving it "
                    "as-is",
                    binding.carry_index,
                    real_name,
                )
            name_map[placeholder_name] = real_name
            ref_map[placeholder_name] = real_input
            continue

        body_output_name = getattr(binding.body_output, "get_name", lambda: None)()
        is_passthrough = body_output_name == placeholder_name
        if body_output_name is not None and not is_passthrough:
            # Real per-iteration rewrite of an ACCUMULATOR carry. Redirect
            # its write in place into the carry's own initial buffer, so the
            # single spliced body copy becomes the WSR accumulator shape the
            # rest of the pipeline already handles:
            #
            #   fill    the initial buffer's own pre-loop producer (a zeros
            #           fill for split_k_fn's `torch.zeros(M, N)` init)
            #   rewrite this op reads the buffer and writes it back in place,
            #           so trip i+1 sees trip i's value
            #   drain   the buffer itself IS the final value after the last
            #           trip, so outside consumers read it directly (see
            #           _rewire_accumulator_output's graph-output patching)
            #
            # This replaces an earlier redirect to `scratch_name`, a name no
            # buffer was ever materialized under: the write kept its own
            # (now unread) name, nothing consumed it, and the whole group
            # DCE'd away leaving a wrapper whose `return (buf9,)` named a
            # buffer that no longer existed. scratch_name is retained on
            # CarryBinding as the identity a future multi-buffer carry
            # scheme can build on, but is no longer what the rewrite targets.
            #
            # The precondition the scratch redirect needed still holds and
            # still matters: nothing else in the body may read this carry's
            # per-iteration output, or the in-place write would be observed
            # mid-update by a sibling op in the same trip.
            _assert_body_output_not_read_elsewhere(body_output_name, body_ops)
            body_ops = _rewire_accumulator_output(
                graph, while_op, binding, body_ops, real_input
            )

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
