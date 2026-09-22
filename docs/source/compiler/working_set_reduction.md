# Working Set Reduction - Design Document

Working set reduction decomposes operations or sequences of operations into
loops doing computations in a piecewise manner, for instance decomposing a
large matrix multiplication `x @ y` into a series of multiplications on groups
of `x`'s rows. The resulting operations operate on smaller tensors with the
following benefits:

- Smaller tensors help alleviate hardware limitations with respect to per-core,
  per-tensor DDR/HBM access span.
- Smaller tensors help reduce memory bandwidth pressure by making it possible
  to keep tensors in scratchpad memory.

This document motivates and walks through the working set reduction approach
adopted in torch-spyre.

**Quick navigation:**

- [Approach](#approach)
- [`for_each_tile`: an explicit, co-indexed tiling loop](#for_each_tile-an-explicit-co-indexed-tiling-loop)
- [Example: tiling `y = a + b; z = y * c`](#example-tiling-y--a--b-z--y--c)
- [Composing and nesting](#composing-and-nesting)
- [Implementation](#implementation)
- [Legacy frontend: `spyre_hint`](#legacy-frontend-spyre_hint)
- [Related documents](#related-documents)

## Approach

We intend to support both implicit (compiler generated) and explicit (source
code driven) working set reduction. Explicit working set reduction lets us
decouple the effort on working set reduction heuristics from downstream tasks
(intermediate representations, analyses, and transformations). Eventually, the
combination of the two can result in better performance and productivity than
either solution in isolation.

The classic illustration is a matrix multiplication. Given `z = x @ y` with
`x: [M, K]` and `y: [K, N]`, multiple tiling choices are valid: tile `x`
along `M`, tile `y` along `N`, tile both, or tile the reduction axis `K`.
Tiling along non-reduction axes produces independent output tiles. Tiling
along the reduction axis is qualitatively different: each tile produces a
partial sum, and an extra accumulation step combines them.

:::{figure} ../_static/images/wsr/matmul-tiling-options.png
:alt: Four tiling options for z = x @ y
:width: 760px
:align: center

Tiling options for a matrix multiplication. Options 1-3 tile non-reduction
axes; each tile is independent. Option 4 tiles the reduction axis K and
introduces an extra accumulation step.
:::

The current explicit frontend for this is **`for_each_tile`**, a prototype
higher-order op (torch-spyre#3965) that expresses a tiling loop directly at the
source level, without a separate dimension-naming step. An older frontend,
**`spyre_hint`**, is still supported and described in [Legacy frontend:
`spyre_hint`](#legacy-frontend-spyre_hint) below, but new code should prefer
`for_each_tile`.

## `for_each_tile`: an explicit, co-indexed tiling loop

`for_each_tile` is `scan` with the tiling made explicit: **one co-indexed loop
level** that reduces every operand to a per-step tile — a narrow view, a whole
invariant, or a gathered pool row — threads an optional carry, and optionally
lays each step's result tile back into a full-size output along one axis.

```python
def for_each_tile(
    body,
    operands,
    *,
    dims,
    tile_size: int,
    init=None,
    out_dim=None,
    reverse: bool = False,
):
    """Run `body` once per tile over a co-indexed tiling of `operands`.

    Args:
        body: ``(carry, tiles) -> (next_carry, out_tile)``. ``tiles`` arrives in
            operand order, each operand already reduced to its per-step tile.
            ``next_carry`` is ignored in map mode (``init=None``) and ``out_tile``
            in reduction mode (``out_dim=None``). Same restriction as ``scan``:
            the body may not alias input to output or output to output.
        operands: flat sequence of every input -- sliced, gathered and invariant.
        dims: per-operand tile spec as a tuple, or a single spec broadcast to all of
            them. ``int d`` slices the operand into ``tile_size``-wide contiguous views
            along ``d``; ``None`` passes it whole every step; ``Gather(axis, index)``
            takes one pool row per step.
        tile_size: the tile's size along each tiled axis. The loop's trip count is
            derived: ``shape[dim] // tile_size`` per sliced operand, ``len(index)`` per
            gathered one. All of them must agree.
        init: carry init (tensor or pytree of tensors); ``None`` means no carry.
        out_dim: ``int d`` lays step ``i``'s tile at ``narrow(d, i*extent, extent)``
            of the returned output. ``None`` means the body emits no tile.
        reverse: visit tiles high to low. The output still lands in natural order.

    Returns:
        ``(final_carry, out)``, either of which is ``None`` for the unused mode.
    """
```

Unlike `spyre_hint`, there is no separate step to declare and name tensor
dimensions before compiling: `dims=`/`tile_size=` directly describe the
tiling of the arguments passed to `for_each_tile` itself, and the trip count
is derived from the operand shapes rather than supplied by hand. Three kinds
of operand are supported per the `dims` entry:

- **Sliced** (`dims[i]` is an `int`): the operand is cut into `tile_size`-wide
  contiguous views along that dimension; step `i` sees
  `narrow(dim, i * tile_size, tile_size)`.
- **Invariant** (`dims[i]` is `None`): the operand is passed whole, unchanged,
  every step — the equivalent of an outer-level-invariant read in the
  `spyre_hint` model.
- **Gathered** (`dims[i]` is a `Gather(axis, index)`): step `i` sees one row
  of a pool tensor, selected by `index[i]` along `axis`
  (`pool.index_select(axis, index[i])`). This has no analog in the
  `spyre_hint` frontend.

`init`/`out_dim` cover the two directions data can cross the loop boundary
that a purely elementwise tiling doesn't need: `init` threads a **carry**
(e.g. a running accumulator) from one step to the next, and `out_dim` lays
each step's output tile back into the correct slice of a full-size result.
Either can be `None` independently — a pure reduction has no `out_dim`; a
pure per-tile map has no `init`.

## Example: tiling `y = a + b; z = y * c`

The example below tiles the same computation used throughout this document's
companion, [`coarse_tiling_loops.md`](coarse_tiling_loops.md) — but with
`for_each_tile` instead of `spyre_hint`. `a`, `b`, `c` are `[1024, 4096]`
tensors; the loop tiles dimension 0 into 8 steps of 128 rows each:

```python
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile

a = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
b = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")
c = torch.randn(1024, 4096, dtype=torch.float16).to("spyre")

def fn(a, b, c):
    def body(_, tiles):
        a_tile, b_tile, c_tile = tiles
        y_tile = a_tile + b_tile
        return None, y_tile * c_tile

    _, z = for_each_tile(body, (a, b, c), dims=(0, 0, 0), tile_size=128, out_dim=0)
    return z

print(torch.compile(fn)(a, b, c))
```

All three operands are sliced along dimension 0 with the same `tile_size`, so
the trip count is `1024 // 128 == 8`. `body` receives one `[128, 4096]` tile
of each input per step, has no carry (`init=None`, so its first return value
is ignored), and returns each step's `[128, 4096]` result tile, which
`out_dim=0` lays into the corresponding `narrow(0, i*128, 128)` slice of the
returned `[1024, 4096]` output.

This is exactly the same tiling shape as the `spyre_hint`
[Small Example](coarse_tiling_loops.md#small-example) in the companion
document — a single loop, `y = a + b` then `z = y * c` — and it lowers to the
same downstream mechanism (`loop_info: CoarseTileInfo`, a `CountedLoopSchedulerNode`,
a `LoopSpec`). `docs/tools/capture_for_each_tile_ir.py` regenerates the real,
captured IR/OpSpec/`bundle.mlir` for this exact example, which the
[implementation reference](coarse_tiling_loops.md) quotes at length.

## Composing and nesting

`for_each_tile` expresses exactly **one** co-indexed loop level per call. A
nested tiling loop nest — the `spyre_hint` model's stacked
`with spyre_hint(...): with spyre_hint(...):` scopes — is expressed by
composing two `for_each_tile` calls, with the inner call inside the outer
call's `body`:

```python
def fn(a, b, c):
    def outer_body(_, outer_tiles):
        a_row, b_row, c_row = outer_tiles

        def inner_body(_, inner_tiles):
            a_tile, b_tile, c_tile = inner_tiles
            y_tile = a_tile + b_tile
            return None, y_tile * c_tile

        _, z_row = for_each_tile(
            inner_body, (a_row, b_row, c_row), dims=(1, 1, 1), tile_size=1024, out_dim=1
        )
        return None, z_row

    _, z = for_each_tile(outer_body, (a, b, c), dims=(0, 0, 0), tile_size=512, out_dim=0)
    return z
```

The compiler's lowering pipeline (see [Layer 1's Prove → Splice → Identify →
Stamp
sequence](coarse_tiling_loops.md#prove-splice-identify-stamp-how-a-for_each_tile-call-becomes-loop_info))
runs to a fixed point over nested `for_each_tile`/`while_loop` structures, so
this composition is handled uniformly rather than as a special case — see
that section for how a two-level nest like this ends up with a two-entry
`loop_group_id` on the innermost ops, analogous to the `spyre_hint` Small
Example's `(0, 0)`.

## Implementation

`for_each_tile` is implemented as a thin frontend over PyTorch's `scan`
higher-order op: it validates and normalizes `dims`/`tile_size` into the
per-operand `TileSpec`s that decide how each operand is reduced to a tile
(slice, gather, or pass through invariant), builds the `scan` body, and calls
`torch._higher_order_ops.scan.scan`. This means a `for_each_tile` call reaches
Inductor as an `ir.WhileLoop` (`scan`'s own lowering), not as a distinct
"tiling loop" IR node in its own right.

The Spyre backend's job is then to recognize which `ir.WhileLoop`s are
provably bounded, tile-shaped loops — as opposed to genuinely
data-dependent `while_loop`s, which stay as `ir.WhileLoop` — and rewrite them
into the same `loop_info`-carrying representation the `spyre_hint` frontend
produces. The mechanics of that recognition and rewrite (`try_prove_for_each_tile`,
`splice_while_loops`, `_stamp_direct_loop_info`, and the surrounding carry
machinery in `while_loop_bridge.py`) are described in detail in
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#layer-1--pre-scheduling-ir-pass),
which both frontends share from that point on: everything past "a run of ops
carries a `loop_info: CoarseTileInfo`" — the `CountedLoopSchedulerNode`
scheduler wrapper (Layer 2) and the `LoopSpec` codegen tree (Layer 3) — is
common machinery, indifferent to whether the `loop_info` came from
`for_each_tile` or `spyre_hint`.

## Legacy frontend: `spyre_hint`

`spyre_hint` is an older, still-supported explicit frontend that names tensor
dimensions up front and tiles them via nested context managers, rather than
expressing the tiling loop directly at the call site. New code should prefer
`for_each_tile`; this section remains for code that still uses `spyre_hint`
and for understanding the hint-driven half of the compiler pipeline.

Explicit working set reduction via `spyre_hint` is decomposed in four stages:

1. Introduce source-level hints on operations and tensors to drive working set
   reduction.
2. Introduce encodings of working set reduction decisions as metadata on LLIR
   operations and buffers.
3. Lower source-level hints to IR metadata.
4. Transform the annotated IR into an executable program.

Implicit working set reduction via compiler heuristics reuses stage 2 and
beyond, and this is also where `for_each_tile` rejoins the pipeline: it
produces the stage-2 IR metadata directly, skipping stages 1 and 3.

To explicitly control working set reduction with `spyre_hint`, we name tensor
dimensions and tile them.

### Example: naming dimensions and tiling

```python
M, K, N = 64, 256, 128

declare_tensor_dim("M", M)
declare_tensor_dim("K", K)
declare_tensor_dim("N", N)


def kernel(x, y, z):
    with spyre_hint(num_tiles_per_dim={"M": 8}):
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            p = x @ y
        return p + z


x = torch.rand(M, K, dtype=torch.float16).to("spyre")
y = torch.rand(K, N, dtype=torch.float16).to("spyre")
z = torch.rand(M, N, dtype=torch.float16).to("spyre")

name_tensor_dims(x, ["M", "K"])
name_tensor_dims(y, ["K", "N"])
name_tensor_dims(z, ["M", "N"])

print(torch.compile(kernel)(x, y, z))
```

In this example, we declare three tensor dimensions `"M"`, `"K"`, and `"N"`
using `declare_tensor_dim`, map three device tensors to these dimensions
using `name_tensor_dims`, and tile the `"M"` and `"K"` dimensions using
`spyre_hint`. The matmul operation is tiled along both `"M"` and `"K"`
whereas the final add operation is only tiled along `"M"`.

Hints are introduced with the `with spyre_hint(**kwargs):` pattern. The
keyword takes a dictionary that maps a dimension name to a tile count. Each
hint scope tiles exactly one named dimension. The value is the number of
tiles to split that dimension into.
`num_tiles_per_dim={"M": 8}` produces 8 tiles along `M`, each of size
`M / 8`.

The keyword is `num_tiles_per_dim=`. Legacy aliases `tiles=` and `slices=`
still parse but are deprecated and will be removed in a future release.

A single `spyre_hint(...)` call accepts at most one tiling keyword, and the
dictionary names at most one dimension. To tile two dimensions, nest two
hint scopes, as in the example above:

```python
def kernel(x, y, z):
    with spyre_hint(num_tiles_per_dim={"M": 8}):
        with spyre_hint(num_tiles_per_dim={"K": 4}):
            return x @ y + z
```

The nested-scope order matches the resulting loop-nest order. The outer
scope is the outer loop.

For operations with no input tensors, such as `torch.full`, the
`named_dims=` keyword supplies dimension names directly on the operation's
output:

```python
with spyre_hint(named_dims=["M", "N"]):
    out = torch.full((M, N), 0.0, dtype=torch.float16, device="spyre")
```

Named tensor dimensions must be provided for inputs to `torch.compile` but
are intended to be derived most of the time for computed tensors.

### Dimensions vs. named dimensions

Named dimensions are deliberately distinct from tensor shape:

> **Dimensions are ephemeral and reflect the current view. Named dimensions
> are durable and reflect the intent and storage.**

A 2D tensor and its `flatten()` produce different `tensor.shape` values but
keep the same named dimensions. Two flat 1D tensors with reversed naming
order are not equivalent even though their shapes match.

:::{figure} ../_static/images/wsr/dims-vs-named-dims.png
:alt: Comparison of tensor shape and named dimensions across reshapes
:width: 760px
:align: center

Named dimensions track the intent of the tensor's storage and survive view
transformations like `flatten()`. Reversed name order produces a tensor that
is *not* equivalent even when the shape matches.
:::

This separation is what allows hints to refer to logical axes (`"M"`,
`"K"`, `"N"`) regardless of whether intermediate views have collapsed or
re-shaped them.

### Example: view-based dimension splitting

Named tensor dimensions are intended to reflect the tensor layout in memory.
For instance, the following code is valid:

```python
def kernel(x_1d, y, z):
    with spyre_hint(num_tiles_per_dim={"M": 8, "K": 4}):
        return x_1d.view(M, K) @ y + z

x_1d = torch.rand(M * K, dtype=torch.float16).to("spyre")

name_tensor_dims(x_1d, ["M", "K"])
```

Here the `name_tensor_dims` invocation records that `x_1d` while declared as
a 1d tensor is in essence a 2d tensor with outer dimension `"M"` and inner
dimension `"K"`. Consequently, the count of dimensions of a tensor or view
may be different from its named dimension count.

The order of named dimensions is significant. The following two declarations
are not equivalent:

```python
name_tensor_dims(x, ["M", "K"]) # M before K
name_tensor_dims(x, ["K", "M"]) # K before M
```

Named tensor dimensions are expected to be consistent with the mathematical
properties of the operations involving the tensors. For instance, in `x @ y`
there must exist `n>0` such that `x_named_dims[-n:] == y_named_dims[:n]`,
as for instance with named dimensions `["A", "B", "C", "D"]` for `x` and
`["C", "D", "E"]` for `y`. In this example, the reduction dimension is the
flattened dimension `["C", "D"]`.

### `spyre_hint` intermediate representation

Hints are automatically assigned a unique id.

We extend LLIR as follows:

- We add a list of computed dimensions to each computed buffer.
- We add iteration dimensions to each operation mapping iteration variables
  to lists of named dimensions.
- We add hints to each operation mapping hint ids to the hint values for
  every enclosing hint.

For instance, for `x @ y` in our example, we add:

- Computed dimensions: `["M", "N"]`
- Iteration dimensions: `{d0: ["M"], d1: ["K"], d2: ["N"]}` assuming
  variables `d0`, `d1`, and `d2` respectively map to dim 0 of `x`, the
  reduction dimension, and dim 1 of `y`.
- Hints: `{3: {"tiles": {"M": 8}}, 4: {"tiles": {"K": 4}}}`

:::{figure} ../_static/images/wsr/named-dim-propagation.png
:alt: Named-dim propagation through a matrix multiplication
:width: 760px
:align: center

Named dimensions on the inputs determine the iteration-variable mapping on
the operation, which in turn determines the named dimensions of the output.
The right-hand panel illustrates the flattened-reduction case where two input
dimensions collapse into a single iteration variable.
:::

Hint ids are positive integers. They are unique, not in general consecutive,
but they respect the nesting order. Concretely, if a hint is nested inside
another hint, the inner hint id will be greater than the outer hint id.

Hint ids make it possible to reconstruct hint scopes from operation metadata.
Nested `with spyre_hint(...)` blocks form a tree that can be recovered from
the recorded ids.

:::{figure} ../_static/images/wsr/hint-scope-tree.png
:alt: Nested spyre_hint scopes form a tree of hint ids
:width: 760px
:align: center

Each `spyre_hint(...)` block gets a unique, monotonically increasing id.
Operations inherit every enclosing hint, and the partial order on ids
recovers the nesting tree.
:::

### `spyre_hint` lowering

Spyre hints are captured on the FX graph using the
`torch.fx.traceback.annotate` context manager and preserved through AOT
using custom pre- and post-AOT passes to save and restore the hints. Node
matching pre- and post-AOT relies on topological sorting.

Hints on LLIR operations are derived from origin FX nodes on demand via a
getter method (`get_op_hints`).

Named tensor dimensions are specified only on input tensors. To use these
names for optimization throughout the PyTorch graph, they must be propagated
to intermediate tensors produced by operations. This requires propagating
dimension name metadata through the Inductor intermediate representation.
This is implemented by the `propagate_named_dims` pass.

In most cases, tracking dimension names through operations is
straightforward. The primary complexity comes from handling views,
particularly views that split or combine dimensions, as shown in
[Example: view-based dimension splitting](#example-view-based-dimension-splitting).

The current implementation assumes that when a view splits a dimension, the
input tensor's corresponding dimension already contains the necessary number
of dimension names with compatible sizes (for example, `["M", "K"]` in that
example). Named dimensions are propagated through intermediate tensors
and aligned to tensor dimensions using stride-based analysis, ensuring
correctness under view transformations.

More automated dimension naming is planned. In the current implementation,
if an input dimension is unnamed, or if a view transformation is
inconsistent with the user-provided dimension names, a warning is emitted
and propagation continues with partial or inferred information.

The pass runs in `CustomPreSchedulingPasses`, split across two slots. A
hint-driven half (`propagate_named_dims`, `assign_dim_hints`, and the
hint-derived half of coarse tiling) runs immediately after dead-code
elimination, **before** stickification — it only needs host-side
`FixedLayout` (size/stride) and loop-variable ranges, so there is no reason
to wait for device layouts. A span-overflow half runs later, after
stickification, because it needs `FixedTiledLayout.device_layout` (device
size, stride map) to detect and correct spans that overflow the hardware
memory budget. `for_each_tile`'s own `splice_while_loops` pass runs earlier
still — before dead-code elimination — since it must resolve every
`for_each_tile`-shaped `ir.WhileLoop` down to a `loop_info`-carrying group
before the hint-driven half's grouping and dead-code elimination can see a
flat op list. Both hint-driven halves still run before work-division and
scratchpad planning consume the resulting iteration spaces: work-division
must see the post-tiling iteration space regardless of which frontend or
which half produced it. See
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#groups-derivation-and-placement-in-custompreschedulingpasses)
for the full pass ordering and the rationale for the two-slot split
(issue #3135).

:::{figure} ../_static/images/wsr/pipeline-placement.png
:alt: Where WSR runs in the Spyre Inductor pipeline
:width: 820px
:align: center

WSR-specific stages (highlighted) sit between Inductor lowering and
work-division/scratchpad planning. AOT pre/post passes preserve hint
metadata across retracing.
:::

### Transformation

The annotated IR — whether produced by `spyre_hint`'s hint-driven passes or by
`for_each_tile`'s direct-stamping pipeline — is transformed into a tiled loop
nest by the **coarse-tiling** machinery. Each contiguous run of operations
sharing the same tiling decision is rewritten with reduced per-iteration
ranges, wrapped in a counted loop, and emitted as nested `LoopSpec` structures
that the SuperDSC codegen lowers to hardware MLIR (`scf.for` + `affine.apply`
- `sdsc_execute`).

The reduction in working set is what makes intermediates fit in LX
scratchpad: an intermediate buffer that is produced and consumed inside the
same loop iteration never needs an HBM allocation.

:::{figure} ../_static/images/wsr/memory-access-before-after.png
:alt: Memory access pattern before and after working set reduction
:width: 880px
:align: center

For `y = a + b; z = y * c`: without WSR, the intermediate `y` is full-size
and spills to HBM. With WSR, each tile of `y` lives in LX scratchpad for the
duration of one iteration and is consumed immediately by the next op.
:::

The full mechanics — how loop identity is carried through Inductor's
flat-list pipeline, how the loop perimeter prevents cross-group fusion, how
buffers crossing the loop boundary are classified — are documented in
[`coarse_tiling_loops.md`](coarse_tiling_loops.md). The design rationale
for the `spyre_hint`-era mechanics is in [RFC 1358: Coarse
Tiling](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md).

A buffer that crosses the loop boundary and has a non-empty
`output_tiled_dims`/`tiled_dims_per_read` entry at a given level (see
[`coarse_tiling_loops.md`](coarse_tiling_loops.md#attribute-contract-on-iroperation))
advances its base address once per loop iteration at that level, so its HBM
pool allocation must be sized for every tile it will occupy across the
loop's run, not just one. `hbm_pool_planning.py`'s `_compute_size_bytes` sizes
each buffer from its full `FixedTiledLayout.device_layout.device_size`, which
spans every tile the buffer occupies across the loop; sizing it for a single
tile would let the loop overrun into whatever buffer the allocator packed
next to it. A buffer whose `output_tiled_dims`/`tiled_dims_per_read` entries
are empty at every level, by contrast, never advances and is a candidate for
a fixed-address LX scratchpad slot instead — see
[`coarse_tiling_loops.md`](coarse_tiling_loops.md) for the `accum_full` /
`accum_tile` buffers this distinction matters most for.

## Related documents

- [`coarse_tiling_loops.md`](coarse_tiling_loops.md) — implementation
  reference for the transformation stage (Layer 1 IR pass, Layer 2
  scheduler wrapper, Layer 3 codegen tree), covering both the
  `for_each_tile` and `spyre_hint` paths into `loop_info`.
- [RFC 1358: Coarse-Tiling Loop IR Design
  Rationale](https://github.com/torch-spyre/rfcs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md),
  which explains the reasoning behind the three-layer design.
- [`scratchpad_planning.md`](scratchpad_planning.md) — how LX scratchpad
  allocation consumes the per-tile iteration spaces produced by WSR.
- [`work_division_planning.md`](work_division_planning.md) — how work
  distribution across cores runs after WSR on the reduced ranges.
