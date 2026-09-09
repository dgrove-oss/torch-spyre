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

"""Unit tests for the generic while_loop -> coarse-tile-group bridge."""

import unittest
from unittest import mock

from torch_spyre._inductor.wsr.while_loop_bridge import (
    CarryBinding,
    carry_bindings_for,
)


class TestCarryBindingsFor(unittest.TestCase):
    def test_one_carry_positional_match(self):
        while_op = mock.Mock()
        while_op.carried_inputs = ["init0"]
        while_op.body_subgraph.graph.graph_outputs = ["out0"]

        bindings = carry_bindings_for(while_op)

        self.assertEqual(len(bindings), 1)
        self.assertIsInstance(bindings[0], CarryBinding)
        self.assertEqual(bindings[0].carry_index, 0)
        self.assertEqual(bindings[0].initial, "init0")
        self.assertEqual(bindings[0].body_output, "out0")
        self.assertTrue(bindings[0].scratch_name)

    def test_multiple_carries_preserve_order(self):
        while_op = mock.Mock()
        while_op.carried_inputs = ["init0", "init1", "init2"]
        while_op.body_subgraph.graph.graph_outputs = ["out0", "out1", "out2"]

        bindings = carry_bindings_for(while_op)

        self.assertEqual([b.carry_index for b in bindings], [0, 1, 2])
        self.assertEqual([b.initial for b in bindings], ["init0", "init1", "init2"])
        self.assertEqual([b.body_output for b in bindings], ["out0", "out1", "out2"])
        # Every binding gets a distinct scratch name.
        names = [b.scratch_name for b in bindings]
        self.assertEqual(len(names), len(set(names)))

    def test_no_carries(self):
        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.body_subgraph.graph.graph_outputs = []

        bindings = carry_bindings_for(while_op)

        self.assertEqual(bindings, [])


class TestSpliceWhileLoop(unittest.TestCase):
    def test_removes_while_op_and_inserts_body_ops(self):
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        body_op_a = mock.Mock(name="body_op_a", spec=["get_operation_name"])
        body_op_b = mock.Mock(name="body_op_b", spec=["get_operation_name"])
        multi_output = mock.Mock(name="multi_output")
        multi_output.inputs = []

        before = mock.Mock(name="before")
        before.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.inputs = []
        while_op.body_subgraph.graph.graph_outputs = []
        while_op.body_subgraph.graph.graph_inputs = {}
        while_op.body_subgraph.graph.operations = [body_op_a, body_op_b]
        while_op.body_subgraph.graph.name_to_op = {}
        while_op.body_subgraph.graph.name_to_buffer = {}

        graph = mock.Mock()
        graph.operations = [before, while_op, multi_output]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        spliced = bridge.splice_while_loop(graph, while_op, carries=[])

        self.assertEqual(spliced, [body_op_a, body_op_b])
        self.assertNotIn(while_op, graph.operations)
        self.assertIn(body_op_a, graph.operations)
        self.assertIn(body_op_b, graph.operations)

    def test_splices_at_while_op_position(self):
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        body_op = mock.Mock(name="body_op", spec=["get_operation_name"])
        before = mock.Mock(name="before")
        before.inputs = []
        after = mock.Mock(name="after")
        after.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.inputs = []
        while_op.body_subgraph.graph.graph_outputs = []
        while_op.body_subgraph.graph.graph_inputs = {}
        while_op.body_subgraph.graph.operations = [body_op]
        while_op.body_subgraph.graph.name_to_op = {}
        while_op.body_subgraph.graph.name_to_buffer = {}

        graph = mock.Mock()
        graph.operations = [before, while_op, after]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        bridge.splice_while_loop(graph, while_op, carries=[])

        self.assertEqual(graph.operations, [before, body_op, after])

    def test_direct_input_ref_and_buffer_transplant_with_real_carry(self):
        """Regression coverage for Bug A (buffer transplant) and Bug B
        (carry/xs-leaf read-side redirection), using mocks that exercise the
        actual code paths rather than just silencing AttributeErrors.

        Shape: one carry (index 0, a pass-through: body_output IS the
        placeholder) plus one non-carry xs leaf (index 1). consumer_op has
        no `.data` (so it is not routed through redirect_computed_buffer_
        reads/ComputedBuffer reconstruction -- that machinery needs a real
        frozen ComputedBuffer and is exercised end-to-end against real
        compiled graphs in test_for_each_tile_lowering.py instead); it
        holds direct .inputs references to both placeholders, mirroring
        DynamicScalar/ExternKernelOut's real read shape. producer_op is a
        distinct op that "produces" the carry's own buffer, standing in for
        an op whose output must become visible to the outer graph
        (Bug A's regression surface).
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        carry_placeholder = mock.Mock(name="carry_placeholder", spec=["get_name"])
        carry_placeholder.get_name.return_value = "while_loop_body_graph_0_0_arg0_1"

        xs_placeholder = mock.Mock(name="xs_placeholder", spec=["get_name"])
        xs_placeholder.get_name.return_value = "while_loop_body_graph_0_0_arg1_1"

        real_carry_init = mock.Mock(name="real_carry_init", spec=["get_name"])
        real_carry_init.get_name.return_value = "outer_carry_buf"

        real_xs_input = mock.Mock(name="real_xs_input", spec=["get_name"])
        real_xs_input.get_name.return_value = "outer_xs_buf"

        # consumer_op has no `.data` -- direct-object-reference read shape
        # (DynamicScalar/ExternKernelOut), routed through
        # _substitute_direct_input_refs, not redirect_computed_buffer_reads.
        consumer_op = mock.Mock(
            name="consumer_op", spec=["get_operation_name", "inputs"]
        )
        consumer_op.get_operation_name.return_value = "consumer_op"
        consumer_op.inputs = [carry_placeholder, xs_placeholder]

        produced_buf = mock.Mock(name="produced_buf", spec=["get_name"])
        produced_buf.get_name.return_value = "while_loop_body_graph_0_0_buf5"

        producer_op = mock.Mock(
            name="producer_op",
            spec=["get_operation_name", "get_outputs", "inputs"],
        )
        producer_op.get_operation_name.return_value = "producer_op"
        producer_op.get_outputs.return_value = [produced_buf]
        producer_op.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = [real_carry_init]
        while_op.inputs = [real_carry_init, real_xs_input]
        while_op.body_subgraph.graph.graph_outputs = [carry_placeholder]
        while_op.body_subgraph.graph.graph_inputs = {
            "while_loop_body_graph_0_0_arg0_1": carry_placeholder,
            "while_loop_body_graph_0_0_arg1_1": xs_placeholder,
        }
        while_op.body_subgraph.graph.operations = [producer_op, consumer_op]
        while_op.body_subgraph.graph.name_to_op = {"producer_op": producer_op}
        while_op.body_subgraph.graph.name_to_buffer = {
            "while_loop_body_graph_0_0_buf5": produced_buf
        }

        graph = mock.Mock()
        graph.operations = [while_op]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        carries = bridge.carry_bindings_for(while_op)
        self.assertEqual(len(carries), 1)

        bridge.splice_while_loop(graph, while_op, carries)

        # Bug B: consumer_op's direct .inputs references to both
        # placeholders must be rewritten to the real outer-graph objects --
        # the carry (pass-through: body_output IS the placeholder) to
        # while_op.carried_inputs[0], the xs leaf to while_op.inputs[1].
        self.assertEqual(consumer_op.inputs, [real_carry_init, real_xs_input])

        # Bug A: producer_op's own output buffer must become visible to the
        # OUTER graph's registries, not just the (mocked) inner body_graph's.
        self.assertIn("while_loop_body_graph_0_0_buf5", graph.name_to_buffer)
        self.assertIs(
            graph.name_to_buffer["while_loop_body_graph_0_0_buf5"], produced_buf
        )
        self.assertIn(produced_buf, graph.buffers)
        self.assertEqual(graph.name_to_op.get("producer_op"), producer_op)

    def test_mutated_carry_read_elsewhere_raises(self):
        """A mutated carry's body_output must not be read a second time
        within the same body pass (see _assert_body_output_not_read_
        elsewhere) -- this is Important issue 1's guard: since no
        fill/drain mechanism backs scratch_name with a real buffer yet, a
        second read of a mutated carry's own per-iteration output must
        fail loudly here rather than silently resolving to a nonexistent
        buffer downstream.
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        carry_placeholder = mock.Mock(name="carry_placeholder", spec=["get_name"])
        carry_placeholder.get_name.return_value = "while_loop_body_graph_0_0_arg0_1"

        # A distinct, real op-produced buffer -- NOT the placeholder itself
        # -- so this carry is "mutated," not pass-through.
        body_output = mock.Mock(name="body_output", spec=["get_name"])
        body_output.get_name.return_value = "while_loop_body_graph_0_0_buf7"

        real_carry_init = mock.Mock(name="real_carry_init", spec=["get_name"])
        real_carry_init.get_name.return_value = "outer_carry_buf"

        # A second, unrelated op that (per this regression case) reads the
        # mutated carry's own per-iteration output a second time.
        rogue_reader = mock.Mock(
            name="rogue_reader", spec=["get_operation_name", "get_read_writes"]
        )
        rogue_reader.get_operation_name.return_value = "rogue_reader"
        rogue_dep = mock.Mock(name="rogue_dep", spec=["name"])
        rogue_dep.name = "while_loop_body_graph_0_0_buf7"
        rogue_reader.get_read_writes.return_value = mock.Mock(reads=[rogue_dep])

        while_op = mock.Mock()
        while_op.carried_inputs = [real_carry_init]
        while_op.inputs = [real_carry_init]
        while_op.body_subgraph.graph.graph_outputs = [body_output]
        while_op.body_subgraph.graph.graph_inputs = {
            "while_loop_body_graph_0_0_arg0_1": carry_placeholder,
        }
        while_op.body_subgraph.graph.operations = [rogue_reader]
        while_op.body_subgraph.graph.name_to_op = {}
        while_op.body_subgraph.graph.name_to_buffer = {}

        graph = mock.Mock()
        graph.operations = [while_op]

        carries = bridge.carry_bindings_for(while_op)
        self.assertEqual(len(carries), 1)

        with self.assertRaises(RuntimeError):
            bridge.splice_while_loop(graph, while_op, carries)


if __name__ == "__main__":
    unittest.main()
