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


if __name__ == "__main__":
    unittest.main()
