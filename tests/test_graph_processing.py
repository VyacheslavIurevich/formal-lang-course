from filecmp import cmp
from os import remove
import pytest  # noqa: F401
from project.graph_processing import get_graph_info, create_dot_two_cycles_graph


def test_get_graph_info():
    info = get_graph_info("bzip")
    assert info.num_nodes == 632
    assert info.num_edges == 556
    assert info.labels == {"d", "a"}


def test_create_dot_two_cycles_graph():
    create_dot_two_cycles_graph((10, 15), ("a", "b"), "file.dot")
    cmp("file.dot", "tests/resources/graph.dot")
    remove("file.dot")
