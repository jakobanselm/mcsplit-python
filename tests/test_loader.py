import io
import struct
import networkx as nx
import pytest

from mcsplit.graph_io.vflib_loader import VFLibBinaryGraphLoader


def test_empty_graph_roundtrip() -> None:
    """Verifies that an empty graph serializes and deserializes cleanly."""
    empty_graph = nx.Graph()
    stream = io.BytesIO()

    VFLibBinaryGraphLoader.dump(empty_graph, stream)
    stream.seek(0)
    restored = VFLibBinaryGraphLoader.load(stream)

    assert restored.number_of_nodes() == 0
    assert restored.number_of_edges() == 0


def test_isolated_nodes_roundtrip() -> None:
    """Verifies graphs containing vertices with degree 0."""
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    stream = io.BytesIO()

    VFLibBinaryGraphLoader.dump(graph, stream)
    stream.seek(0)
    restored = VFLibBinaryGraphLoader.load(stream)

    assert restored.number_of_nodes() == 5
    assert restored.number_of_edges() == 0
    assert all(restored.degree[node] == 0 for node in restored.nodes())


def test_undirected_cycle_roundtrip() -> None:
    """Verifies topology and isomorphism for undirected cyclic graphs."""
    original = nx.cycle_graph(6)
    stream = io.BytesIO()

    VFLibBinaryGraphLoader.dump(original, stream)
    stream.seek(0)
    restored = VFLibBinaryGraphLoader.load(stream, directed=False)

    assert original.number_of_nodes() == restored.number_of_nodes()
    assert original.number_of_edges() == restored.number_of_edges()
    assert nx.is_isomorphic(original, restored)


def test_directed_graph_roundtrip() -> None:
    """Verifies edge direction preservation for DiGraphs."""
    digraph = nx.DiGraph()
    digraph.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 2)])
    stream = io.BytesIO()

    VFLibBinaryGraphLoader.dump(digraph, stream)
    stream.seek(0)
    restored = VFLibBinaryGraphLoader.load(stream, directed=True)

    assert digraph.number_of_nodes() == restored.number_of_nodes()
    assert digraph.number_of_edges() == restored.number_of_edges()
    assert set(digraph.edges()) == set(restored.edges())


def test_truncated_header_returns_empty_graph() -> None:
    """Guards against incomplete binary streams (fewer than 2 bytes)."""
    stream = io.BytesIO(b"\x01")
    restored = VFLibBinaryGraphLoader.load(stream)
    assert restored.number_of_nodes() == 0


def test_truncated_edge_payload_raises_value_error() -> None:
    """Guards against unexpected EOF during adjacency unpacking."""
    # 2 nodes, node 0 declares degree 2, but only 1 target provided
    corrupt_bytes = struct.pack("<HHH", 2, 2, 1)
    stream = io.BytesIO(corrupt_bytes)

    with pytest.raises(ValueError, match="Corrupted binary data"):
        VFLibBinaryGraphLoader.load(stream)


def test_node_id_out_of_bounds_raises_value_error() -> None:
    """Guards against target node IDs referencing non-existent vertices."""
    # 2 nodes (IDs 0, 1), node 0 points to node 99
    corrupt_bytes = struct.pack("<HHHH", 2, 1, 99, 0)
    stream = io.BytesIO(corrupt_bytes)

    with pytest.raises(ValueError, match="Invalid node target"):
        VFLibBinaryGraphLoader.load(stream)


def test_max_node_limit_guard() -> None:
    """Guards against graphs exceeding 16-bit unsigned capacity (65,535)."""
    large_graph = nx.Graph()
    large_graph.add_nodes_from(range(65536))
    stream = io.BytesIO()

    with pytest.raises(ValueError, match="supports up to 65,535 nodes"):
        VFLibBinaryGraphLoader.dump(large_graph, stream)