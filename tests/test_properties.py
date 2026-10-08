from typing import Any, Dict, List, Set, Tuple
import networkx as nx

from mcsplit.wrapper import MCSplitMode, MCSplitWrapper


# ---------------------------------------------------------------------------
# 1. Connected Property Tests (MCIS / MCCIS)
# ---------------------------------------------------------------------------

def test_connected_property_size_reduction() -> None:
    """
    Validates that connected=True strictly limits the MCIS to a single
    connected component when multiple disjoint components are present.
    """
    # G and H each consist of two disjoint triangles (K3 + K3, total 6 nodes)
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    H = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))

    for node in G.nodes():
        G.nodes[node]["atomic_num"] = 0
    for node in H.nodes():
        H.nodes[node]["atomic_num"] = 0

    wrapper_unconnected = MCSplitWrapper(connected=False)
    wrapper_connected = MCSplitWrapper(connected=True)

    # A) Test McSplit Base
    base_unconn = wrapper_unconnected.solve(G, H, mode=MCSplitMode.BASE)
    base_conn = wrapper_connected.solve(G, H, mode=MCSplitMode.BASE)

    # Guard: Unconnected mode must capture both components (6 nodes)
    if base_unconn.size != 6:
        raise AssertionError(
            f"Base unconnected failed: expected 6 nodes, got {base_unconn.size}"
        )

    # Guard: Connected mode must only capture a single component (3 nodes)
    if base_conn.size != 3:
        raise AssertionError(
            f"Base connected failed: expected 3 nodes, got {base_conn.size}"
        )

    # B) Test McSplit Down
    down_unconn = wrapper_unconnected.solve(G, H, mode=MCSplitMode.DOWN)
    down_conn = wrapper_connected.solve(G, H, mode=MCSplitMode.DOWN)

    # Guard: Down mode size consistency
    if down_unconn.size != 6 or down_conn.size != 3:
        raise AssertionError(
            f"Down connected/unconnected mismatch: unconn={down_unconn.size}, conn={down_conn.size}"
        )

    print("[PASS] Connected Size Reduction Test (Disjoint Triangles: 6 -> 3 nodes).")


def test_connected_subgraph_topology() -> None:
    """
    Verifies that the mapped vertices strictly induce connected subgraphs
    in both input graphs G and H.
    """
    # Create two arbitrary connected/partially-overlapping graphs
    G = nx.lollipop_graph(m=5, n=3)  # Clique of 5 + path of 3
    H = nx.barbell_graph(m1=4, m2=2) # Two cliques of 4 connected by bridge

    for node in G.nodes():
        G.nodes[node]["atomic_num"] = 0
    for node in H.nodes():
        H.nodes[node]["atomic_num"] = 0

    wrapper = MCSplitWrapper(connected=True)

    for mode in [MCSplitMode.BASE, MCSplitMode.DOWN]:
        result = wrapper.solve(G, H, mode=mode)

        # Guard: Ensure solution exists
        if result.size == 0 or not result.solutions:
            raise AssertionError(f"Expected non-empty solution in mode {mode}")

        first_sol: List[Tuple[int, int]] = result.solutions[0]
        nodes_g = [u for u, _ in first_sol]
        nodes_h = [v for _, v in first_sol]

        sub_g = G.subgraph(nodes_g)
        sub_h = H.subgraph(nodes_h)

        # Guard: Validate G induction connectivity
        if not nx.is_connected(sub_g):
            raise AssertionError(f"Mode {mode} produced disconnected subgraph in G!")

        # Guard: Validate H induction connectivity
        if not nx.is_connected(sub_h):
            raise AssertionError(f"Mode {mode} produced disconnected subgraph in H!")

    print("[PASS] Connected Subgraph Topology Test (Induced subgraphs are connected).")


# ---------------------------------------------------------------------------
# 2. Line Graph & Whitney Ambiguity Tests (MCES)
# ---------------------------------------------------------------------------

def test_whitney_isomorphism_resolution() -> None:
    """
    Tests the Whitney (1932) theorem counterexample:
      L(K3) == L(K1,3) == K3 (both have 3 mutually adjacent line nodes).
    A correct MCES solver must reject matching 3 edges because K3 is not
    subgraph isomorphic to K1,3; the true MCES has 2 edges (P3).
    """
    k3 = nx.complete_graph(3)    # Triangle (3 nodes, 3 edges)
    k1_3 = nx.star_graph(3)      # Claw (4 nodes, 3 edges)

    wrapper = MCSplitWrapper(connected=True)
    res_line = wrapper.solve(k3, k1_3, mode=MCSplitMode.LINE_GRAPH)

    # Guard: Exact edge count check
    if res_line.size != 2:
        raise AssertionError(
            f"Whitney resolution breached! Expected 2 edges, got {res_line.size}"
        )

    # Guard: Verify that the dual vertex mapping has exactly 3 vertices (P3 path)
    sol = res_line.solutions[0]
    atom_count = sol.get("atom_count", 0)
    if atom_count != 3:
        raise AssertionError(
            f"Whitney resolution invalid: expected 3 mapped vertices, got {atom_count}"
        )

    print(f"[PASS] Whitney Ambiguity Test: Correctly resolved K3 vs K1,3 to {res_line.size} edges.")


def test_line_graph_edge_incidence_consistency() -> None:
    """
    Verifies that all edge mappings returned by McSplitLineGraphSolver
    strictly preserve endpoint incidence according to the dual atom_map.
    """
    # Cycle C5 vs Wheel W5
    G = nx.cycle_graph(5)
    H = nx.wheel_graph(5)

    wrapper = MCSplitWrapper(connected=True)
    result = wrapper.solve(G, H, mode=MCSplitMode.LINE_GRAPH)

    # Guard: Solution must be non-empty
    if result.size == 0 or not result.solutions:
        raise AssertionError("Expected non-empty MCES solution for C5 and W5.")

    sol = result.solutions[0]
    edge_map: Dict[Tuple[int, int], Tuple[int, int]] = sol["edge_map"]
    atom_map: Dict[int, int] = sol["atom_map"]

    for (u, v), (u_prime, v_prime) in edge_map.items():
        # Guard: Both endpoints must be present in atom_map
        if u not in atom_map or v not in atom_map:
            raise AssertionError(f"Edge endpoints ({u}, {v}) missing from dual vertex map!")

        mapped_endpoints = {atom_map[u], atom_map[v]}
        target_endpoints = {u_prime, v_prime}

        # Guard: Endpoints of the mapped edge must match atom_map targets
        if mapped_endpoints != target_endpoints:
            raise AssertionError(
                f"Edge incidence violated: ({u}, {v}) -> ({u_prime}, {v_prime}) "
                f"conflicts with atom mapping {atom_map[u]}, {atom_map[v]}"
            )

    print("[PASS] Line Graph Incidence Consistency Test (Dual atom mapping is valid).")


def test_line_graph_connected_vs_unconnected() -> None:
    """
    Verifies the connected flag in McSplitLineGraphSolver.
    When connected=True, the edge set must induce a single connected edge component.
    """
    # G has two disjoint edges: (0, 1) and (2, 3)
    G = nx.Graph([(0, 1), (2, 3)])
    # H has two disjoint edges: (10, 11) and (12, 13)
    H = nx.Graph([(10, 11), (12, 13)])

    wrap_unconn = MCSplitWrapper(connected=False)
    wrap_conn = MCSplitWrapper(connected=True)

    res_unconn = wrap_unconn.solve(G, H, mode=MCSplitMode.LINE_GRAPH)
    res_conn = wrap_conn.solve(G, H, mode=MCSplitMode.LINE_GRAPH)

    # Guard: Unconnected should map both independent edges (size = 2)
    if res_unconn.size != 2:
        raise AssertionError(
            f"LineGraph unconnected failed: expected 2 edges, got {res_unconn.size}"
        )

    # Guard: Connected must only map one single edge (size = 1)
    if res_conn.size != 1:
        raise AssertionError(
            f"LineGraph connected failed: expected 1 edge, got {res_conn.size}"
        )

    print("[PASS] Line Graph Connected vs Unconnected Test (Disjoint edges: 2 -> 1 edge).")


# ---------------------------------------------------------------------------
# Main Execution Runner
# ---------------------------------------------------------------------------

def run_all_property_tests() -> None:
    """Executes the complete test suite for connected and line-graph invariants."""
    print("=" * 80)
    print("STARTING PROPERTY VALIDATION SUITE")
    print("=" * 80)

    # 1. Connected MCIS tests
    test_connected_property_size_reduction()
    test_connected_subgraph_topology()

    # 2. Line Graph MCES tests
    test_whitney_isomorphism_resolution()
    test_line_graph_edge_incidence_consistency()
    test_line_graph_connected_vs_unconnected()

    print("=" * 80)
    print("ALL PROPERTY TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_property_tests()