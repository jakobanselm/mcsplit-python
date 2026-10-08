from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

# Type aliases
Edge = Tuple[Any, Any]
LineNode = int
Solution = Dict[str, Any]


class McSplitLineGraphSolver:
    """
    Exact Maximum Common Edge Subgraph (MCES) solver using line graph transformations
    combined with on-the-fly dual vertex consistency pruning to resolve the Whitney
    isomorphism ambiguity (K3 vs K1,3) directly during search.
    """

    def __init__(
        self,
        G: nx.Graph,
        H: nx.Graph,
        node_label_attr: str = "atomic_num",
        edge_label_attr: str = "order",
        connected: bool = False,
    ):
        self.G = G
        self.H = H
        self.node_label = node_label_attr
        self.edge_label = edge_label_attr
        self.connected = connected

        # Canonical edge representations (u < v)
        self.g_edges: List[Edge] = [
            tuple(sorted((u, v))) for u, v in self.G.edges()
        ]
        self.h_edges: List[Edge] = [
            tuple(sorted((u, v))) for u, v in self.H.edges()
        ]

        self.num_g_edges = len(self.g_edges)
        self.num_h_edges = len(self.h_edges)

        # Adjacency bitmasks for line graphs
        self.lg_adj: List[int] = []
        self.lh_adj: List[int] = []

        # Search state tracking
        self.max_size: int = 0
        self.solutions: List[Solution] = []
        self.find_all: bool = False

    def solve(self, find_all: bool = False) -> Tuple[int, List[Solution]]:
        """
        Main entry point for solving MCES.
        Returns: (max_edge_count, list_of_valid_dual_solutions)
        """
        # Guard: Check for edge-empty graphs
        if self.num_g_edges == 0 or self.num_h_edges == 0:
            return 0, []

        self.find_all = find_all
        self.max_size = 0
        self.solutions = []

        self._build_line_graph_representations()
        initial_future = self._build_initial_domains()

        # Guard: Exit if no compatible edges exist
        if not initial_future:
            return 0, []

        self._search(
            future=initial_future,
            edge_mapping=[],
            vertex_map={},
            inv_vertex_map={},
            g_reachable_mask=0,
        )

        return self.max_size, self.solutions

    def _build_line_graph_representations(self) -> None:
        """
        Constructs adjacency bitmasks for the line graphs L(G) and L(H).
        Two line nodes are adjacent iff their original edges share an endpoint.
        """
        # Build L(G) bitmasks
        self.lg_adj = [0] * self.num_g_edges
        for i in range(self.num_g_edges):
            u1, v1 = self.g_edges[i]
            mask = 0
            for j in range(self.num_g_edges):
                if i == j:
                    continue
                u2, v2 = self.g_edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask |= 1 << j
            self.lg_adj[i] = mask

        # Build L(H) bitmasks
        self.lh_adj = [0] * self.num_h_edges
        for i in range(self.num_h_edges):
            u1, v1 = self.h_edges[i]
            mask = 0
            for j in range(self.num_h_edges):
                if i == j:
                    continue
                u2, v2 = self.h_edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask |= 1 << j
            self.lh_adj[i] = mask

    def _get_edge_signature(self, graph: nx.Graph, edge: Edge) -> Tuple[Any, Tuple[Any, Any]]:
        """
        Extracts invariant edge signature: (edge_order, sorted_endpoint_atom_labels).
        """
        u, v = edge
        order = graph.edges[u, v].get(self.edge_label, 1)
        u_lbl = graph.nodes[u].get(self.node_label, 0)
        v_lbl = graph.nodes[v].get(self.node_label, 0)
        return (order, tuple(sorted((u_lbl, v_lbl))))

    def _build_initial_domains(self) -> List[Tuple[int, int]]:
        """
        Partitions line graph nodes into compatible domains matching edge order
        and endpoint atom types.
        """
        g_by_sig: Dict[Any, int] = defaultdict(int)
        for i, edge in enumerate(self.g_edges):
            sig = self._get_edge_signature(self.G, edge)
            g_by_sig[sig] |= 1 << i

        h_by_sig: Dict[Any, int] = defaultdict(int)
        for j, edge in enumerate(self.h_edges):
            sig = self._get_edge_signature(self.H, edge)
            h_by_sig[sig] |= 1 << j

        future: List[Tuple[int, int]] = []
        for sig, g_mask in g_by_sig.items():
            h_mask = h_by_sig.get(sig, 0)
            # Guard: Skip signatures not present in both graphs
            if h_mask == 0:
                continue
            future.append((g_mask, h_mask))

        return future

    def _search(
        self,
        future: List[Tuple[int, int]],
        edge_mapping: List[Tuple[int, int]],
        vertex_map: Dict[Any, Any],
        inv_vertex_map: Dict[Any, Any],
        g_reachable_mask: int,
    ) -> None:
        """
        Recursive McSplit branch-and-bound search with strict Whitney-safe pruning.
        """
        current_len = len(edge_mapping)

        # 1. Update incumbent
        if current_len > self.max_size:
            self.max_size = current_len
            self.solutions = [self._format_solution(edge_mapping, vertex_map)]
        elif self.find_all and current_len == self.max_size and current_len > 0:
            self.solutions.append(self._format_solution(edge_mapping, vertex_map))

        # 2. Upper bound pruning
        bound = current_len + sum(
            min(g_mask.bit_count(), h_mask.bit_count()) for g_mask, h_mask in future
        )
        if self.find_all and bound < self.max_size:
            return
        if not self.find_all and bound <= self.max_size:
            return

        # 3. Select domain and pivot edge
        domain_idx, pivot_g = self._select_pivot(future, current_len, g_reachable_mask)
        # Guard: Exit if no valid pivot is reachable
        if pivot_g is None or domain_idx is None:
            return

        g_mask, h_mask = future[domain_idx]
        e_g = self.g_edges[pivot_g]

        # 4. Branch 1: Try matching pivot_g to compatible e_h in h_mask
        curr_h_mask = h_mask
        while curr_h_mask:
            h_bit = curr_h_mask & -curr_h_mask
            cand_h = (h_bit).bit_length() - 1
            curr_h_mask &= ~h_bit

            e_h = self.h_edges[cand_h]

            # Dual Vertex Consistency Check (Strict Whitney resolution)
            valid_orientations = self._get_valid_orientations(
                e_g, e_h, vertex_map, inv_vertex_map
            )
            # Guard: Skip candidates violating dual vertex bijection
            if not valid_orientations:
                continue

            for v_orient in valid_orientations:
                # Apply incremental vertex mapping
                new_vmap = dict(vertex_map)
                new_inv_vmap = dict(inv_vertex_map)
                new_vmap.update(v_orient)
                for u_k, v_k in v_orient.items():
                    new_inv_vmap[v_k] = u_k

                # Filter future partitions by line graph adjacency
                new_future = self._filter_domains(
                    future, domain_idx, pivot_g, cand_h
                )
                new_reachable = g_reachable_mask | self.lg_adj[pivot_g]

                self._search(
                    future=new_future,
                    edge_mapping=edge_mapping + [(pivot_g, cand_h)],
                    vertex_map=new_vmap,
                    inv_vertex_map=new_inv_vmap,
                    g_reachable_mask=new_reachable,
                )

        # 5. Branch 2: Exclude pivot_g from mapping
        new_g_mask = g_mask & ~(1 << pivot_g)
        rem_future: List[Tuple[int, int]] = []
        for i, (gm, hm) in enumerate(future):
            if i == domain_idx:
                if new_g_mask > 0:
                    rem_future.append((new_g_mask, hm))
            else:
                rem_future.append((gm, hm))

        self._search(
            future=rem_future,
            edge_mapping=edge_mapping,
            vertex_map=vertex_map,
            inv_vertex_map=inv_vertex_map,
            g_reachable_mask=g_reachable_mask,
        )

    def _select_pivot(
        self,
        future: List[Tuple[int, int]],
        current_len: int,
        g_reachable_mask: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Chooses the most constrained domain minimizing max(|S_G|, |S_H|).
        Enforces connectedness constraint when active.
        """
        best_domain_idx = None
        best_pivot = None
        min_score = float("inf")

        for idx, (gm, hm) in enumerate(future):
            # Guard: Connected search requires candidate to be adjacent to existing edge
            eligible_g = gm & g_reachable_mask if (self.connected and current_len > 0) else gm
            if eligible_g == 0:
                continue

            score = max(eligible_g.bit_count(), hm.bit_count())
            if score < min_score:
                min_score = score
                best_domain_idx = idx
                # Pick lowest-index eligible edge
                lowest_bit = eligible_g & -eligible_g
                best_pivot = (lowest_bit).bit_length() - 1

                # Fast exit for uniquely constrained domain
                if score == 1:
                    break

        return best_domain_idx, best_pivot

    def _get_valid_orientations(
        self,
        edge_g: Edge,
        edge_h: Edge,
        vertex_map: Dict[Any, Any],
        inv_vertex_map: Dict[Any, Any],
    ) -> List[Dict[Any, Any]]:
        """
        Determines consistent endpoint assignments between edge_g and edge_h.
        Guarantees that no triangle (K3) is ever mapped to a star (K1,3).
        """
        u1, u2 = edge_g
        v1, v2 = edge_h

        lbl_u1 = self.G.nodes[u1].get(self.node_label, 0)
        lbl_u2 = self.G.nodes[u2].get(self.node_label, 0)
        lbl_v1 = self.H.nodes[v1].get(self.node_label, 0)
        lbl_v2 = self.H.nodes[v2].get(self.node_label, 0)

        def can_bind(u: Any, v: Any) -> bool:
            if u in vertex_map:
                return vertex_map[u] == v
            # If u is unmapped, v must also be unmapped
            return v not in inv_vertex_map

        valid: List[Dict[Any, Any]] = []

        # Orientation 1: u1 -> v1, u2 -> v2
        if lbl_u1 == lbl_v1 and lbl_u2 == lbl_v2 and can_bind(u1, v1) and can_bind(u2, v2):
            valid.append({u1: v1, u2: v2})

        # Orientation 2: u1 -> v2, u2 -> v1
        if lbl_u1 == lbl_v2 and lbl_u2 == lbl_v1 and can_bind(u1, v2) and can_bind(u2, v1):
            orient_2 = {u1: v2, u2: v1}
            # Avoid duplicate symmetric orientation if endpoints are unmapped and identical
            if not valid or orient_2 != valid[0]:
                valid.append(orient_2)

        return valid

    def _filter_domains(
        self,
        future: List[Tuple[int, int]],
        pivot_domain_idx: int,
        pivot_g: int,
        cand_h: int,
    ) -> List[Tuple[int, int]]:
        """
        Splits domains into adjacent and non-adjacent classes relative to (pivot_g, cand_h).
        Preserves strictly induced line graph adjacency.
        """
        new_future: List[Tuple[int, int]] = []
        adj_g_mask = self.lg_adj[pivot_g]
        adj_h_mask = self.lh_adj[cand_h]

        pivot_bit_g = 1 << pivot_g
        pivot_bit_h = 1 << cand_h

        for idx, (gm, hm) in enumerate(future):
            # Exclude current decision from its originating partition
            cur_gm = gm & ~pivot_bit_g if idx == pivot_domain_idx else gm
            cur_hm = hm & ~pivot_bit_h if idx == pivot_domain_idx else hm

            if cur_gm == 0 or cur_hm == 0:
                continue

            # Class 1: Edges adjacent in line graph (sharing an endpoint)
            gm_adj = cur_gm & adj_g_mask
            hm_adj = cur_hm & adj_h_mask
            if gm_adj > 0 and hm_adj > 0:
                new_future.append((gm_adj, hm_adj))

            # Class 2: Edges non-adjacent in line graph (disjoint endpoints)
            gm_non = cur_gm & ~adj_g_mask
            hm_non = cur_hm & ~adj_h_mask
            if gm_non > 0 and hm_non > 0:
                new_future.append((gm_non, hm_non))

        return new_future

    def _format_solution(
        self,
        edge_mapping: List[Tuple[int, int]],
        vertex_map: Dict[Any, Any],
    ) -> Solution:
        """
        Decodes internal index representations into canonical node and edge dictionaries.
        """
        edge_map: Dict[Edge, Edge] = {
            self.g_edges[gi]: self.h_edges[hi] for gi, hi in edge_mapping
        }
        return {
            "edge_map": edge_map,
            "atom_map": dict(vertex_map),
            "edge_count": len(edge_map),
            "atom_count": len(vertex_map),
        }


# ---------------------------------------------------------------------------
# Verification & Whitney Proof
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("DEMO 1: WHITNEY COUNTEREXAMPLE (K3 vs K1,3)")
    print("=" * 80)

    # Graph G is a Triangle (K3): 3 vertices, 3 edges
    G_triangle = nx.Graph()
    G_triangle.add_nodes_from([0, 1, 2], atomic_num=6)
    G_triangle.add_edges_from([(0, 1), (1, 2), (2, 0)], order=1)

    # Graph H is a Claw / Star (K1,3): 4 vertices, 3 edges
    H_claw = nx.Graph()
    H_claw.add_nodes_from([0, 1, 2, 3], atomic_num=6)
    H_claw.add_edges_from([(0, 1), (0, 2), (0, 3)], order=1)

    # In plain Line Graph MCIS: L(K3) has 3 nodes (all adjacent), L(K1,3) has 3 nodes (all adjacent).
    # A naive line graph solver would report max_size = 3 (claiming K3 is isomorphic to K1,3).
    # The true Maximum Common Edge Subgraph (MCES) between K3 and K1,3 has size 2 (Path P3).

    solver = McSplitLineGraphSolver(G_triangle, H_claw, connected=True)
    max_edges, solutions = solver.solve(find_all=False)

    print(f"Graph G edges (Triangle): {list(G_triangle.edges())}")
    print(f"Graph H edges (Claw)    : {list(H_claw.edges())}")
    print(f"\nResulting MCES Edge Count: {max_edges}")
    print(f"Found Dual Solution      : {solutions[0]}")

    assert max_edges == 2, f"Failed! Expected 2 edges, got {max_edges} (Whitney anomaly breached)"
    print("\n[SUCCESS] Whitney resolution verified: K3 <-> K1,3 was correctly rejected!") 