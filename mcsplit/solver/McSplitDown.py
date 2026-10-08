import networkx as nx
from typing import List, Set, Dict, Tuple, Optional, Any, Iterator

# Type aliases
Node = Any
BitMask = int
DomainPair = Tuple[BitMask, BitMask]
Solution = List[Tuple[Node, Node]]


class McSplitDown:
    """
    McSplit-downward solver implementing the sequence-of-decision-problems
    approach (Trimble 2023, Section 3.2.2 & Algorithm 3).
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

        self.solutions: List[Solution] = []
        self.find_all: bool = False

        # Cache best incumbent found opportunistically during search (Trimble Footnote 3)
        self.best_incumbent: List[Tuple[int, int]] = []

        # Graph node mappings to internal 0..N-1 indices
        self.g_nodes: List[Node] = []
        self.h_nodes: List[Node] = []
        self.g_map: Dict[Node, int] = {}
        self.h_map: Dict[Node, int] = {}

        # Adjacency bitmasks: adj[node_idx][edge_label] = bitmask_of_neighbors
        self.g_adj: List[Dict[Any, BitMask]] = []
        self.h_adj: List[Dict[Any, BitMask]] = []
        self.edge_labels: Set[Any] = set()

    def solve(self, find_all: bool = False) -> Tuple[int, List[Solution]]:
        """
        Executes McSplit-downward across descending target sizes t.
        """
        self.find_all = find_all
        self.solutions = []
        self.best_incumbent = []

        # Guard: Check for empty input graphs
        if self.G.number_of_nodes() == 0 or self.H.number_of_nodes() == 0:
            return 0, []

        # 1. Degree Pre-Ordering heuristic (Trimble Section 3.4.1)
        # Ascending degree order if target density > 0.5, descending otherwise
        n_h = self.H.number_of_nodes()
        m_h = self.H.number_of_edges()
        density_h = (2.0 * m_h) / (n_h * (n_h - 1)) if n_h > 1 else 0.0
        reverse_order = density_h <= 0.5

        self.g_nodes = sorted(
            self.G.nodes(),
            key=lambda u: self.G.degree[u],
            reverse=reverse_order,
        )
        self.h_nodes = sorted(
            self.H.nodes(),
            key=lambda v: self.H.degree[v],
            reverse=reverse_order,
        )

        self.g_map = {node: idx for idx, node in enumerate(self.g_nodes)}
        self.h_map = {node: idx for idx, node in enumerate(self.h_nodes)}

        # 2. Build adjacency bitmasks
        self.g_adj, g_labels = self._build_bit_adjacency(self.G, self.g_map)
        self.h_adj, h_labels = self._build_bit_adjacency(self.H, self.h_map)
        self.edge_labels = g_labels.union(h_labels)
        self.edge_labels.discard(0)

        # 3. Build initial domain partitions
        initial_domains = self._initial_partition_bitset()
        # Guard: Abort if no overlapping node labels exist
        if not initial_domains:
            return 0, []

        # 4. Compute theoretical maximum upper bound
        max_bound = sum(min(g.bit_count(), h.bit_count()) for g, h in initial_domains)
        upper_limit = min(self.G.number_of_nodes(), self.H.number_of_nodes(), max_bound)

        # Guard: Exit if no common nodes are possible
        if upper_limit == 0:
            return 0, []

        # 5. Sequence of Decision Problems (t = upper_limit down to 1)
        for target in range(upper_limit, 0, -1):
            # Guard: Exploit cached incumbent without search (Trimble Footnote 3)
            if len(self.best_incumbent) >= target:
                self.solutions = [self._decode_solution(self.best_incumbent)]
                return target, self.solutions

            found = self._search_decision(initial_domains, [], 0, target)
            if not found:
                continue

            # Target proven: return immediately
            return target, self.solutions

        return 0, []

    def _search_decision(
        self,
        future: List[DomainPair],
        current_mapping: List[Tuple[int, int]],
        g_reachable_mask: BitMask,
        target: int,
    ) -> bool:
        """
        Decision variant of the search procedure (Trimble Algorithm 3).
        Returns True immediately once a mapping of cardinality target is found.
        """
        current_len = len(current_mapping)

        # Opportunistic caching: retain largest subgraphs found along any branch
        if current_len > len(self.best_incumbent):
            self.best_incumbent = list(current_mapping)

        # Guard: Target size reached (Algorithm 3, Line 3)
        if current_len == target:
            self.solutions.append(self._decode_solution(current_mapping))
            return True

        # Bound Pruning: Upper bound strictly less than goal size (Algorithm 3, Lines 4-5)
        potential = sum(min(g.bit_count(), h.bit_count()) for g, h in future)
        bound = current_len + potential
        if bound < target:
            return False

        # Guard: No branching domains left
        if not future:
            return False

        # Branching Selection (Algorithm 3, Lines 6-7)
        selection = self._select_domain_and_pivot(future, current_len, g_reachable_mask)
        if selection is None:
            return False

        domain_idx, v_idx = selection

        # Bring chosen label class to front
        future[0], future[domain_idx] = future[domain_idx], future[0]
        g_domain_mask, h_domain_mask = future[0]

        v_adj_map = self.g_adj[v_idx]
        v_neighbors_all = 0
        for mask in v_adj_map.values():
            v_neighbors_all |= mask

        found_any = False

        # Branch across all candidates w in H (Algorithm 3, Lines 8-19)
        for w_idx in self._iterate_bits(h_domain_mask):
            w_adj_map = self.h_adj[w_idx]
            w_neighbors_all = 0
            for mask in w_adj_map.values():
                w_neighbors_all |= mask

            future_prime: List[DomainPair] = []

            # Partitioning: Filter edge and non-edge consistency
            for g_cands, h_cands in future:
                # A) Preserved labeled edges
                for label in self.edge_labels:
                    v_nbrs_l = v_adj_map.get(label, 0)
                    w_nbrs_l = w_adj_map.get(label, 0)
                    if not v_nbrs_l or not w_nbrs_l:
                        continue

                    g_subset = g_cands & v_nbrs_l
                    if not g_subset:
                        continue

                    h_subset = h_cands & w_nbrs_l
                    if not h_subset:
                        continue

                    future_prime.append((g_subset, h_subset))

                # B) Preserved non-edges
                g_non_nbrs = (g_cands & ~v_neighbors_all) & ~(1 << v_idx)
                if not g_non_nbrs:
                    continue

                h_non_nbrs = (h_cands & ~w_neighbors_all) & ~(1 << w_idx)
                if not h_non_nbrs:
                    continue

                future_prime.append((g_non_nbrs, h_non_nbrs))

            # Maintain connected reachability mask
            new_reachable = g_reachable_mask
            if self.connected:
                new_reachable |= v_neighbors_all

            new_mapping = current_mapping + [(v_idx, w_idx)]
            success = self._search_decision(
                future_prime, new_mapping, new_reachable, target
            )

            if success:
                found_any = True
                # Guard: Exit branch immediately when single solution is requested
                if not self.find_all:
                    return True

        # Backtracking: Exclude pivot v from domain (Algorithm 3, Lines 20-23)
        g_domain_remaining = g_domain_mask & ~(1 << v_idx)
        future_backtrack = future[1:]

        if g_domain_remaining:
            future_backtrack.insert(0, (g_domain_remaining, h_domain_mask))

        backtrack_success = self._search_decision(
            future_backtrack, current_mapping, g_reachable_mask, target
        )

        return found_any or backtrack_success

    def _select_domain_and_pivot(
        self,
        future: List[DomainPair],
        current_len: int,
        g_reachable_mask: BitMask,
    ) -> Optional[Tuple[int, int]]:
        """
        Dynamically chooses domain minimizing max(|S_G|, |S_H|) and isolates pivot bit.
        """
        if not future:
            return None

        # Guard: In connected mode, cannot branch without reachable vertices
        if self.connected and current_len > 0 and not g_reachable_mask:
            return None

        best_idx: Optional[int] = None
        best_score = float("inf")
        best_v_idx: Optional[int] = None

        for idx, (g_mask, h_mask) in enumerate(future):
            if not g_mask or not h_mask:
                continue

            candidate_g = g_mask
            if self.connected and current_len > 0:
                candidate_g &= g_reachable_mask
                if not candidate_g:
                    continue

            # Fail-first heuristic (Trimble Section 3.4)
            score = max(candidate_g.bit_count(), h_mask.bit_count())
            if score >= best_score:
                continue

            best_score = score
            best_idx = idx
            best_v_idx = (candidate_g & -candidate_g).bit_length() - 1

        if best_idx is None or best_v_idx is None:
            return None

        return best_idx, best_v_idx

    def _build_bit_adjacency(
        self, graph: nx.Graph, mapping: Dict[Node, int]
    ) -> Tuple[List[Dict[Any, BitMask]], Set[Any]]:
        """
        Builds adjacency bitmasks indexed by vertex index and edge label.
        """
        n_count = len(mapping)
        adj_list: List[Dict[Any, BitMask]] = [{} for _ in range(n_count)]
        labels: Set[Any] = set()

        for u, v, data in graph.edges(data=True):
            if u not in mapping or v not in mapping:
                continue

            u_idx = mapping[u]
            v_idx = mapping[v]
            label = data.get(self.edge_label, 1)
            labels.add(label)

            adj_list[u_idx][label] = adj_list[u_idx].get(label, 0) | (1 << v_idx)
            adj_list[v_idx][label] = adj_list[v_idx].get(label, 0) | (1 << u_idx)

        return adj_list, labels

    def _initial_partition_bitset(self) -> List[DomainPair]:
        """
        Groups vertices with identical labels into initial bitmask domain pairs.
        """
        g_buckets: Dict[Any, BitMask] = {}
        h_buckets: Dict[Any, BitMask] = {}

        for node, idx in self.g_map.items():
            lbl = self.G.nodes[node].get(self.node_label, 0)
            g_buckets[lbl] = g_buckets.get(lbl, 0) | (1 << idx)

        for node, idx in self.h_map.items():
            lbl = self.H.nodes[node].get(self.node_label, 0)
            h_buckets[lbl] = h_buckets.get(lbl, 0) | (1 << idx)

        domains: List[DomainPair] = []
        for lbl, g_mask in g_buckets.items():
            if lbl not in h_buckets:
                continue
            domains.append((g_mask, h_buckets[lbl]))

        return domains

    def _iterate_bits(self, n: int) -> Iterator[int]:
        """
        Iterates over set bit indices in integer n.
        """
        while n:
            b = n & -n
            n ^= b
            yield b.bit_length() - 1

    def _decode_solution(self, mapping_indices: List[Tuple[int, int]]) -> Solution:
        """
        Decodes internal integer index pairs back to graph node objects.
        """
        return [(self.g_nodes[g_i], self.h_nodes[h_i]) for g_i, h_i in mapping_indices]