import networkx as nx
from typing import List, Set, Dict, Tuple, Optional, Any, Iterator

# Type aliases
Node = Any
BitMask = int
DomainPair = Tuple[BitMask, BitMask]
Solution = List[Tuple[Node, Node]]


class McSplitBase:
    """
    Optimized McSplit solver using bitsets, dynamic domain selection,
    and degree pre-ordering based on Trimble (2023).
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
        self.max_size: int = 0
        self.find_all: bool = False

        # Mappings from original Node objects to internal 0..N-1 indices
        self.g_nodes: List[Node] = []
        self.h_nodes: List[Node] = []
        self.g_map: Dict[Node, int] = {}
        self.h_map: Dict[Node, int] = {}

        # Adjacency bitmasks: adj[node_idx][label] = bitmask_of_neighbors
        self.g_adj: List[Dict[Any, BitMask]] = []
        self.h_adj: List[Dict[Any, BitMask]] = []

        # All unique edge labels found across both graphs
        self.edge_labels: Set[Any] = set()

    def solve(self, find_all: bool = False) -> Tuple[int, List[Solution]]:
        """
        Main entry point. Pre-sorts vertices by degree, builds adjacency bitmasks,
        and initiates branch-and-bound search.
        """
        self.find_all = find_all
        self.solutions = []
        self.max_size = 0

        # Guard: Fast exit if either graph is empty
        if self.G.number_of_nodes() == 0 or self.H.number_of_nodes() == 0:
            return 0, []

        # 1. Degree Pre-Ordering (Trimble Section 3.4.1)
        # Use ascending order if density(H) > 0.5, descending otherwise
        n_h = self.H.number_of_nodes()
        m_h = self.H.number_of_edges()
        density_h = (2.0 * m_h) / (n_h * (n_h - 1)) if n_h > 1 else 0.0
        reverse_degree_order = density_h <= 0.5

        self.g_nodes = sorted(
            self.G.nodes(),
            key=lambda u: self.G.degree[u],
            reverse=reverse_degree_order,
        )
        self.h_nodes = sorted(
            self.H.nodes(),
            key=lambda v: self.H.degree[v],
            reverse=reverse_degree_order,
        )

        self.g_map = {node: idx for idx, node in enumerate(self.g_nodes)}
        self.h_map = {node: idx for idx, node in enumerate(self.h_nodes)}

        # 2. Build Adjacency Bitmasks
        self.g_adj, g_labels = self._build_bit_adjacency(self.G, self.g_map)
        self.h_adj, h_labels = self._build_bit_adjacency(self.H, self.h_map)
        self.edge_labels = g_labels.union(h_labels)
        self.edge_labels.discard(0)

        # 3. Create Initial Domains
        initial_domains = self._initial_partition_bitset()
        # Guard: Exit if no compatible vertex labels exist
        if not initial_domains:
            return 0, []

        # 4. Start Recursive Search
        self._search(initial_domains, [], 0)

        return self.max_size, self.solutions

    def _build_bit_adjacency(
        self, graph: nx.Graph, mapping: Dict[Node, int]
    ) -> Tuple[List[Dict[Any, BitMask]], Set[Any]]:
        """
        Converts graph adjacencies into label-indexed bitmasks.
        """
        n_count = len(mapping)
        adj_list: List[Dict[Any, BitMask]] = [{} for _ in range(n_count)]
        labels: Set[Any] = set()

        for u, v, data in graph.edges(data=True):
            # Guard: Skip unmapped nodes if any exist
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
        Partitions vertices by matching node labels into initial domain bitmasks.
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
            # Guard: Label must exist in both graphs to form a domain
            if lbl not in h_buckets:
                continue
            domains.append((g_mask, h_buckets[lbl]))

        return domains

    def _search(
        self,
        future: List[DomainPair],
        current_mapping: List[Tuple[int, int]],
        g_reachable_mask: BitMask,
    ) -> None:
        """
        Recursive branch-and-bound search with pruning and dynamic heuristics.
        """
        current_len = len(current_mapping)

        # 1. Update Incumbent and Track Solutions
        if current_len > self.max_size:
            self.max_size = current_len
            self.solutions = [self._decode_solution(current_mapping)]
        elif self.find_all and current_len == self.max_size and current_len > 0:
            decoded = self._decode_solution(current_mapping)
            # Guard: Prevent duplicate registration during backtracking re-entry
            #if decoded not in self.solutions:
            self.solutions.append(decoded)

        # 2. Bound Pruning (Trimble Algorithm 1, Lines 4-5)
        potential = sum(min(g.bit_count(), h.bit_count()) for g, h in future)
        bound = current_len + potential

        # Guard: Prune if upper bound cannot improve or equal incumbent
        if self.find_all and bound < self.max_size:
            return
        if not self.find_all and bound <= self.max_size:
            return

        # Guard: Exit if no future domains remain to branch on
        if not future:
            return

        # 3. Dynamic Branching Selection (Trimble Section 3.4)
        selection = self._select_domain_and_pivot(future, current_len, g_reachable_mask)
        # Guard: Abort branch if no valid pivot is found
        if selection is None:
            return

        domain_idx, v_idx = selection

        # Bring selected domain to front for processing
        future[0], future[domain_idx] = future[domain_idx], future[0]
        g_domain_mask, h_domain_mask = future[0]

        v_adj_map = self.g_adj[v_idx]
        v_neighbors_all = 0
        for mask in v_adj_map.values():
            v_neighbors_all |= mask

        # 4. Branching on Values w in H Domain
        for w_idx in self._iterate_bits(h_domain_mask):
            w_adj_map = self.h_adj[w_idx]
            w_neighbors_all = 0
            for mask in w_adj_map.values():
                w_neighbors_all |= mask

            future_prime: List[DomainPair] = []

            # Partitioning: Filter consistency for edges and non-edges
            for g_cands, h_cands in future:
                # A) Edge label consistency
                for label in self.edge_labels:
                    v_nbrs_l = v_adj_map.get(label, 0)
                    w_nbrs_l = w_adj_map.get(label, 0)
                    # Guard: Label must exist in both vertex neighborhoods
                    if not v_nbrs_l or not w_nbrs_l:
                        continue

                    g_subset = g_cands & v_nbrs_l
                    if not g_subset:
                        continue

                    h_subset = h_cands & w_nbrs_l
                    if not h_subset:
                        continue

                    future_prime.append((g_subset, h_subset))

                # B) Non-edge consistency
                g_non_nbrs = (g_cands & ~v_neighbors_all) & ~(1 << v_idx)
                # Guard: Both non-neighbor subsets must be non-empty
                if not g_non_nbrs:
                    continue

                h_non_nbrs = (h_cands & ~w_neighbors_all) & ~(1 << w_idx)
                if not h_non_nbrs:
                    continue

                future_prime.append((g_non_nbrs, h_non_nbrs))

            # Update reachability for connected mode
            new_reachable = g_reachable_mask
            if self.connected:
                new_reachable |= v_neighbors_all

            # Explore child branch
            new_mapping = current_mapping + [(v_idx, w_idx)]
            self._search(future_prime, new_mapping, new_reachable)

        # 5. Backtracking (Trimble Algorithm 1, Lines 20-23)
        # Exclude vertex v from future consideration
        g_domain_remaining = g_domain_mask & ~(1 << v_idx)
        future_backtrack = future[1:]

        if g_domain_remaining:
            future_backtrack.insert(0, (g_domain_remaining, h_domain_mask))

        self._search(future_backtrack, current_mapping, g_reachable_mask)

    def _select_domain_and_pivot(
        self,
        future: List[DomainPair],
        current_len: int,
        g_reachable_mask: BitMask,
    ) -> Optional[Tuple[int, int]]:
        """
        Dynamically selects the label class minimizing max(|S_G|, |S_H|)
        and isolates the lowest set bit as pivot vertex v_idx.
        """
        # Guard: Empty domain list
        if not future:
            return None

        # Guard: Cannot expand in connected mode without reachable vertices
        if self.connected and current_len > 0 and not g_reachable_mask:
            return None

        best_idx: Optional[int] = None
        best_score = float("inf")
        best_v_idx: Optional[int] = None

        for idx, (g_mask, h_mask) in enumerate(future):
            # Guard: Skip empty masks
            if not g_mask or not h_mask:
                continue

            candidate_g = g_mask
            if self.connected and current_len > 0:
                candidate_g &= g_reachable_mask
                # Guard: Candidate class must connect to existing mapping
                if not candidate_g:
                    continue

            # Fail-first heuristic (Trimble Section 3.4): minimize max(|S_G|, |S_H|)
            score = max(candidate_g.bit_count(), h_mask.bit_count())
            if score >= best_score:
                continue

            best_score = score
            best_idx = idx
            # Isolate lowest set bit
            best_v_idx = (candidate_g & -candidate_g).bit_length() - 1

        # Guard: No viable class found
        if best_idx is None or best_v_idx is None:
            return None

        return best_idx, best_v_idx

    def _iterate_bits(self, n: int) -> Iterator[int]:
        """
        Yields indices of set bits in integer n using bit manipulation.
        """
        while n:
            lowest_bit = n & -n
            n ^= lowest_bit
            yield lowest_bit.bit_length() - 1

    def _decode_solution(self, mapping_indices: List[Tuple[int, int]]) -> Solution:
        """
        Converts internal integer indices back to original graph node identifiers.
        """
        return [(self.g_nodes[g_i], self.h_nodes[h_i]) for g_i, h_i in mapping_indices]