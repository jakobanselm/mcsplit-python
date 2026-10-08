from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
import networkx as nx

from mcsplit.wrapper import MCSplitMode, MCSplitWrapper
from mcsplit.graph_io. vflib_loader import VFLibBinaryGraphLoader


@dataclass(frozen=True)
class DatasetPair:
    pair_id: str
    graph_g: nx.Graph
    graph_h: nx.Graph


class SolutionVerifier:
    """Rigorously validates topological correctness of common subgraph solutions."""

    @staticmethod
    def verify_mcis(
        G: nx.Graph,
        H: nx.Graph,
        solution: List[Tuple[Any, Any]],
        expected_size: int,
        connected: bool = False,
    ) -> None:
        """Validates that a node mapping induces isomorphic subgraphs in G and H."""
        # Guard: Length check
        if len(solution) != expected_size:
            raise AssertionError(
                f"Cardinality error: reported {expected_size}, got {len(solution)}"
            )

        # Guard: Empty solutions are trivially valid
        if expected_size == 0:
            return

        g_nodes = [u for u, _ in solution]
        h_nodes = [v for _, v in solution]

        # Guard: Bijection uniqueness
        if len(g_nodes) != len(set(g_nodes)) or len(h_nodes) != len(set(h_nodes)):
            raise AssertionError("Mapping contains duplicate vertex assignments (not bijective)!")

        mapping = dict(solution)

        # Guard: Strictly induced adjacency check
        for i in range(len(g_nodes)):
            u1 = g_nodes[i]
            v1 = mapping[u1]
            for j in range(i + 1, len(g_nodes)):
                u2 = g_nodes[j]
                v2 = mapping[u2]

                if G.has_edge(u1, u2) != H.has_edge(v1, v2):
                    raise AssertionError(
                        f"Induced subgraph violation between ({u1}, {u2}) and ({v1}, {v2})"
                    )

        # Guard: Connectivity constraint
        if not connected:
            return

        if not nx.is_connected(G.subgraph(g_nodes)):
            raise AssertionError("Connected mode breached: mapped G vertices are disconnected!")
        if not nx.is_connected(H.subgraph(h_nodes)):
            raise AssertionError("Connected mode breached: mapped H vertices are disconnected!")

    @staticmethod
    def verify_mces(
        G: nx.Graph,
        H: nx.Graph,
        solution: Dict[str, Any],
        expected_edge_count: int,
        connected: bool = False,
    ) -> None:
        """Validates that an edge mapping is consistent with its dual atom map."""
        edge_map = solution.get("edge_map", {})
        atom_map = solution.get("atom_map", {})

        # Guard: Edge count check
        if len(edge_map) != expected_edge_count:
            raise AssertionError(
                f"Edge count error: reported {expected_edge_count}, got {len(edge_map)}"
            )

        # Guard: Empty mapping is valid
        if expected_edge_count == 0:
            return

        # Guard: Edge incidence consistency with atom map
        for (u, v), (u_p, v_p) in edge_map.items():
            if u not in atom_map or v not in atom_map:
                raise AssertionError(f"Edge ({u}, {v}) missing from dual vertex map!")

            expected_targets = {atom_map[u], atom_map[v]}
            actual_targets = {u_p, v_p}

            if expected_targets != actual_targets:
                raise AssertionError(
                    f"Incidence violation: ({u}, {v}) -> ({u_p}, {v_p}) conflicts with atom map!"
                )

        # Guard: Connectivity of common edge subgraph
        if not connected:
            return

        if not nx.is_connected(nx.Graph(list(edge_map.keys()))):
            raise AssertionError("Connected mode breached: MCES edge set is disconnected!")


def load_dataset_pairs(directory: Union[str, Path]) -> List[DatasetPair]:
    """Finds matching _A.bin and _B.bin pairs or consecutive pairs in directory."""
    path = Path(directory)
    # Guard: Directory existence
    if not path.is_dir():
        return []

    files = sorted(list(path.glob("*.bin")))
    # Guard: Need at least 2 files
    if len(files) < 2:
        return []

    pairs: List[DatasetPair] = []
    used_indices: Set[int] = set()

    for i, file_a in enumerate(files):
        if i in used_indices:
            continue

        stem = file_a.stem
        # Match _A.bin with _B.bin
        if stem.endswith(("_A", "_a")):
            prefix = stem[:-2]
            for j, file_b in enumerate(files):
                if j in used_indices or i == j:
                    continue
                if file_b.stem == f"{prefix}_B" or file_b.stem == f"{prefix}_b":
                    pairs.append(
                        DatasetPair(
                            pair_id=prefix,
                            graph_g=VFLibBinaryGraphLoader.load(file_a),
                            graph_h=VFLibBinaryGraphLoader.load(file_b),
                        )
                    )
                    used_indices.add(i)
                    used_indices.add(j)
                    break

    # Fallback to consecutive pairing for remaining files
    remaining = [f for idx, f in enumerate(files) if idx not in used_indices]
    for k in range(0, len(remaining) - 1, 2):
        fa, fb = remaining[k], remaining[k + 1]
        pairs.append(
            DatasetPair(
                pair_id=f"{fa.stem}_{fb.stem}",
                graph_g=VFLibBinaryGraphLoader.load(fa),
                graph_h=VFLibBinaryGraphLoader.load(fb),
            )
        )

    return pairs


def run_benchmark(
    dataset_dir: Union[str, Path] = "./vflib_dataset",
    connected: bool = False,
    limit: int = 15,
) -> None:
    """Executes verification and benchmarks across discovered graph pairs."""
    pairs = load_dataset_pairs(dataset_dir)
    # Guard: Require pairs to run
    if not pairs:
        print(f"[WARN] No binary graph pairs found in '{dataset_dir}'.")
        return

    wrapper = MCSplitWrapper(connected=connected)
    mode_str = "CONNECTED" if connected else "UNCONNECTED"

    print("\n" + "=" * 92)
    print(f"BENCHMARKING VFLIB BINARY DATASET ({mode_str} MODE)")
    print("=" * 92)
    header = (
        f"{'Pair ID':<26} | {'Nodes':<7} | {'MCIS':<5} | "
        f"{'Base (s)':<10} | {'Down (s)':<10} | {'Winner':<14} | {'MCES':<5} | {'LineG (s)':<10}"
    )
    print(header)
    print("-" * len(header))

    for pair in pairs[:limit]:
        G = pair.graph_g
        H = pair.graph_h

        # 1. McSplit Base
        res_base = wrapper.solve(G, H, mode=MCSplitMode.BASE)
        if res_base.solutions:
            SolutionVerifier.verify_mcis(G, H, res_base.solutions[0], res_base.size, connected)

        # 2. McSplit Down
        res_down = wrapper.solve(G, H, mode=MCSplitMode.DOWN)
        if res_down.solutions:
            SolutionVerifier.verify_mcis(G, H, res_down.solutions[0], res_down.size, connected)

        # Guard: Invariant check - Base and Down must agree on size
        if res_base.size != res_down.size:
            raise AssertionError(f"Size mismatch on {pair.pair_id}: {res_base.size} vs {res_down.size}")

        # 3. McSplit LineGraph
        res_line = wrapper.solve(G, H, mode=MCSplitMode.LINE_GRAPH)
        if res_line.solutions:
            SolutionVerifier.verify_mces(G, H, res_line.solutions[0], res_line.size, connected)

        t_base = max(res_base.runtime_seconds, 1e-6)
        t_down = max(res_down.runtime_seconds, 1e-6)
        winner = f"Down ({t_base / t_down:.1f}x)" if t_down < t_base else f"Base ({t_down / t_base:.1f}x)"

        print(
            f"{pair.pair_id[:26]:<26} | "
            f"{f'{G.number_of_nodes()}/{H.number_of_nodes()}':<7} | "
            f"{res_base.size:<5} | "
            f"{t_base:<10.4f} | "
            f"{t_down:<10.4f} | "
            f"{winner:<14} | "
            f"{res_line.size:<5} | "
            f"{res_line.runtime_seconds:<10.4f}"
        )

    print("=" * len(header))


if __name__ == "__main__":
    run_benchmark(dataset_dir="./vflib_dataset", connected=False)