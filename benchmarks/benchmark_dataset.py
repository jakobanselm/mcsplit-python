import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

from mcsplit.graph_io.vflib_loader import VFLibBinaryGraphLoader
from mcsplit.wrapper import MCSplitMode, MCSplitResult, MCSplitWrapper


@dataclass(frozen=True)
class DatasetPair:
    pair_id: str
    graph_g: nx.Graph
    graph_h: nx.Graph
    meta: Optional[Dict[str, Any]] = None


class SolutionVerifier:
    """Rigorously validates topological correctness and ground-truth bounds of solutions."""

    @staticmethod
    def verify_mcis(
        G: nx.Graph,
        H: nx.Graph,
        solution: List[Tuple[Any, Any]],
        expected_size: int,
        connected: bool = False,
    ) -> None:
        """Validates that a node mapping induces isomorphic subgraphs in G and H."""
        # Guard: Length check against reported size
        if len(solution) != expected_size:
            raise AssertionError(
                f"Cardinality error: reported {expected_size}, got {len(solution)}"
            )

        # Guard: Empty solutions are trivially valid
        if expected_size == 0:
            return

        g_nodes = [u for u, _ in solution]
        h_nodes = [v for _, v in solution]

        # Guard: Strict bijection (unique vertex assignments in both domains)
        if len(g_nodes) != len(set(g_nodes)):
            raise AssertionError("Duplicate vertices detected in G mapping (not bijective)!")
        if len(h_nodes) != len(set(h_nodes)):
            raise AssertionError("Duplicate vertices detected in H mapping (not bijective)!")

        mapping = dict(solution)

        # Guard: Strictly induced adjacency check (edge iff edge)
        for i in range(len(g_nodes)):
            u1 = g_nodes[i]
            v1 = mapping[u1]
            for j in range(i + 1, len(g_nodes)):
                u2 = g_nodes[j]
                v2 = mapping[u2]

                if G.has_edge(u1, u2) != H.has_edge(v1, v2):
                    raise AssertionError(
                        f"Induced subgraph violation: ({u1}, {u2}) in G is {G.has_edge(u1, u2)}, "
                        f"but ({v1}, {v2}) in H is {H.has_edge(v1, v2)}"
                    )

        # Guard: Connectivity constraint check
        if not connected:
            return

        if not nx.is_connected(G.subgraph(g_nodes)):
            raise AssertionError("Connected mode breached: mapped G vertices form a disconnected graph!")
        if not nx.is_connected(H.subgraph(h_nodes)):
            raise AssertionError("Connected mode breached: mapped H vertices form a disconnected graph!")

    @staticmethod
    def verify_mces(
        G: nx.Graph,
        H: nx.Graph,
        solution: Dict[str, Any],
        expected_edge_count: int,
        connected: bool = False,
    ) -> None:
        """Validates that an edge mapping is strictly consistent with its dual atom map."""
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

        # Guard: Dual atom map bijection
        g_atoms = list(atom_map.keys())
        h_atoms = list(atom_map.values())
        if len(g_atoms) != len(set(g_atoms)) or len(h_atoms) != len(set(h_atoms)):
            raise AssertionError("Dual atom mapping violates bijection!")

        # Guard: Edge endpoint incidence consistency with dual atom map
        for (u, v), (u_p, v_p) in edge_map.items():
            if u not in atom_map or v not in atom_map:
                raise AssertionError(f"Edge ({u}, {v}) missing from dual atom map!")

            expected_targets = {atom_map[u], atom_map[v]}
            actual_targets = {u_p, v_p}

            if expected_targets != actual_targets:
                raise AssertionError(
                    f"Incidence violation: ({u}, {v}) -> ({u_p}, {v_p}) conflicts with atom map {expected_targets}!"
                )

        # Guard: Connectivity of common edge subgraph
        if not connected:
            return

        if not nx.is_connected(nx.Graph(list(edge_map.keys()))):
            raise AssertionError("Connected mode breached: MCES edge set forms a disconnected graph!")

    @staticmethod
    def verify_ground_truth(
        res_mcis_size: int,
        meta: Dict[str, Any],
        pair_id: str,
        connected: bool,
    ) -> None:
        """
        Validates solution size against planted ground-truth metadata.
        The solver must find at least the order of the planted core.
        """
        # Guard: If connected mode is active but planted core was disconnected, size can be smaller
        if connected and not meta.get("core_is_connected", False):
            return

        planted_mcis = meta.get("planted_mcis_size")
        # Guard: Skip check if no planted size is defined
        if planted_mcis is None:
            return

        if res_mcis_size < planted_mcis:
            raise AssertionError(
                f"Ground-truth violated on '{pair_id}': Solver found {res_mcis_size}, "
                f"which is strictly less than planted core size {planted_mcis}!"
            )


def load_dataset_pairs(directory: Union[str, Path]) -> List[DatasetPair]:
    """
    Finds matching _A.bin and _B.bin pairs in directory and loads
    corresponding _meta.json ground-truth files when present.
    """
    path = Path(directory)
    # Guard: Directory existence check
    if not path.is_dir():
        return []

    files = sorted(list(path.glob("*.bin")))
    # Guard: Need at least 2 files to form a pair
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
                if file_b.stem in (f"{prefix}_B", f"{prefix}_b"):
                    # Check for ground-truth JSON metadata
                    meta_file = path / f"{prefix}_meta.json"
                    meta = None
                    if meta_file.is_file():
                        with open(meta_file, "r", encoding="utf-8") as f_meta:
                            meta = json.load(f_meta)

                    pairs.append(
                        DatasetPair(
                            pair_id=prefix,
                            graph_g=VFLibBinaryGraphLoader.load(file_a),
                            graph_h=VFLibBinaryGraphLoader.load(file_b),
                            meta=meta,
                        )
                    )
                    used_indices.add(i)
                    used_indices.add(j)
                    break

    # Fallback to consecutive pairing for unmatched files
    remaining = [f for idx, f in enumerate(files) if idx not in used_indices]
    for k in range(0, len(remaining) - 1, 2):
        fa, fb = remaining[k], remaining[k + 1]
        prefix = f"{fa.stem}_{fb.stem}"
        meta_file = path / f"{prefix}_meta.json"
        meta = None
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f_meta:
                meta = json.load(f_meta)

        pairs.append(
            DatasetPair(
                pair_id=prefix,
                graph_g=VFLibBinaryGraphLoader.load(fa),
                graph_h=VFLibBinaryGraphLoader.load(fb),
                meta=meta,
            )
        )

    return pairs


def run_benchmark(
    dataset_dir: Union[str, Path] = "./vflib_dataset",
    connected: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Executes verification and benchmarks across discovered graph pairs."""
    pairs = load_dataset_pairs(dataset_dir)
    # Guard: Require pairs to run
    if not pairs:
        print(f"[WARN] No binary graph pairs found in '{dataset_dir}'.")
        return

    active_pairs = pairs if limit is None else pairs[:limit]

    wrapper = MCSplitWrapper(connected=connected)
    mode_str = "CONNECTED" if connected else "UNCONNECTED"

    print("\n" + "=" * 92)
    print(
        f"BENCHMARKING VFLIB BINARY DATASET ({mode_str} MODE) — "
        f"Processing {len(active_pairs)} of {len(pairs)} pairs"
    )
    print("=" * 92)
    header = (
        f"{'Pair ID':<26} | {'Nodes':<7} | {'MCIS':<5} | "
        f"{'Base (s)':<10} | {'Down (s)':<10} | {'Winner':<14} | {'MCES':<5} | {'LineG (s)':<10}"
    )
    print(header)
    print("-" * len(header))

    for pair in active_pairs:
        G = pair.graph_g
        H = pair.graph_h

        # 1. McSplit Base
        res_base: MCSplitResult = wrapper.solve(G, H, mode=MCSplitMode.BASE)
        if res_base.solutions:
            SolutionVerifier.verify_mcis(G, H, res_base.solutions[0], res_base.size, connected)

        # 2. McSplit Down
        res_down: MCSplitResult = wrapper.solve(G, H, mode=MCSplitMode.DOWN)
        if res_down.solutions:
            SolutionVerifier.verify_mcis(G, H, res_down.solutions[0], res_down.size, connected)

        # Guard: Invariant check - Base and Down must agree on maximum size
        if res_base.size != res_down.size:
            raise AssertionError(
                f"Size mismatch on '{pair.pair_id}': Base={res_base.size} vs Down={res_down.size}"
            )

        # Guard: Ground-truth validation (if metadata exists)
        if pair.meta is not None:
            SolutionVerifier.verify_ground_truth(
                res_mcis_size=res_base.size,
                meta=pair.meta,
                pair_id=pair.pair_id,
                connected=connected,
            )

        # 3. McSplit LineGraph (MCES)
        res_line: MCSplitResult = wrapper.solve(G, H, mode=MCSplitMode.LINE_GRAPH)
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
    parser = argparse.ArgumentParser(description="Benchmark McSplit against VFLib binary pairs.")
    parser.add_argument("--dir", default="./vflib_dataset", help="Path to dataset directory.")
    parser.add_argument("--connected", action="store_true", help="Enforce connected subgraphs.")
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to test (default: all).")
    args = parser.parse_args()

    run_benchmark(dataset_dir=args.dir, connected=args.connected, limit=args.limit)