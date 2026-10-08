import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple, Union
import networkx as nx

from mcsplit.graph_io.vflib_loader import VFLibBinaryGraphLoader


def _generate_family_core(
    family: str,
    target_nodes: int,
    rnd: random.Random,
) -> nx.Graph:
    """
    Generates the shared core subgraph for a given graph family,
    strictly maintaining graph-theoretic existence invariants.
    """
    if family == "er_random":
        density = rnd.uniform(0.15, 0.40)
        return nx.erdos_renyi_graph(n=target_nodes, p=density, seed=rnd.randint(0, 10000))

    if family == "barabasi_albert":
        m_edges = max(1, min(target_nodes - 1, 2))
        return nx.barabasi_albert_graph(n=target_nodes, m=m_edges, seed=rnd.randint(0, 10000))

    if family == "bounded_valence":
        degree = min(3, target_nodes - 1)
        # Handshaking lemma guard: (n * d) must be even
        if (target_nodes * degree) % 2 != 0:
            degree = degree - 1 if degree > 1 else 2
        degree = min(degree, max(0, target_nodes - 1))
        return nx.random_regular_graph(d=degree, n=target_nodes, seed=rnd.randint(0, 10000))

    if family == "grid_mesh":
        side_x = max(2, int(target_nodes**0.5))
        side_y = max(2, target_nodes // side_x)
        grid = nx.grid_2d_graph(side_x, side_y)
        return nx.convert_node_labels_to_integers(grid)

    raise ValueError(f"Unknown graph family: '{family}'")


def generate_benchmark_suite(
    target_dir: Union[str, Path] = "./vflib_dataset",
    num_pairs_per_family: int = 10,
    seed: int = 420,
) -> Path:
    """
    Generates synthetic graph pairs in VFLib binary format along with
    a ground-truth JSON metadata file containing the planted isomorphism.
    """
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    families = ["er_random", "bounded_valence", "barabasi_albert", "grid_mesh"]
    print(f"[INFO] Generating {num_pairs_per_family * len(families)} pairs with ground-truth in '{out_path}'...")

    rnd = random.Random(seed)

    for fam in families:
        for idx in range(num_pairs_per_family):
            order = rnd.randint(12, 22)
            overlap_ratio = rnd.uniform(0.25, 0.90)
            desired_common = max(4, int(order * overlap_ratio))

            # 1. Build common core
            core = _generate_family_core(fam, desired_common, rnd)
            actual_common = core.number_of_nodes()

            # Guard: Adjust order if core dimensions exceeded target
            if actual_common >= order:
                order = actual_common + 2

            # 2. Build G and H with the identical core vertices [0 .. actual_common - 1]
            G = nx.Graph()
            G.add_nodes_from(range(order))
            G.add_edges_from(core.edges())

            H = nx.Graph()
            H.add_nodes_from(range(order))
            H.add_edges_from(core.edges())

            # 3. Add disjoint edges to non-overlapping partitions
            rem_nodes = order - actual_common
            if rem_nodes > 1:
                seed_g = rnd.randint(0, 100000)
                seed_h = rnd.randint(0, 100000)

                rem_g = nx.erdos_renyi_graph(n=rem_nodes, p=0.25, seed=seed_g)
                map_g = {i: i + actual_common for i in rem_g.nodes()}
                G.add_edges_from(nx.relabel_nodes(rem_g, map_g).edges())

                rem_h = nx.erdos_renyi_graph(n=rem_nodes, p=0.25, seed=seed_h)
                map_h = {i: i + actual_common for i in rem_h.nodes()}
                H.add_edges_from(nx.relabel_nodes(rem_h, map_h).edges())

            # 4. Canonical file naming
            actual_overlap = actual_common / order
            pair_name = f"{fam}_n{order}_ov{int(actual_overlap * 100):02d}_{idx:02d}"

            # Save binary graphs
            VFLibBinaryGraphLoader.dump(G, out_path / f"{pair_name}_A.bin")
            VFLibBinaryGraphLoader.dump(H, out_path / f"{pair_name}_B.bin")

            # 5. Save Ground Truth Metadata
            # The planted node mapping is identity on [0 .. actual_common - 1]
            planted_node_mapping = {str(v): v for v in range(actual_common)}
            planted_edges = [list(e) for e in core.edges()]

            meta_data: Dict[str, Any] = {
                "pair_id": pair_name,
                "family": fam,
                "order_g": order,
                "order_h": order,
                "planted_mcis_size": actual_common,
                "planted_mces_size": core.number_of_edges(),
                "core_is_connected": nx.is_connected(core),
                "planted_node_mapping": planted_node_mapping,
                "planted_edges": planted_edges,
            }

            with open(out_path / f"{pair_name}_meta.json", "w", encoding="utf-8") as f_meta:
                json.dump(meta_data, f_meta, indent=2)

    total_meta = len(list(out_path.glob("*_meta.json")))
    print(f"[SUCCESS] Generation complete: {total_meta} benchmark pairs + ground-truth files ready.")
    return out_path


if __name__ == "__main__":
    generate_benchmark_suite(target_dir="./vflib_dataset", num_pairs_per_family=10)