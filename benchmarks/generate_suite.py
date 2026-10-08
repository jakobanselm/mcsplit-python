from pathlib import Path
import random
from typing import Union
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
        # Guard: Barabasi-Albert requires 1 <= m < n
        m_edges = max(1, min(target_nodes - 1, 2))
        return nx.barabasi_albert_graph(n=target_nodes, m=m_edges, seed=rnd.randint(0, 10000))

    if family == "bounded_valence":
        degree = min(3, target_nodes - 1)
        # Guard: Handshaking Lemma requires (n * d) to be even for regular graphs
        if (target_nodes * degree) % 2 != 0:
            degree = degree - 1 if degree > 1 else 2

        # Guard: Regular graph degree cannot exceed n - 1
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
    seed: int = 42,
) -> Path:
    """
    Generates a diverse set of synthetic VFLib .bin graph pairs covering
    Erdos-Renyi, Bounded Valence, 2D Mesh, and Barabasi-Albert topologies.
    """
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    families = ["er_random", "bounded_valence", "barabasi_albert", "grid_mesh"]
    print(f"[INFO] Generating {num_pairs_per_family * len(families)} pairs in '{out_path}'...")

    rnd = random.Random(seed)

    for fam in families:
        for idx in range(num_pairs_per_family):
            order = rnd.randint(16, 26)
            overlap_ratio = rnd.uniform(0.25, 0.90)
            desired_common = max(4, int(order * overlap_ratio))

            # 1. Build common induced subgraph core
            core = _generate_family_core(fam, desired_common, rnd)
            actual_common = core.number_of_nodes()

            # Guard: Adjust order if core dimensions expanded beyond initial target
            if actual_common >= order:
                order = actual_common + 2

            # 2. Build G and H with common core
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

            # 4. Save into canonical VFLib binary format
            actual_overlap = actual_common / order
            pair_name = f"{fam}_n{order}_ov{int(actual_overlap * 100):02d}_{idx:02d}"

            VFLibBinaryGraphLoader.dump(G, out_path / f"{pair_name}_A.bin")
            VFLibBinaryGraphLoader.dump(H, out_path / f"{pair_name}_B.bin")

    file_count = len(list(out_path.glob("*.bin")))
    print(f"[SUCCESS] Generation complete: {file_count} binary files ready in '{out_path}'.")
    return out_path


if __name__ == "__main__":
    generate_benchmark_suite(target_dir="./vflib_dataset", num_pairs_per_family=100)