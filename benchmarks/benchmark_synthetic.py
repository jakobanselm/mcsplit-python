import random
import time
from typing import Any, Dict, Optional, Tuple
import networkx as nx

from mcsplit.wrapper import MCSplitMode, MCSplitWrapper


def generate_synthetic_pair(
    order: int,
    overlap_ratio: float,
    density: float = 0.25,
    seed: Optional[int] = 42,
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Constructs a controlled pair of graphs sharing an induced subgraph
    of exact order = int(order * overlap_ratio).
    """
    # Guard: Require positive graph order
    if order <= 0:
        raise ValueError("Order must be strictly positive.")

    clamped_ratio = max(0.0, min(1.0, overlap_ratio))
    common_nodes = int(order * clamped_ratio)

    common_core = nx.erdos_renyi_graph(n=common_nodes, p=density, seed=seed)

    G = nx.Graph()
    G.add_nodes_from(range(order))
    G.add_edges_from(common_core.edges())

    H = nx.Graph()
    H.add_nodes_from(range(order))
    H.add_edges_from(common_core.edges())

    rem_count = order - common_nodes
    # Guard: Early return if overlap is complete
    if rem_count <= 1:
        return G, H

    rnd = random.Random(seed)
    rem_g = nx.erdos_renyi_graph(n=rem_count, p=density, seed=rnd.randint(0, 10000))
    map_g = {i: i + common_nodes for i in rem_g.nodes()}
    G.add_edges_from(nx.relabel_nodes(rem_g, map_g).edges())

    rem_h = nx.erdos_renyi_graph(n=rem_count, p=density, seed=rnd.randint(0, 10000))
    map_h = {i: i + common_nodes for i in rem_h.nodes()}
    H.add_edges_from(nx.relabel_nodes(rem_h, map_h).edges())

    return G, H


def run_regime_analysis(order: int = 20, density: float = 0.25) -> Dict[str, Any]:
    """
    Evaluates runtime performance across high, medium, and low overlap regimes.
    Validates Trimble (2023) Section 3.7.3:
      - Overlap >= 70%: McSplitDown solves few decision problems and terminates early.
      - Overlap <= 30%: McSplitBase avoids exhaustive decision failure proofs.
    """
    wrapper = MCSplitWrapper()

    scenarios = [
        {"name": "High Overlap (90%)", "overlap": 0.90},
        {"name": "Medium Overlap (50%)", "overlap": 0.50},
        {"name": "Low Overlap (20%)", "overlap": 0.20},
    ]

    metrics: Dict[str, Any] = {}

    print("\n" + "=" * 80)
    print(f"MCSPLIT REGIME ANALYSIS (Order={order}, Density={density})")
    print("=" * 80)
    print(
        f"{'Scenario':<24} | {'MCIS':<6} | {'Base (s)':<12} | "
        f"{'Down (s)':<12} | {'Winner':<16}"
    )
    print("-" * 80)

    for sc in scenarios:
        G, H = generate_synthetic_pair(
            order=order,
            overlap_ratio=sc["overlap"],
            density=density,
            seed=999,
        )

        res_base = wrapper.solve(G, H, mode=MCSplitMode.BASE)
        res_down = wrapper.solve(G, H, mode=MCSplitMode.DOWN)

        # Guard: Invariant check - solvers must agree on maximum cardinality
        if res_base.size != res_down.size:
            raise AssertionError(
                f"Solvers disagreed on {sc['name']}: Base={res_base.size}, Down={res_down.size}"
            )

        t_base = max(res_base.runtime_seconds, 1e-6)
        t_down = max(res_down.runtime_seconds, 1e-6)

        if t_down < t_base:
            winner_str = f"Down ({t_base / t_down:.2f}x)"
        else:
            winner_str = f"Base ({t_down / t_base:.2f}x)"

        metrics[sc["name"]] = {
            "mcis_size": res_base.size,
            "base_time": t_base,
            "down_time": t_down,
            "winner": winner_str,
        }

        print(
            f"{sc['name']:<24} | {res_base.size:<6} | {t_base:<12.5f} | "
            f"{t_down:<12.5f} | {winner_str:<16}"
        )

    print("=" * 80)
    return metrics


if __name__ == "__main__":
    run_regime_analysis(order=20, density=0.25)