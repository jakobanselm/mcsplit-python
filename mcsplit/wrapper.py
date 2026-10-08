from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Tuple, Union
import networkx as nx

# Import the existing solver implementations
from mcsplit.solver.MCSplitBase import McSplitBase
from mcsplit.solver.McSplitDown import McSplitDown
from mcsplit.solver.McSplitLineGraph import McSplitLineGraphSolver


class MCSplitMode(str, Enum):
    """Execution modes for maximum common subgraph search."""
    BASE = "base"              # Branch-and-bound for MCIS
    DOWN = "down"              # Sequence-of-decision-problems for high similarity
    LINE_GRAPH = "line_graph"  # MCES using line graph with Whitney resolution


@dataclass(frozen=True)
class MCSplitResult:
    """Standardized result container for all McSplit execution modes."""
    size: int
    solutions: List[Any]
    runtime_seconds: float
    mode_used: MCSplitMode
    graph_g_nodes: int
    graph_h_nodes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCSplitWrapper:
    """
    Unified facade for Trimble's McSplit family of algorithms.
    Dispatches to Base, Down, or LineGraph implementations based on configuration.
    """

    def __init__(
        self,
        node_label_attr: str = "atomic_num",
        edge_label_attr: str = "order",
        connected: bool = False,
    ) -> None:
        self.node_label_attr = node_label_attr
        self.edge_label_attr = edge_label_attr
        self.connected = connected

    def solve(
        self,
        G: nx.Graph,
        H: nx.Graph,
        mode: Union[MCSplitMode, str] = MCSplitMode.BASE,
        find_all: bool = False,
    ) -> MCSplitResult:
        """
        Executes common subgraph extraction on G and H using the requested mode.
        """
        # Guard: Validate mode parameter
        try:
            mode_enum = MCSplitMode(mode)
        except ValueError:
            valid_modes = [m.value for m in MCSplitMode]
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {valid_modes}")

        # Guard: Reject non-graph inputs
        if not isinstance(G, nx.Graph) or not isinstance(H, nx.Graph):
            raise TypeError("Both inputs G and H must be instances of networkx.Graph")

        # Guard: Fast exit for empty inputs
        if G.number_of_nodes() == 0 or H.number_of_nodes() == 0:
            return MCSplitResult(
                size=0,
                solutions=[],
                runtime_seconds=0.0,
                mode_used=mode_enum,
                graph_g_nodes=G.number_of_nodes(),
                graph_h_nodes=H.number_of_nodes(),
            )

        start_time = time.perf_counter()

        if mode_enum == MCSplitMode.LINE_GRAPH:
            result_size, solutions = self._run_line_graph(G, H, find_all)
            elapsed = time.perf_counter() - start_time
            return MCSplitResult(
                size=result_size,
                solutions=solutions,
                runtime_seconds=elapsed,
                mode_used=mode_enum,
                graph_g_nodes=G.number_of_nodes(),
                graph_h_nodes=H.number_of_nodes(),
                metadata={"metric": "edges"},
            )

        if mode_enum == MCSplitMode.DOWN:
            result_size, solutions = self._run_down(G, H, find_all)
            elapsed = time.perf_counter() - start_time
            return MCSplitResult(
                size=result_size,
                solutions=solutions,
                runtime_seconds=elapsed,
                mode_used=mode_enum,
                graph_g_nodes=G.number_of_nodes(),
                graph_h_nodes=H.number_of_nodes(),
                metadata={"metric": "vertices"},
            )

        # Default: Base branch-and-bound
        result_size, solutions = self._run_base(G, H, find_all)
        elapsed = time.perf_counter() - start_time
        return MCSplitResult(
            size=result_size,
            solutions=solutions,
            runtime_seconds=elapsed,
            mode_used=mode_enum,
            graph_g_nodes=G.number_of_nodes(),
            graph_h_nodes=H.number_of_nodes(),
            metadata={"metric": "vertices"},
        )

    def _run_base(
        self, G: nx.Graph, H: nx.Graph, find_all: bool
    ) -> Tuple[int, List[Any]]:
        solver = McSplitBase(
            G,
            H,
            node_label_attr=self.node_label_attr,
            edge_label_attr=self.edge_label_attr,
            connected=self.connected,
        )
        return solver.solve(find_all=find_all)

    def _run_down(
        self, G: nx.Graph, H: nx.Graph, find_all: bool
    ) -> Tuple[int, List[Any]]:
        solver = McSplitDown(
            G,
            H,
            node_label_attr=self.node_label_attr,
            edge_label_attr=self.edge_label_attr,
            connected=self.connected,
        )
        return solver.solve(find_all=find_all)

    def _run_line_graph(
        self, G: nx.Graph, H: nx.Graph, find_all: bool
    ) -> Tuple[int, List[Any]]:
        solver = McSplitLineGraphSolver(
            G,
            H,
            node_label_attr=self.node_label_attr,
            edge_label_attr=self.edge_label_attr,
            connected=self.connected,
        )
        return solver.solve(find_all=find_all)