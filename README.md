# McSplit: Exact Maximum Common Subgraph Algorithms

A high-performance Python implementation of the **McSplit** algorithm family for exact Maximum Common Subgraph problems, based on the PhD thesis of James Trimble (University of Glasgow, 2023).

The library supports Maximum Common Induced Subgraphs (**MCIS**) via Branch-and-Bound and Downward Decision Sequences, as well as Maximum Common Edge Subgraphs (**MCES**) via line graph transformations with exact Whitney isomorphism resolution.

---

## Key Features

- **`MCSplitMode.BASE`**: Classic Branch-and-Bound solver for Maximum Common Induced Subgraphs (MCIS) using bitset domain stores and dynamic fail-first class partitioning (Trimble Alg. 1).
- **`MCSplitMode.DOWN`**: Sequence-of-decision-problems solver (Trimble Alg. 3). Yields significant speedups when the common subgraph order is high ($\ge 70\,\%$ node overlap).
- **`MCSplitMode.LINE_GRAPH`**: Exact Maximum Common Edge Subgraph (MCES) solver. Resolves the Whitney isomorphism ambiguity ($K_3 \cong K_{1,3}$) on-the-fly via dual vertex-incidence tracking.
- **VFLib Binary I/O**: Native parser and serializer for the canonical 16-bit little-endian binary graph format used in the VFLib and Santo et al. benchmark datasets.

---

## Parameter Guide

### `connected: bool` (default: `False`)
Controls whether the common subgraph must form a single connected component.
- **Recommended for Chemistry:** When matching molecular structures (e.g., SMILES, SDF), set `connected=True`. Unconnected matchings produce fragmented sets of disjoint atoms that do not represent valid chemical scaffolds or pharmacophores.
- **General Graphs:** Set `connected=False` for arbitrary network topologies or when disconnected subgraphs are permitted.

### `find_all: bool` (default: `False`)
Determines solution enumeration behavior:
- `False`: Terminates immediately once the first optimal maximum mapping is proven.
- `True`: Exhaustively searches the search tree to return **all** symmetry-equivalent maximum common subgraphs.

### Label Attributes
- `node_label_attr` (default: `"atomic_num"`): NetworkX node attribute key used for vertex label compatibility.
- `edge_label_attr` (default: `"order"`): NetworkX edge attribute key used for edge type compatibility.

---

## Installation

### From Source
```bash
git clone [https://github.com/your-username/mcsplit-python.git](https://github.com/your-username/mcsplit-python.git)
cd mcsplit-python
pip install -e .
```

### With Optional Dependencies
```bash
# For development and testing tools
pip install -e ".[dev]"

# For chemical SMILES support (RDKit)
pip install -e ".[chem]"
```

---

## Quickstart

```python
import networkx as nx
from mcsplit import MCSplitWrapper, MCSplitMode

# Initialize the unified wrapper
wrapper = MCSplitWrapper(
    node_label_attr="atomic_num",
    edge_label_attr="order",
    connected=False,
)

# Construct example graphs
G = nx.cycle_graph(5)
H = nx.wheel_graph(5)

# 1. Compute Maximum Common Induced Subgraph (MCIS) via Base solver
result_base = wrapper.solve(G, H, mode=MCSplitMode.BASE)
print(f"MCIS Node Count: {result_base.size}")
print(f"Node Mapping (G -> H): {result_base.solutions[0]}")

# 2. Compute via Downward solver (ideal for high-overlap graphs)
result_down = wrapper.solve(G, H, mode=MCSplitMode.DOWN)
print(f"MCIS Node Count (Down): {result_down.size}")

# 3. Compute Maximum Common Edge Subgraph (MCES) via Line Graph
result_mces = wrapper.solve(G, H, mode=MCSplitMode.LINE_GRAPH)
print(f"MCES Edge Count: {result_mces.size}")
print(f"Edge Mapping: {result_mces.solutions[0]['edge_map']}")
print(f"Dual Atom Mapping: {result_mces.solutions[0]['atom_map']}")
```

---

## Algorithm Modes & Performance Characteristics

| Mode | Target Problem | Optimization Strategy | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **`BASE`** | MCIS ($\vert{}V_{\text{sub}}\vert{}$) | Branch-and-Bound with upper bounding | General instances, low-to-moderate similarity ($<70\,\%$) |
| **`DOWN`** | MCIS ($\vert{}V_{\text{sub}}\vert{}$) | Sequence of decision problems ($t = n, n-1, \dots$) | High graph similarity ($\ge 70\,\%$ node overlap) |
| **`LINE_GRAPH`** | MCES ($\vert{}E_{\text{sub}}\vert{}$) | Line graph MCIS with dual vertex consistency | Edge-centric matching, chemical reactions, bond mappings |

---

## Testing & Verification

The test suite validates graph-theoretic invariants, Whitney resolutions, and VFLib format roundtrips:

```bash
# Run all property and unit tests
pytest tests/

# Execute synthetic runtime comparison across overlap regimes
python benchmarks/benchmark_synthetic.py

# Benchmark against local VFLib binary datasets
python benchmarks/benchmark_dataset.py
```

---

## References

1. **Trimble, J. (2023).** *Partitioning algorithms for induced subgraph problems.* PhD thesis, University of Glasgow.
2. **McCreesh, C., Prosser, P., & Trimble, J. (2017).** *A partitioning algorithm for maximum common subgraph problems.* In Proceedings of IJCAI 2017.
3. **Whitney, H. (1932).** *Congruent graphs and the connectivity of graphs.* American Journal of Mathematics.
4. **Foggia, P. (2001).** *The VFLib Graph Matching Library, Version 2.0.* University of Naples "Federico II".

---

## License

This project is licensed under the MIT License.
