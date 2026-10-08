import io
from pathlib import Path
import struct
from typing import BinaryIO, Union
import networkx as nx


class VFLibBinaryGraphLoader:
    """
    Parser and serializer for the VFLib binary graph format (.bin / .sub).
    Reference: Foggia (2001), VFLib 2.0 manual Section 4.5.

    Binary Format Specification:
      - 16-bit unsigned integers, little-endian encoded (<H).
      - Word 0: Number of nodes N (0 <= N <= 65535).
      - For each node i from 0 to N-1:
          - Word: Out-degree d_i (count of outgoing edges from node i).
          - Next d_i words: Target node IDs adjacent to node i.
    """

    @staticmethod
    def load(
        file_or_path: Union[str, Path, BinaryIO],
        directed: bool = False,
        default_node_label: int = 0,
        default_edge_label: int = 1,
        node_label_attr: str = "atomic_num",
        edge_label_attr: str = "order",
    ) -> nx.Graph:
        """
        Reads a VFLib binary file or binary stream into a networkx Graph/DiGraph.
        """
        # Guard: Open file if path string or Path object is provided
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "rb") as stream:
                return VFLibBinaryGraphLoader._parse_stream(
                    stream=stream,
                    directed=directed,
                    default_node_label=default_node_label,
                    default_edge_label=default_edge_label,
                    node_label_attr=node_label_attr,
                    edge_label_attr=edge_label_attr,
                )

        return VFLibBinaryGraphLoader._parse_stream(
            stream=file_or_path,
            directed=directed,
            default_node_label=default_node_label,
            default_edge_label=default_edge_label,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
        )

    @staticmethod
    def _parse_stream(
        stream: BinaryIO,
        directed: bool,
        default_node_label: int,
        default_edge_label: int,
        node_label_attr: str,
        edge_label_attr: str,
    ) -> nx.Graph:
        raw_header = stream.read(2)
        # Guard: Handle empty file or immediate EOF
        if len(raw_header) < 2:
            return nx.DiGraph() if directed else nx.Graph()

        (num_nodes,) = struct.unpack("<H", raw_header)
        graph = nx.DiGraph() if directed else nx.Graph()

        # Guard: Exit early if graph has zero vertices
        if num_nodes == 0:
            return graph

        # Initialize nodes with required attributes for McSplit
        for node_id in range(num_nodes):
            graph.add_node(node_id, **{node_label_attr: default_node_label})

        # Sequentially parse adjacency entries for every vertex
        for src_node in range(num_nodes):
            raw_degree = stream.read(2)
            # Guard: Detect truncated file when reading out-degree
            if len(raw_degree) < 2:
                raise ValueError(
                    f"Corrupted binary data: unexpected EOF reading degree for node {src_node}"
                )

            (out_degree,) = struct.unpack("<H", raw_degree)
            # Guard: Skip reading edges when degree is zero
            if out_degree == 0:
                continue

            expected_bytes = 2 * out_degree
            raw_targets = stream.read(expected_bytes)
            # Guard: Detect incomplete edge array
            if len(raw_targets) < expected_bytes:
                raise ValueError(
                    f"Corrupted binary data: expected {out_degree} targets for node {src_node}, "
                    f"received only {len(raw_targets) // 2}"
                )

            target_nodes = struct.unpack(f"<{out_degree}H", raw_targets)
            for tgt_node in target_nodes:
                # Guard: Reject invalid node index references
                if tgt_node >= num_nodes:
                    raise ValueError(
                        f"Invalid node target {tgt_node}: total node count is {num_nodes}"
                    )
                graph.add_edge(
                    src_node,
                    tgt_node,
                    **{edge_label_attr: default_edge_label},
                )

        return graph

    @staticmethod
    def dump(
        graph: nx.Graph,
        file_or_path: Union[str, Path, BinaryIO],
    ) -> None:
        """
        Serializes a networkx Graph/DiGraph into the canonical VFLib binary format.
        """
        num_nodes = graph.number_of_nodes()
        # Guard: Enforce 16-bit word capacity limit
        if num_nodes > 65535:
            raise ValueError(
                f"Graph contains {num_nodes} nodes; VFLib format supports up to 65,535 nodes."
            )

        # Build contiguous 0..N-1 indexing
        nodes_list = list(graph.nodes())
        node_to_id = {node: idx for idx, node in enumerate(nodes_list)}

        buffer = io.BytesIO()
        buffer.write(struct.pack("<H", num_nodes))

        is_directed = graph.is_directed()

        for u in nodes_list:
            if is_directed:
                targets = sorted(node_to_id[v] for v in graph.successors(u))
            else:
                targets = sorted(node_to_id[v] for v in graph.neighbors(u))

            out_degree = len(targets)
            buffer.write(struct.pack("<H", out_degree))

            # Guard: Skip edge payload writing if degree is zero
            if out_degree == 0:
                continue

            buffer.write(struct.pack(f"<{out_degree}H", *targets))

        data = buffer.getvalue()

        # Guard: Write to disk path if string or Path provided
        if isinstance(file_or_path, (str, Path)):
            with open(file_or_path, "wb") as f:
                f.write(data)
            return

        file_or_path.write(data)