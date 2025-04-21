# src/network_utils.py
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Any, Tuple

# Assuming matrix_utils contains vector_to_symmetric_matrix
from matrix_utils import vector_to_symmetric_matrix

def get_network_indices(parcel_names: List[str], delimiter: str = '_', network_part_index: int = 1) -> Dict[str, np.ndarray]:
    """
    Creates a dictionary mapping network names to the indices of parcels belonging to that network.

    Args:
        parcel_names: List of parcel names (e.g., '7Networks_Vis_1').
        delimiter: Character separating parts of the parcel name.
        network_part_index: Index of the network name part after splitting by delimiter.

    Returns:
        Dictionary {network_name: array_of_indices}.
    """
    network_inds = {}
    for i, p_name in enumerate(parcel_names):
        try:
            parts = p_name.split(delimiter)
            if len(parts) > network_part_index:
                 network = parts[network_part_index]
                 if network in network_inds:
                     network_inds[network].append(i)
                 else:
                     network_inds[network] = [i]
            else:
                 print(f"Warning: Could not extract network from parcel name '{p_name}' with delimiter '{delimiter}' at index {network_part_index}")
        except Exception as e:
             print(f"Error processing parcel name '{p_name}': {e}")

    # Convert lists to numpy arrays
    for network in network_inds:
        network_inds[network] = np.array(network_inds[network])
    return network_inds

def get_network_colors_numeric(network_inds: Dict[str, np.ndarray], n_rois: int) -> np.ndarray:
    """Assigns a unique integer index to each network for coloring."""
    # Define a consistent order for networks (important for consistent colors)
    network_order = sorted(network_inds.keys())
    network_map = {name: i for i, name in enumerate(network_order)}

    network_colors = np.full(n_rois, -1, dtype=int) # Initialize with -1
    for network_name, indices in network_inds.items():
        if network_name in network_map:
             network_colors[indices] = network_map[network_name]
        else:
             print(f"Warning: Network '{network_name}' not in defined order, cannot assign color index.")

    if np.any(network_colors == -1):
        print("Warning: Some ROIs were not assigned a network color index.")
    return network_colors

def parcel_names_to_indices(comm_names: Set[str], all_parcel_names: List[str]) -> np.ndarray:
    """Converts a set of parcel names to their corresponding indices."""
    # Create a mapping for faster lookup
    name_to_index = {name: i for i, name in enumerate(all_parcel_names)}
    indices = [name_to_index[name] for name in comm_names if name in name_to_index]
    if len(indices) != len(comm_names):
        print("Warning: Some community names were not found in all_parcel_names.")
    return np.array(indices, dtype=int)

def get_correlation_indices(region_indices: np.ndarray, n_rois: int) -> np.ndarray:
    """
    Gets the flattened upper-triangle indices corresponding to connections
    involving any of the given region indices.
    """
    if region_indices.ndim != 1:
        region_indices = region_indices.flatten()
    temp_matrix = np.zeros((n_rois, n_rois), dtype=bool)
    # Mark rows and columns corresponding to the regions
    temp_matrix[region_indices, :] = True
    temp_matrix[:, region_indices] = True
    # Extract upper triangle indices where the marker is true
    # Need to be careful here - we want connections *between* regions or *within* the set?
    # Original code likely meant *any* connection involving one of these regions.
    # Let's create a boolean mask for the upper triangle
    ut_mask = np.zeros((n_rois, n_rois), dtype=bool)
    ut_mask[np.triu_indices(n_rois, k=1)] = True

    # Find where the region mask AND the upper triangle mask are true
    valid_connections = temp_matrix & ut_mask
    vector_indices = np.where(valid_connections[np.triu_indices(n_rois, k=1)])[0]
    return vector_indices


class GraphFromConnectivity:
    """Creates and analyzes a networkx graph from a connectivity vector or matrix."""
    def __init__(self, connectivity_data: np.ndarray, parcel_names: List[str], network_indices: Dict[str, np.ndarray]):
        """
        Args:
            connectivity_data: Either a symmetric matrix or a vectorized upper triangle.
            parcel_names: List of names corresponding to rows/cols of the matrix.
            network_indices: Dict mapping network names to parcel indices.
        """
        if connectivity_data.ndim == 1:
            self.matrix = vector_to_symmetric_matrix(connectivity_data)
        elif connectivity_data.ndim == 2 and connectivity_data.shape[0] == connectivity_data.shape[1]:
            self.matrix = connectivity_data
        else:
            raise ValueError("connectivity_data must be a vector or a square matrix.")

        if self.matrix.shape[0] != len(parcel_names):
            raise ValueError("Matrix dimension must match number of parcel names.")

        self.parcel_names = parcel_names
        self.network_indices = network_indices
        self.n_rois = len(parcel_names)

        # Generate graph and relabel nodes
        # Note: Assumes connectivity_data represents edge weights (non-negative often expected for community detection)
        # If negative values exist and are meaningful, consider absolute values or specific graph algorithms
        self.graph = nx.from_numpy_array(self.matrix)
        index_mapping = dict(zip(range(self.n_rois), self.parcel_names))
        self.graph = nx.relabel.relabel_nodes(self.graph, index_mapping)

        self._add_node_attributes()

    def _add_node_attributes(self):
        """Adds network assignment and degree centrality as node attributes."""
        # Network assignment
        network_list = sorted(self.network_indices.keys())
        network_map = {name: i for i, name in enumerate(network_list)}
        node_networks = {}
        for i, parcel_name in enumerate(self.parcel_names):
            found_network = False
            for network_name, indices in self.network_indices.items():
                if i in indices:
                    node_networks[parcel_name] = network_map.get(network_name, -1)
                    found_network = True
                    break
            if not found_network:
                 node_networks[parcel_name] = -1 # Assign default if not found

        nx.set_node_attributes(self.graph, node_networks, "network_id")

        # Degree centrality (weighted by default if edges have weights)
        # Use absolute weights if negative connections exist but magnitude matters for degree
        # graph_abs = nx.from_numpy_array(np.abs(self.matrix)) # Optional: use absolute weights for degree
        # index_mapping = dict(zip(range(self.n_rois), self.parcel_names))
        # graph_abs = nx.relabel.relabel_nodes(graph_abs, index_mapping)
        # degree_centrality = nx.degree_centrality(graph_abs)

        degree_centrality = nx.degree_centrality(self.graph) # Based on original graph weights
        nx.set_node_attributes(self.graph, degree_centrality, "degree_centrality")

    def find_communities(self, levels: int = 1) -> Tuple[Set[str], ...]:
        """
        Detects communities using the Girvan-Newman algorithm.

        Args:
            levels: The level of the community hierarchy to return.

        Returns:
            A tuple of sets, where each set contains the parcel names in a community.
        """
        communities_generator = nx.algorithms.community.girvan_newman(self.graph)
        level_communities = None
        try:
             for _ in range(levels):
                 level_communities = next(communities_generator)
        except StopIteration:
             print(f"Warning: Girvan-Newman algorithm stopped before reaching level {levels}. Returning last found level.")
             # level_communities will hold the last generated communities
        except Exception as e:
             print(f"Error during community detection: {e}")
             return tuple() # Return empty tuple on error

        # Convert node indices back to names if needed (already done via relabeling)
        # Ensure output is tuple of sets of strings (parcel names)
        if level_communities:
             return tuple(set(comm) for comm in level_communities)
        else:
             return tuple()


# Alias for backward compatibility
GraphFromCorrv2 = GraphFromConnectivity