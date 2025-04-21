# src/network_analysis.py
import numpy as np
import pandas as pd
import networkx as nx
from networkx import algorithms # For efficiency measures
import logging
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats # For efficiency comparison

# Import necessary functions/classes from other src modules
# Ensure these imports use absolute paths (assuming src is findable via sys.path)
from network_utils import GraphFromConnectivity, parcel_names_to_indices
from matrix_utils import vector_to_symmetric_matrix, convert_upper_mat # Added convert_upper_mat alias

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_run_level_graphs(
    run_wfc_data: Dict[str, Dict[str, np.ndarray]],
    parcel_names: List[str],
    network_indices: Dict[str, np.ndarray],
    sub_names_per_run: np.ndarray, # Subject ID corresponding to each run in WFC data
    threshold_percentile: float = 98.0
) -> Dict[str, Dict[str, List[Tuple[str, GraphFromConnectivity]]]]:
    """
    Creates GraphFromConnectivity objects for each run, thresholding WFC.

    Args:
        run_wfc_data: Dict[model_key][group_key] -> WFC vectors (n_runs, n_features).
        parcel_names: List of parcel names.
        network_indices: Dict mapping network names to parcel indices.
        sub_names_per_run: Array of subject IDs, one per run/row in WFC arrays.
        threshold_percentile: Percentile to threshold WFC vectors before graph creation.

    Returns:
        Dict[model_key][group_key] -> List of tuples [(subject_id, graph_object)].
    """
    all_graphs = {}
    # Need sub_names to match the FIRST dimension of WFC vectors
    if run_wfc_data:
        # Get expected number of runs from the first available entry
        first_model = list(run_wfc_data.keys())[0]
        first_group = list(run_wfc_data[first_model].keys())[0]
        expected_runs = run_wfc_data[first_model][first_group].shape[0]
        if expected_runs != len(sub_names_per_run):
             logging.error(f"Shape mismatch: Sub names count ({len(sub_names_per_run)}) != WFC runs ({expected_runs}). Cannot reliably create graphs.")
             return {} # Return empty if shapes mismatch fundamentally

    for model_key, model_wfc in run_wfc_data.items():
        all_graphs[model_key] = {}
        logging.info(f"Creating graphs for model: {model_key}")
        for group_key, wfc_vectors in model_wfc.items():
            # Double check shape again for this specific entry
            if wfc_vectors.shape[0] != len(sub_names_per_run):
                 logging.warning(f"Mismatch runs vs sub_names for {model_key}/{group_key}. Skipping.")
                 continue

            graph_list = []
            logging.info(f"  Processing group: {group_key} ({wfc_vectors.shape[0]} runs)")
            for i, wfc_vector in enumerate(wfc_vectors):
                subject_id = sub_names_per_run[i]
                try:
                    # Thresholding based on non-zero positive values
                    positive_wfc = wfc_vector[wfc_vector > 1e-9]
                    thresholded_wfc = np.zeros_like(wfc_vector) # Start with zeros
                    if len(positive_wfc) > 0:
                        threshold_value = np.percentile(positive_wfc, threshold_percentile)
                        thresholded_wfc[wfc_vector >= threshold_value] = wfc_vector[wfc_vector >= threshold_value]

                    graph_obj = GraphFromConnectivity(thresholded_wfc, parcel_names, network_indices)
                    graph_list.append((subject_id, graph_obj))
                except Exception as e:
                    logging.error(f"    Failed to create graph for run {i}, subject {subject_id}: {e}", exc_info=True)

            all_graphs[model_key][group_key] = graph_list
    return all_graphs


def calculate_degree_centrality(
    all_graphs: Dict[str, Dict[str, List[Tuple[str, GraphFromConnectivity]]]],
    parcel_names: List[str]
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calculates average and standard deviation of degree centrality per parcel,
    grouped by model and analysis group.

    Args:
        all_graphs: Dictionary containing run-level graph objects.
        parcel_names: List of parcel names.

    Returns:
        Tuple containing:
        - avg_dc_dfs: Dict[model_key][group_key] -> DataFrame(index=parcel_name, columns=['Mean DC'])
        - std_dc_dfs: Dict[model_key][group_key] -> DataFrame(index=parcel_name, columns=['Std Dev DC'])
    """
    model_group_wise_avg_dc = {}
    model_group_wise_std_dc = {}
    n_parcels = len(parcel_names)

    for model_key, group_data in all_graphs.items():
        model_group_wise_avg_dc[model_key] = {}
        model_group_wise_std_dc[model_key] = {}
        logging.info(f"Calculating Degree Centrality for model: {model_key}")
        for group_key, graph_list in group_data.items():
            logging.info(f"  Processing group: {group_key}")
            parcel_dc_values = {name: [] for name in parcel_names}

            for _, graph_obj in graph_list:
                try:
                    dc = nx.degree_centrality(graph_obj.graph) # Recalculate here? Or trust attribute? Let's trust attribute if present.
                    for parcel_name, attributes in graph_obj.graph.nodes(data=True):
                        dc_val = attributes.get('degree_centrality', np.nan) # Get from attribute, default NaN
                        parcel_dc_values[parcel_name].append(dc_val)
                except Exception as e:
                     logging.error(f"    Error processing graph node attributes: {e}")
                     # Handle appropriately, maybe add NaNs for all parcels for this graph?

            avg_dc = np.full(n_parcels, np.nan)
            std_dc = np.full(n_parcels, np.nan)
            for i, name in enumerate(parcel_names):
                values = [v for v in parcel_dc_values[name] if not np.isnan(v)] # Filter NaNs
                if values:
                    avg_dc[i] = np.mean(values)
                    std_dc[i] = np.std(values)

            avg_df = pd.DataFrame({'Mean DC': avg_dc}, index=parcel_names)
            std_df = pd.DataFrame({'Std Dev DC': std_dc}, index=parcel_names)
            model_group_wise_avg_dc[model_key][group_key] = avg_df
            model_group_wise_std_dc[model_key][group_key] = std_df

    return model_group_wise_avg_dc, model_group_wise_std_dc


def calculate_efficiency(
    all_graphs: Dict[str, Dict[str, List[Tuple[str, GraphFromConnectivity]]]],
    labels_per_run: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculates global and local efficiency per run, expecting numeric results
    from NetworkX functions. Optionally compares groups if labels are provided.

    Args:
        all_graphs: Dictionary containing run-level graph objects.
        labels_per_run: Optional array of labels (0 or 1 assumed) corresponding
                        to each run in the flattened data (across all groups/models).

    Returns:
        Dict[model_key][group_key] -> DataFrame with efficiency measures per run.
        Performs and prints Mann-Whitney U test results if labels are valid
        and allows for a 2-group comparison.
    """
    all_efficiency_results = {}
    perform_comparison = False

    if labels_per_run is not None:
        # Check if comparison is feasible (at least two groups needed)
        unique_labels = np.unique(labels_per_run[~np.isnan(labels_per_run)])
        if len(unique_labels) == 2:
            perform_comparison = True
            # logging.info("Labels provided, will perform efficiency comparison.") # Removed logging

    for model_key, group_data in all_graphs.items():
        all_efficiency_results[model_key] = {}
        # logging.info(f"Calculating Efficiency for model: {model_key}") # Removed logging
        for group_key, graph_list in group_data.items():
            # logging.info(f"  Processing group: {group_key}") # Removed logging
            n_runs = len(graph_list)
            if n_runs == 0:
                all_efficiency_results[model_key][group_key] = pd.DataFrame({
                    'subject_id': [], 'label': [],
                    'global_efficiency': [], 'local_efficiency': []
                })
                continue

            # Extract subject IDs for this group's graphs
            current_group_subject_ids = [item[0] for item in graph_list]
            n_current_runs = len(current_group_subject_ids) # Number of runs for this model/group combo

            # Prepare labels for this subset of runs if comparison is enabled
            current_run_labels = np.full(n_current_runs, np.nan) # Default to NaN
            if perform_comparison:
                # Need to slice the main labels_per_run array to match the runs in this graph_list
                # This assumes the order of graph_list matches the order in the original 'fcs' array
                # which matches the order in 'labels_per_run'. This is a strong assumption.
                # A more robust way would be to pass subject_ids and labels for *all* runs and filter.
                # Let's rely on the assumption based on how data was prepared in notebook 1.

                # Find the indices in the original flattened 'fcs' array that correspond to
                # the subject IDs in 'current_group_subject_ids'. This is complex without
                # the original full list of subject IDs per run.

                # Simpler (less safe) assumption: The graph_list order IS the order in labels_per_run
                if n_current_runs == len(labels_per_run):
                     current_run_labels = labels_per_run
                else:
                     # logging.warning(f"  Efficiency: Mismatch runs ({n_current_runs}) vs labels ({len(labels_per_run)}). Cannot confidently align labels. Skipping comparison for {group_key}.") # Removed logging
                     perform_comparison = False # Disable comparison if unable to align labels


            global_eff = np.full(n_current_runs, np.nan)
            local_eff = np.full(n_current_runs, np.nan)


            for i, (_, graph_obj) in enumerate(graph_list):
                # Get subject ID for better (but removed) logging messages, still useful conceptually
                subject_id = current_group_subject_ids[i]

                try:
                    G = graph_obj.graph # Get the NetworkX graph object
                    # Basic check for graph validity
                    if not isinstance(G, nx.Graph) or G.number_of_nodes() == 0:
                        global_eff[i] = np.nan
                        local_eff[i] = np.nan
                        continue # Skip to the next run if the graph object is invalid or empty

                    # --- Calculate Global Efficiency (Expect Float/Int ONLY) ---
                    global_eff_result = nx.global_efficiency(G)
                    if isinstance(global_eff_result, (float, int, np.floating, np.integer)):
                         global_eff[i] = float(global_eff_result)
                         if not np.isfinite(global_eff[i]):
                             global_eff[i] = np.nan
                    else:
                         # Unexpected type - set to NaN
                         global_eff[i] = np.nan


                    # --- Calculate Local Efficiency (Expect Float/Int ONLY) ---
                    local_eff_result = nx.local_efficiency(G)
                    # This will explicitly mark the standard dictionary return as an error (set to NaN)
                    if isinstance(local_eff_result, (float, int, np.floating, np.integer)):
                         local_eff[i] = float(local_eff_result)
                         if not np.isfinite(local_eff[i]):
                            local_eff[i] = np.nan
                    else:
                        # Unexpected type (likely the standard dictionary) - set to NaN
                        local_eff[i] = np.nan

                except Exception:
                    # Catch any other unexpected errors during calculation
                    global_eff[i] = np.nan
                    local_eff[i] = np.nan


            # Create DataFrame for this model/group
            eff_df = pd.DataFrame({
                'subject_id': current_group_subject_ids,
                'label': current_run_labels if current_run_labels is not None else np.nan,
                'global_efficiency': global_eff,
                'local_efficiency': local_eff
            })
            all_efficiency_results[model_key][group_key] = eff_df

            # --- Perform statistical comparison if possible ---
            if perform_comparison and current_run_labels is not None:
                 # Filter NaNs and perform test
                 eff_df_cleaned = eff_df.dropna(subset=['global_efficiency', 'local_efficiency', 'label'])
                 unique_labels_found = eff_df_cleaned['label'].dropna().unique()
                 if len(unique_labels_found) == 2:
                      unique_labels_found.sort() # Ensure consistent order [0, 1]
                      group0_glob = eff_df_cleaned.loc[eff_df_cleaned['label'] == unique_labels_found[0], 'global_efficiency']
                      group1_glob = eff_df_cleaned.loc[eff_df_cleaned['label'] == unique_labels_found[1], 'global_efficiency']
                      group0_loc = eff_df_cleaned.loc[eff_df_cleaned['label'] == unique_labels_found[0], 'local_efficiency']
                      group1_loc = eff_df_cleaned.loc[eff_df_cleaned['label'] == unique_labels_found[1], 'local_efficiency']

                      # Ensure groups have enough samples (>0 after filtering) before testing
                      if len(group0_glob) > 0 and len(group1_glob) > 0:
                          try:
                              stat_g, p_g = stats.mannwhitneyu(group0_glob, group1_glob, alternative='two-sided')
                              print(f"    Mann-Whitney U ({group_key} - Global Eff): p = {p_g:.4f}") # Keep this print as it's analysis output
                          except ValueError: # Handle cases like identical distributions where test raises error
                               print(f"    Mann-Whitney U failed for Global Eff ({group_key}).") # Keep this print for failure indication

                          if len(group0_loc) > 0 and len(group1_loc) > 0:
                              try:
                                  stat_l, p_l = stats.mannwhitneyu(group0_loc, group1_loc, alternative='two-sided')
                                  print(f"    Mann-Whitney U ({group_key} - Local Eff): p = {p_l:.4f}") # Keep this print
                              except ValueError:
                                   print(f"    Mann-Whitney U failed for Local Eff ({group_key}).") # Keep this print
                          # else: print(...) # Removed logging

                 # else: print(...) # Removed logging
            # elif perform_comparison and current_run_labels is None: print(...) # Removed logging


    return all_efficiency_results


def detect_communities_on_average(
    run_wfc_data: Dict[str, Dict[str, np.ndarray]],
    parcel_names: List[str],
    network_indices: Dict[str, np.ndarray],
    threshold_percentile: float = 98.0,
    levels: int = 7 # Corresponds to Yeo 7 networks usually
) -> Dict[str, Dict[str, Any]]:
    """
    Calculates average WFC for each model/group, thresholds it, creates a graph,
    finds communities using Girvan-Newman, and stores results.

    Args:
        run_wfc_data: Dict[model_key][group_key] -> WFC vectors (n_runs, n_features).
                      Assumes WFC vectors contain non-negative weights suitable for thresholding.
        parcel_names: List of parcel names.
        network_indices: Dict mapping network names to parcel indices.
        threshold_percentile: Percentile to threshold average WFC (applied to positive values).
        levels: Girvan-Newman community detection levels (number of communities to find).

    Returns:
        Dictionary mapping 'model_key_group_key' (e.g., 'OVR_L1_logistic_HC')
        to analysis results:
        {'graphobj': GraphFromConnectivity object of the thresholded avg graph,
         'avg_wfc_vector': The original average WFC vector (before thresholding),
         'thresholded_avg_wfc': The thresholded average WFC vector,
         'communities': tuple of sets, where each set contains parcel names in a community,
         'error': str - present only if detection failed for this key}
    """
    community_results = {}

    if not run_wfc_data:
        logging.warning("Input 'run_wfc_data' is empty. Cannot detect communities.")
        return community_results

    for model_key, model_wfc_groups in run_wfc_data.items():
        logging.info(f"Detecting communities for model: {model_key}")
        for group_key, wfc_vectors in model_wfc_groups.items():
            # Create a unique key combining model and group for the results dictionary
            analysis_key = f"{model_key}_{group_key}"
            logging.info(f"  Processing group: {group_key} -> Analysis Key: {analysis_key}")

            # Initialize result dict for this key
            community_results[analysis_key] = {}

            if wfc_vectors.size == 0 or wfc_vectors.ndim != 2:
                logging.warning(f"    Skipping community detection for {analysis_key}: Invalid or empty WFC data (shape: {wfc_vectors.shape}).")
                community_results[analysis_key]['error'] = "Invalid or empty WFC data"
                continue

            try:
                # 1. Calculate Average WFC Vector
                avg_wfc_vector = np.mean(wfc_vectors, axis=0)
                community_results[analysis_key]['avg_wfc_vector'] = avg_wfc_vector

                # 2. Threshold the Average WFC Vector
                # Apply thresholding only to positive values to avoid issues with zeros/negatives
                positive_avg_wfc = avg_wfc_vector[avg_wfc_vector > 1e-9] # Small epsilon for floating point
                thresholded_avg_wfc = np.zeros_like(avg_wfc_vector) # Start with zeros

                if len(positive_avg_wfc) > 0:
                    # Calculate threshold value based on the specified percentile of positive values
                    threshold_value = np.percentile(positive_avg_wfc, threshold_percentile)
                    # Apply threshold: Keep values >= threshold, set others to 0
                    thresholded_avg_wfc[avg_wfc_vector >= threshold_value] = avg_wfc_vector[avg_wfc_vector >= threshold_value]
                    logging.info(f"    Applied {threshold_percentile}% threshold (value >= {threshold_value:.4f}) to average positive WFC.")
                else:
                    logging.warning(f"    No positive average WFC values found for {analysis_key}. Thresholded graph will have no edges.")

                community_results[analysis_key]['thresholded_avg_wfc'] = thresholded_avg_wfc

                # 3. Create Graph Object from Thresholded Average WFC
                # GraphFromConnectivity handles conversion from vector to matrix and graph creation
                graph_obj = GraphFromConnectivity(thresholded_avg_wfc, parcel_names, network_indices)
                community_results[analysis_key]['graphobj'] = graph_obj

                # 4. Find Communities using Girvan-Newman
                logging.info(f"    Running Girvan-Newman community detection (levels={levels})...")
                communities = graph_obj.find_communities(levels=levels) # Tuple of sets
                community_results[analysis_key]['communities'] = communities
                logging.info(f"    Detected {len(communities)} communities for {analysis_key} at level {levels}.")

            except Exception as e:
                 # Catch any error during the process for this specific model/group
                 logging.error(f"    Community detection process failed for {analysis_key}: {e}", exc_info=True)
                 community_results[analysis_key]['error'] = str(e) # Store error message

    return community_results