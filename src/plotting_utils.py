# src/plotting_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging # For informative messages
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import nilearn

# Assumes network_utils contains get_network_colors_numeric
from network_utils import get_network_colors_numeric

def get_network_rgba(network_indices: Dict[str, np.ndarray], n_rois: int, cmap_name: str = "Set3") -> Tuple[np.ndarray, List[str]]:
    """
    Generates RGBA colors for networks based on indices.

    Args:
        network_indices: Dict mapping network names to parcel indices.
        n_rois: Total number of ROIs.
        cmap_name: Name of the matplotlib colormap to use.

    Returns:
        Tuple containing:
        - network_rgba_tuple: RGBA values as a numpy array (N_ROIS, 4), scaled 0-1.
        - network_rgba_str: RGBA values as a list of strings 'rgba(r,g,b,a)'.
    """
    network_colors_numeric = get_network_colors_numeric(network_indices, n_rois)
    valid_indices = network_colors_numeric != -1
    min_val = 0
    max_val = np.max(network_colors_numeric) if np.any(valid_indices) else 0

    try:
        cmap = cm.get_cmap(cmap_name)
        my_norm = Normalize(vmin=min_val, vmax=max_val) # Normalize based on actual network indices used
        scalar_map = cm.ScalarMappable(norm=my_norm, cmap=cmap)

        # Assign colors only to valid indices, others remain default (e.g., grey or black)
        network_rgba_tuple = np.zeros((n_rois, 4))
        network_rgba_tuple[valid_indices] = scalar_map.to_rgba(network_colors_numeric[valid_indices])
        # Assign a default color (e.g., grey) to unassigned ROIs
        default_color = [0.5, 0.5, 0.5, 1.0] # Grey
        network_rgba_tuple[~valid_indices] = default_color

    except ValueError: # Handle invalid cmap_name
         print(f"Warning: Colormap '{cmap_name}' not found. Using default grey.")
         network_rgba_tuple = np.tile([0.5, 0.5, 0.5, 1.0], (n_rois, 1)) # Default to grey

    # Convert to integer RGBA strings for Plotly/other libraries if needed
    network_rgba_int = (network_rgba_tuple * 255).astype(int)
    network_rgba_str = [f"rgba({r},{g},{b},{a})" for r, g, b, a in network_rgba_int]

    return network_rgba_tuple, network_rgba_str # Return tuple (0-1 scale) and string list

def plot_permutation_distribution(
    perm_distribution: np.ndarray,
    observed_statistic: float,
    title: str = "Permutation Distribution",
    xlabel: str = "Statistic Value",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plots a histogram of the permutation distribution and the observed statistic."""
    if ax is None:
        fig, ax = plt.subplots()

    sns.histplot(perm_distribution, kde=True, ax=ax, stat="density", common_norm=False)
    ax.axvline(observed_statistic, 0, 0.95, color="black", linestyle='--', label=f'Observed ({observed_statistic:.3f})')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel)
    ax.legend()
    return ax

def plot_degree_centrality_violins(
    dc_values_dict: Dict[str, List[float]],
    selected_parcel_names: List[str],
    parcel_name_to_color: Optional[Dict[str, Tuple[float, float, float, float]]] = None, # Optional color mapping
    title: str = "Regional Degree Centrality",
    figure_path: Optional[Union[str, Path]] = None,
    y_label: str = "Degree Centrality",
    x_label: str = "Brain Region",
    figsize: Tuple[int, int] = (10, 6) # Default figure size
):
    """
    Generates a Seaborn violin plot for degree centrality distributions
    across selected parcels.

    Args:
        dc_values_dict: Dictionary mapping parcel names to a list of their DC values.
        selected_parcel_names: List of parcel names to include in the plot (defines x-axis order).
        parcel_name_to_color: Optional dictionary mapping parcel names to RGBA tuple (0-1). If provided,
                              used to color violins. Otherwise, uses default Seaborn palette.
        title: Title for the plot.
        figure_path: Optional path to save the figure (e.g., as .svg or .png).
        y_label: Label for the y-axis.
        x_label: Label for the x-axis.
        figsize: Tuple for figure size (width, height).
    """
    # Prepare data for Seaborn DataFrame
    plot_data = []
    for parcel_name in selected_parcel_names:
        dcs = dc_values_dict.get(parcel_name, [])
        if dcs: # Only include parcels with data
             for dc_val in dcs:
                 plot_data.append({'Parcel': parcel_name, 'DC': dc_val})
        else:
             logging.warning(f"No DC values found for parcel '{parcel_name}' for plotting.")

    if not plot_data:
        logging.warning("No data available to plot degree centrality violins.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Determine palette
    palette = None
    if parcel_name_to_color:
        # Create a mapping from the order of selected_parcel_names to their colors
        palette = {name: parcel_name_to_color.get(name, (0.5, 0.5, 0.5, 1.0)) # Default grey
                   for name in selected_parcel_names if name in df_plot['Parcel'].unique()}


    # Create the plot
    plt.figure(figsize=figsize) # Adjust width dynamically
    ax = sns.violinplot(
        data=df_plot,
        x='Parcel',
        y='DC',
        order=selected_parcel_names, # Ensure correct order on x-axis
        palette=palette,            # Use custom palette if provided
        inner='quartile',           # Show quartiles inside violin (alt: 'box', 'stick', None)
        cut=0,                      # Don't extend violins beyond data range
        width=5,                  # Width of the violins
    )

    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
    plt.tight_layout()
    sns.despine(ax=ax, left=True) # Remove left spine for cleaner look

    # Save the figure if path is provided
    if figure_path:
        try:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(fig_path), bbox_inches='tight')
            logging.info(f"Saved DC violin plot to {fig_path}")
        except Exception as e:
            logging.error(f"Failed to save DC violin plot to {figure_path}: {e}", exc_info=True)

    plt.show() # Display the plot


# Modify the plotting part of plot_efficiency_violins_seaborn in src/plotting_utils.py

def plot_efficiency_violins(
    efficiency_df: pd.DataFrame,
    group_labels: Optional[List[str]] = None, # Names for labels 0 and 1
    title_prefix: str = "",
    figure_dir: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    group_name: str = "group",
    figsize: Tuple[int, int] = (10, 5), # Adjusted default size
    point_size: int = 3, # Size for stripplot points
    point_alpha: float = 0.6, # Alpha for stripplot points
    violin_linewidth: float = 1.5, # Linewidth for violin outline
    palette: Optional[Union[str, List[str]]] = "Set2" # Seaborn palette
):
    """
    Generates Seaborn violin plots (outline only) with overlaid stripplots
    comparing global and local efficiency between two groups.
    # ... (rest of docstring) ...
    """
    # --- Start of function ---
    # ... (Initial checks, data cleaning, mapping labels to 'Group' column - same as before) ...
    if efficiency_df.empty or 'label' not in efficiency_df.columns:
        logging.warning("Efficiency DataFrame is empty or missing 'label' column. Cannot plot.")
        return
    eff_df_cleaned = efficiency_df.dropna(subset=['global_efficiency', 'local_efficiency', 'label']).copy()
    unique_labels = eff_df_cleaned['label'].unique()
    if len(unique_labels) != 2:
        logging.warning(f"Efficiency DataFrame contains {len(unique_labels)} unique non-NaN labels after cleaning. Need exactly 2 for comparison plot.")
        return
    label_map = {
        unique_labels[0]: group_labels[0] if group_labels and len(group_labels)>0 else f"Label {int(unique_labels[0])}",
        unique_labels[1]: group_labels[1] if group_labels and len(group_labels)>1 else f"Label {int(unique_labels[1])}"
    }
    eff_df_cleaned['Group'] = eff_df_cleaned['label'].map(label_map)
    plot_order = list(label_map.values()) # Order for x-axis

    # --- Modified Plotting Section ---
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'{title_prefix} Efficiency Comparison', fontsize=16, fontweight='bold')

    metrics_to_plot = [('Global Efficiency', 'global_efficiency'), ('Local Efficiency', 'local_efficiency')]

    for i, (title, metric_col) in enumerate(metrics_to_plot):
        ax = axes[i]
        # 1. Draw the violin outlines
        sns.violinplot(
            data=eff_df_cleaned, x='Group', y=metric_col, ax=ax,
            order=plot_order,
            palette=palette,
            inner=None,          # <<<--- No inner elements (box, stick, etc.)
            linewidth=violin_linewidth, # <<<--- Control outline width
            cut=0,
            saturation=0.7       # Slightly desaturate violin colors
        )
        # Ensure the violin outlines are not filled
        for collection in ax.collections:
             collection.set_facecolor('none') # Set facecolor to none (alternative depends on seaborn version)
             # Or collection.set_alpha(0.3) # Make fill very transparent

        # 2. Overlay the stripplot
        sns.stripplot(
            data=eff_df_cleaned, x='Group', y=metric_col, ax=ax,
            order=plot_order,
            palette=palette,
            size=point_size,         # <<<--- Control point size
            alpha=point_alpha,       # <<<--- Control point transparency
            jitter=0.2               # Add some jitter to prevent overlap
        )

        # Formatting
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Efficiency Value' if i == 0 else '') # Only label y-axis on the first plot
        sns.despine(ax=ax) # Remove top and right spines

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # --- Saving Section (same as before) ---
    if figure_dir:
        try:
            figure_dir = Path(figure_dir)
            figure_dir.mkdir(parents=True, exist_ok=True)
            filename = f"efficiency_violin_{model_name.replace('OVR_', '').replace('OVO_', '')}_{group_name}.svg" # Cleaned filename
            file_path = figure_dir / filename
            fig.savefig(str(file_path), bbox_inches='tight')
            logging.info(f"Saved efficiency plot to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save efficiency plot to {file_path}: {e}", exc_info=True)

    plt.show()


def plot_connectivity_heatmap(
    matrix: np.ndarray,
    reordered_indices: Optional[np.ndarray] = None,
    title: str = "Connectivity Heatmap",
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    figure_path: Optional[Union[str, Path]] = None,
    **heatmap_kwargs # Pass extra args to sns.heatmap
):
    """Plots a connectivity matrix heatmap, optionally reordered."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        logging.error("Input matrix must be square 2D.")
        return None

    plot_matrix = matrix
    if reordered_indices is not None:
         if len(reordered_indices) == matrix.shape[0]:
              plot_matrix = matrix[np.ix_(reordered_indices, reordered_indices)]
         else:
              logging.warning("Length of reordered_indices does not match matrix dimension. Plotting original order.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6)) # Create a figure if no axis provided
    else:
        fig = ax.get_figure() # Get figure from axis

    sns.heatmap(plot_matrix, square=True, cmap=cmap, ax=ax, **heatmap_kwargs)
    ax.set_title(title)
    ax.set_xticks([]) # Often hide ticks for connectivity matrices
    ax.set_yticks([])

    if figure_path:
        try:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fig_path), bbox_inches='tight')
            logging.info(f"Saved heatmap plot to {fig_path}")
        except Exception as e:
            logging.error(f"Failed to save heatmap plot to {figure_path}: {e}", exc_info=True)
    return ax


def plot_community_connectome(
    adjacency_matrix: np.ndarray,
    node_coords: np.ndarray,
    node_colors: Optional[np.ndarray] = 'black', # Expects RGBA tuple array (N, 4) or color string
    title: str = "Community Connectome",
    ax: Optional[plt.Axes] = None,
    node_size: float = 50,
    display_mode: str = 'lyrz',
    figure_path: Optional[Union[str, Path]] = None,
    **connectome_kwargs # Pass extra args to nilearn.plotting.plot_connectome
):
    """Plots the connectome for a specific community."""
    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        logging.error("Adjacency matrix must be square 2D.")
        return None
    if node_coords.shape[0] != adjacency_matrix.shape[0]:
        logging.error("Number of node coordinates must match adjacency matrix dimension.")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6)) # Create a figure if no axis provided
        # Note: plot_connectome can modify the figure, passing axes is preferred
    else:
        fig = ax.get_figure()

    try:
        # plot_connectome sometimes raises errors with single nodes or specific inputs
        if adjacency_matrix.shape[0] > 1:
             nilearn.plotting.plot_connectome(
                 adjacency_matrix=adjacency_matrix,
                 node_coords=node_coords,
                 node_color=node_colors,
                 node_size=node_size,
                 display_mode=display_mode,
                 axes=ax,
                 title=title,
                 **connectome_kwargs
             )
        elif adjacency_matrix.shape[0] == 1:
             # Handle single node case - just plot the node
             ax.scatter(node_coords[0, 0], node_coords[0, 1], c=[node_colors[0]] if isinstance(node_colors, np.ndarray) else node_colors, s=node_size*2)
             ax.set_title(title + " (1 Node)")
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_frame_on(False)
        else: # No nodes
             ax.set_title(title + " (0 Nodes)")
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_frame_on(False)


        if figure_path:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            # Need to save the *figure* that the axes belong to
            fig.savefig(str(fig_path), bbox_inches='tight', facecolor='k' if 'black_bg' in connectome_kwargs and connectome_kwargs['black_bg'] else 'w')
            logging.info(f"Saved connectome plot to {fig_path}")
    except Exception as e:
         logging.error(f"Failed to plot or save connectome for '{title}': {e}", exc_info=True)

    return ax # Return the axes object

# Add this function to src/plotting_utils.py

# Ensure these imports are present at the top of the file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

def plot_network_radar(
    dc_data_per_group: Dict[str, Dict[str, float]],
    parcel_names: List[str],
    network_indices: Dict[str, np.ndarray],
    network_order: List[str],
    group_colors: Optional[Dict[str, str]] = None,
    title: str = "Average Degree Centrality by Network",
    figsize: Tuple[int, int] = (8, 8),
    y_limit: Optional[Tuple[float, float]] = None,
    figure_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None # Allow passing existing polar axes
):
    """
    Creates a radar plot comparing average Degree Centrality (or other metric)
    across specified brain networks for different groups.

    Args:
        dc_data_per_group: Dictionary where keys are group names (e.g., 'HC', 'Alcohol', 'Cannabis')
                           and values are dictionaries mapping parcel names to their average DC score
                           for that group. Example: {'HC': {'Vis_1': 0.1, ...}, 'Alcohol': {...}}
        parcel_names: List of all parcel names in the atlas order.
        network_indices: Dictionary mapping network names (e.g., 'Vis', 'Default')
                         to numpy arrays of parcel indices belonging to that network.
        network_order: List of network names in the desired order for the radar plot axes.
        group_colors: Optional dictionary mapping group names to color strings.
                      Defaults will be used if None.
        title: Title for the plot.
        figsize: Figure size if creating a new figure.
        y_limit: Optional tuple (min, max) for the radial axis limits. If None, calculated automatically.
        figure_path: Optional path to save the figure.
        ax: Optional Matplotlib polar axes object to plot on. If None, a new figure/axes is created.

    Returns:
        The matplotlib Axes object containing the plot.
    """
    # --- 1. Data Aggregation: Calculate Mean DC per Network per Group ---
    n_networks = len(network_order)
    if n_networks < 3:
        logging.error("Radar plot requires at least 3 categories (networks).")
        return None

    aggregated_data = {group: {} for group in dc_data_per_group}
    parcel_to_network = {} # Create mapping for easier lookup
    for net_name, indices in network_indices.items():
        for idx in indices:
             if idx < len(parcel_names):
                  parcel_to_network[parcel_names[idx]] = net_name

    max_dc_value = 0 # To dynamically set y-limits if not provided

    for network_name in network_order:
        # Find parcels belonging to this network based on the provided mapping
        parcels_in_network = [p_name for p_name, net in parcel_to_network.items() if net == network_name]

        if not parcels_in_network:
             logging.warning(f"No parcels found for network '{network_name}'. Assigning NaN.")
             for group in aggregated_data:
                 aggregated_data[group][network_name] = np.nan
             continue

        for group, parcel_dcs in dc_data_per_group.items():
            # Get DC values for parcels in this network for this group
            network_dcs = [parcel_dcs.get(p_name, np.nan) for p_name in parcels_in_network]
            # Filter out NaNs before calculating mean
            valid_dcs = [dc for dc in network_dcs if not np.isnan(dc)]

            if valid_dcs:
                mean_dc = np.mean(valid_dcs)
                aggregated_data[group][network_name] = mean_dc
                max_dc_value = max(max_dc_value, mean_dc) # Track max for ylim
            else:
                logging.warning(f"No valid DC scores found for group '{group}' in network '{network_name}'. Assigning NaN.")
                aggregated_data[group][network_name] = np.nan

    # Convert aggregated data to DataFrame for easier plotting access
    df_plot = pd.DataFrame(aggregated_data)
    # Reindex rows to match the specified network_order
    df_plot = df_plot.reindex(network_order)

    # --- 2. Setup for Plotting ---
    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, n_networks, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    # Define default colors if not provided
    default_colors = {'HC': 'purple', 'Alcohol': 'red', 'Cannabis': 'blue'}
    if group_colors is None:
        group_colors = default_colors
    else:
        # Ensure all groups have a color, use default if missing
        for group in df_plot.columns:
            if group not in group_colors:
                group_colors[group] = default_colors.get(group, plt.cm.get_cmap('tab10')(len(group_colors) % 10))


    # --- 3. Create Polar Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    else:
        if not ax.name == 'polar':
            logging.error("Provided axes object is not polar. Cannot create radar plot.")
            return None
        fig = ax.get_figure()

    # Set y-axis limits
    if y_limit:
        y_min, y_max = y_limit
    else:
        y_min = 0
        # Add padding to the max value found in data
        y_max = max_dc_value * 1.15 if max_dc_value > 0 else 0.01 # Ensure some range if max is 0
    ax.set_ylim(y_min, y_max)

    # Set radial axis ticks (example: 4 ticks) - Adjust number and formatting as needed
    yticks = np.linspace(y_min, y_max, 4, endpoint=True)
    ax.set_yticks(yticks)
    # Format tick labels (e.g., 3 decimal places)
    ax.set_yticklabels([f"{tick:.3f}" for tick in yticks])
    ax.tick_params(axis='y', labelsize=10)


    # --- 4. Plot Data for Each Group ---
    for group in df_plot.columns:
        values = df_plot[group].values.flatten().tolist()
        # Handle potential NaNs - replace with 0 for plotting shape, or handle differently? Let's use 0.
        values_plot = [v if not np.isnan(v) else 0 for v in values]
        values_plot += values_plot[:1] # Complete the loop

        color = group_colors.get(group, 'grey') # Fallback color

        # Plot lines
        ax.plot(angles, values_plot, linewidth=1.5, linestyle='solid', label=group, color=color, zorder=3) # Ensure lines on top
        # Fill areas
        ax.fill(angles, values_plot, color=color, alpha=0.25)


    # --- 5. Customize Axes ---
    # Set network labels on the axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(network_order)
    ax.tick_params(axis='x', labelsize=12) # Adjust category label size

    # Set the position of the radial labels (0=top, 90=right, 180=bottom)
    ax.set_rlabel_position(30)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- 6. Title and Legend ---
    ax.set_title(title, size=16, y=1.1) # Adjust title position
    # Place legend outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))


    # --- 7. Save Figure ---
    if figure_path:
        try:
            fig_path = Path(figure_path)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fig_path), bbox_inches='tight')
            logging.info(f"Saved radar plot to {fig_path}")
        except Exception as e:
            logging.error(f"Failed to save radar plot to {figure_path}: {e}", exc_info=True)

    return ax