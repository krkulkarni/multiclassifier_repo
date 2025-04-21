# New function - Place this in src/data_loading.py (or similar)
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from nilearn.image import load_img
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt # For PCA scree plot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_connectivity_data_for_group(
    group_df: pd.DataFrame,
    data_dir: Path,
    file_pattern: str,
    group_name: str # For logging
) -> tuple:
    """
    Loads connectivity matrices for subjects in a given group DataFrame.

    Args:
        group_df: DataFrame filtered for the specific group, containing
                  'subject', 'label', 'training' columns.
        data_dir: Path to the directory containing the .npy files.
        file_pattern: An f-string pattern for the filename, e.g.,
                      "sub-{sub}_run-{run}_corr-100.npy" or
                      "{sub}_residuals_run_{run}_connectivity_matrix_100.npy".
                      Must contain {sub} and {run}.
        group_name: Name of the group for logging purposes.

    Returns:
        A tuple containing:
        - fcs (np.ndarray): Array of flattened upper triangle connectivity vectors.
        - labels (np.ndarray): Array of labels.
        - sub_names (np.ndarray): Array of subject IDs corresponding to each FC vector.
        - training_flags (np.ndarray): Array of training flags (0 or 1).
    """
    fcs_matrices = []
    labels = []
    sub_names = []
    training_flags = []
    runs = [] # Keep track of run ids if needed later, currently unused after loading

    skipped_files = 0

    # Use iterrows for clarity, though itertuples is faster for large dfs
    for _, row in group_df.iterrows():
        sub = row['subject']
        label = row['label']
        training = row['training']

        # Load data for runs 1 and 2
        for r in range(1, 3): # Runs are 1 and 2
            try:
                # Construct filename using the provided pattern
                filename = file_pattern.format(sub=sub, run=r)
                file_path = data_dir / filename
                
                if file_path.exists():
                    fc_matrix = np.load(file_path)
                    # Check if matrix is square before taking upper triangle
                    if fc_matrix.ndim == 2 and fc_matrix.shape[0] == fc_matrix.shape[1]:
                        fcs_matrices.append(fc_matrix)
                        labels.append(label)
                        sub_names.append(sub) # Store subject ID for each run's data
                        training_flags.append(training)
                        runs.append(r)
                    else:
                        logging.warning(f"Skipping non-square matrix: {file_path} for {group_name} sub {sub} run {r}. Shape: {fc_matrix.shape}")
                        skipped_files += 1

                else:
                    # logging.info(f"File not found, skipping: {file_path} for {group_name} sub {sub} run {r}")
                    skipped_files += 1

            except Exception as e:
                logging.error(f"Error loading file {file_path} for {group_name} sub {sub} run {r}: {e}")
                skipped_files += 1
                continue

    if not fcs_matrices:
         logging.warning(f"No valid connectivity matrices loaded for group: {group_name}")
         # Return empty arrays with correct dimensions to avoid errors later
         return np.array([]).reshape(0,0), np.array([]), np.array([]), np.array([])


    logging.info(f"Loaded {len(fcs_matrices)} matrices for group {group_name}. Skipped {skipped_files} files.")

    # Convert to upper triangle vectors
    # Ensure all matrices have the same shape before proceeding
    first_shape = fcs_matrices[0].shape
    if not all(m.shape == first_shape for m in fcs_matrices):
        logging.error(f"Inconsistent matrix shapes found within group {group_name}. Cannot compute upper triangles reliably.")
        # Handle error appropriately - maybe raise exception or return empty
        raise ValueError(f"Inconsistent matrix shapes in group {group_name}")

    num_features = len(fcs_matrices[0][np.triu_indices(first_shape[0], k=1)])
    fcs_vectors = np.zeros((len(fcs_matrices), num_features))
    for i, fc_matrix in enumerate(fcs_matrices):
         fcs_vectors[i, :] = fc_matrix[np.triu_indices(first_shape[0], k=1)]

    return (
        fcs_vectors,
        np.array(labels),
        np.array(sub_names),
        np.array(training_flags),
    )


def load_and_split_fmri_data(
    config_df: pd.DataFrame,
    data_dir: Path,
    alc_file_pattern: str,
    can_file_pattern: str,
    output_npz_path: Path = None,
):
    """
    Loads, processes, and splits functional connectivity data for alcohol and
    cannabis groups based on a configuration DataFrame.

    Args:
        config_df: DataFrame with subject info ('subject', 'group', 'label',
                   'training', 'inclusion').
        data_dir: Path to the directory containing the raw .npy connectivity files.
        alc_file_pattern: F-string pattern for alcohol subject files (needs {sub}, {run}).
        can_file_pattern: F-string pattern for cannabis subject files (needs {sub}, {run}).
        output_npz_path: Optional path to save the processed data (before splitting)
                         as an .npz file.

    Returns:
        A tuple containing three tuples:
        - Aggregated data: (X_train, X_test, y_train, y_test, train_ids, test_ids)
        - Alcohol data: (alc_X_train, alc_X_test, alc_y_train, alc_y_test, alc_train_ids, alc_test_ids)
        - Cannabis data: (can_X_train, can_X_test, can_y_train, can_y_test, can_train_ids, can_test_ids)
    """

    # --- 1. Load Alcohol Data ---
    logging.info("Loading Alcohol group data...")
    alcohol_df = config_df[config_df["group"] == "alcohol"].copy()
    if alcohol_df.empty:
        logging.warning("No subjects found for the 'alcohol' group in the config DataFrame.")
        # Initialize empty arrays if no subjects
        (alc_fcs, alc_labels, alc_sub_names, alc_training_flags) = (np.array([]).reshape(0,0), np.array([]), np.array([]), np.array([]))
    else:
        (alc_fcs, alc_labels, alc_sub_names, alc_training_flags) = \
            _load_connectivity_data_for_group(
                alcohol_df, data_dir, alc_file_pattern, "Alcohol"
            )

    # --- 2. Load Cannabis Data ---
    logging.info("Loading Cannabis group data...")
    # Apply inclusion filter *before* loading
    cannabis_df = config_df[
        (config_df["group"] == "cannabis") & (config_df["inclusion"] == 1)
    ].copy()
    if cannabis_df.empty:
         logging.warning("No subjects found for the 'cannabis' group after inclusion filter.")
         # Initialize empty arrays if no subjects
         (can_fcs, can_labels, can_sub_names, can_training_flags) = (np.array([]).reshape(0,0), np.array([]), np.array([]), np.array([]))
    else:
        (can_fcs, can_labels, can_sub_names, can_training_flags) = \
            _load_connectivity_data_for_group(
                cannabis_df, data_dir, can_file_pattern, "Cannabis"
            )

    # --- 3. Optional: Save combined loaded data before splitting ---
    if output_npz_path:
        try:
            output_npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_npz_path,
                alcohol_fcs=alc_fcs,
                alcohol_labels=alc_labels,
                alcohol_sub_names=alc_sub_names,
                alcohol_training=alc_training_flags,
                cannabis_fcs=can_fcs,
                cannabis_labels=can_labels,
                cannabis_sub_names=can_sub_names,
                cannabis_training=can_training_flags,
            )
            logging.info(f"Saved processed data to: {output_npz_path}")
        except Exception as e:
            logging.error(f"Failed to save data to {output_npz_path}: {e}")


    # --- 4. Split data into Train/Test ---
    # Convert training flags to boolean masks
    alc_train_mask = alc_training_flags.astype(bool)
    can_train_mask = can_training_flags.astype(bool)

    # Handle cases where a group might have no data
    alc_X_train = alc_fcs[alc_train_mask] if alc_fcs.size > 0 else np.array([]).reshape(0, alc_fcs.shape[1] if alc_fcs.ndim > 1 else 0)
    alc_y_train = alc_labels[alc_train_mask] if alc_labels.size > 0 else np.array([])
    alc_train_ids = alc_sub_names[alc_train_mask] if alc_sub_names.size > 0 else np.array([])
    alc_X_test = alc_fcs[~alc_train_mask] if alc_fcs.size > 0 else np.array([]).reshape(0, alc_fcs.shape[1] if alc_fcs.ndim > 1 else 0)
    alc_y_test = alc_labels[~alc_train_mask] if alc_labels.size > 0 else np.array([])
    alc_test_ids = alc_sub_names[~alc_train_mask] if alc_sub_names.size > 0 else np.array([])


    can_X_train = can_fcs[can_train_mask] if can_fcs.size > 0 else np.array([]).reshape(0, can_fcs.shape[1] if can_fcs.ndim > 1 else 0)
    can_y_train = can_labels[can_train_mask] if can_labels.size > 0 else np.array([])
    can_train_ids = can_sub_names[can_train_mask] if can_sub_names.size > 0 else np.array([])
    can_X_test = can_fcs[~can_train_mask] if can_fcs.size > 0 else np.array([]).reshape(0, can_fcs.shape[1] if can_fcs.ndim > 1 else 0)
    can_y_test = can_labels[~can_train_mask] if can_labels.size > 0 else np.array([])
    can_test_ids = can_sub_names[~can_train_mask] if can_sub_names.size > 0 else np.array([])


    # --- 5. Aggregate Train/Test Sets ---
    # Check if both groups have training data before concatenating
    if alc_X_train.size > 0 and can_X_train.size > 0:
        X_train = np.concatenate((alc_X_train, can_X_train), axis=0)
        y_train = np.concatenate((alc_y_train, can_y_train), axis=0)
        # Original code prefixes IDs, let's replicate that
        train_ids = np.array(
            ["ah-" + str(a) for a in alc_train_ids] + ["cb-" + str(a) for a in can_train_ids]
            )
    elif alc_X_train.size > 0: # Only alcohol has training data
        X_train, y_train = alc_X_train, alc_y_train
        train_ids = np.array(["ah-" + str(a) for a in alc_train_ids])
    elif can_X_train.size > 0: # Only cannabis has training data
         X_train, y_train = can_X_train, can_y_train
         train_ids = np.array(["cb-" + str(a) for a in can_train_ids])
    else: # Neither group has training data
        X_train = np.array([]).reshape(0, alc_X_train.shape[1] if alc_X_train.ndim > 1 else (can_X_train.shape[1] if can_X_train.ndim > 1 else 0))
        y_train = np.array([])
        train_ids = np.array([])


    # Check if both groups have testing data before concatenating
    if alc_X_test.size > 0 and can_X_test.size > 0:
        X_test = np.concatenate((alc_X_test, can_X_test), axis=0)
        y_test = np.concatenate((alc_y_test, can_y_test), axis=0)
        test_ids = np.array(
            ["ah-" + str(a) for a in alc_test_ids] + ["cb-" + str(a) for a in can_test_ids]
        )
    elif alc_X_test.size > 0: # Only alcohol has test data
         X_test, y_test = alc_X_test, alc_y_test
         test_ids = np.array(["ah-" + str(a) for a in alc_test_ids])
    elif can_X_test.size > 0: # Only cannabis has test data
        X_test, y_test = can_X_test, can_y_test
        test_ids = np.array(["cb-" + str(a) for a in can_test_ids])
    else: # Neither group has test data
        X_test = np.array([]).reshape(0, alc_X_test.shape[1] if alc_X_test.ndim > 1 else (can_X_test.shape[1] if can_X_test.ndim > 1 else 0))
        y_test = np.array([])
        test_ids = np.array([])


    # --- 6. Print shapes for verification ---
    logging.info(f"Final Shapes - Aggregated: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}, train_ids={train_ids.shape}, test_ids={test_ids.shape}")
    logging.info(f"Final Shapes - Alcohol: alc_X_train={alc_X_train.shape}, alc_y_train={alc_y_train.shape}, alc_X_test={alc_X_test.shape}, alc_y_test={alc_y_test.shape}, alc_train_ids={alc_train_ids.shape}, alc_test_ids={alc_test_ids.shape}")
    logging.info(f"Final Shapes - Cannabis: can_X_train={can_X_train.shape}, can_y_train={can_y_train.shape}, can_X_test={can_X_test.shape}, can_y_test={can_y_test.shape}, can_train_ids={can_train_ids.shape}, can_test_ids={can_test_ids.shape}")

    # --- 7. Return data in the expected nested tuple format ---
    return (
        (X_train, X_test, y_train, y_test, train_ids, test_ids),
        (alc_X_train, alc_X_test, alc_y_train, alc_y_test, alc_train_ids, alc_test_ids),
        (can_X_train, can_X_test, can_y_train, can_y_test, can_train_ids, can_test_ids),
    )

def load_activation_maps(
    subjects_df: pd.DataFrame,
    group_name: str,
    base_data_dir: Path,
    contrast_name: str,
    sub_col: str = 'subject',
    label_col: str = 'label',
    train_col: str = 'training'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Loads and flattens activation maps (.nii.gz) for subjects in a specific group.

    Args:
        subjects_df: DataFrame containing subject IDs and metadata.
        group_name: The name of the group to filter subjects ('alcohol' or 'cannabis').
        base_data_dir: Path to the base directory containing group-specific GLM output folders.
        contrast_name: Name of the contrast file (e.g., 'tasteMinusWashout').
        sub_col: Column name for subject IDs in subjects_df.
        label_col: Column name for labels in subjects_df.
        train_col: Column name for training flag (0/1 or bool) in subjects_df.

    Returns:
        Tuple containing:
        - img_matrix (np.ndarray or None): Matrix of flattened image data (n_runs, n_voxels).
        - labels (np.ndarray or None): Labels for each run.
        - run_ids (np.ndarray or None): Run index (0 or 1) for each image.
        - sub_names (np.ndarray or None): Subject ID for each run.
        - training_flags (np.ndarray or None): Training flag (boolean) for each run.
        - missing_runs (List[str]): List of subject_run strings that were missing.
        Returns None for arrays if loading fails for the group.
    """
    logging.info(f"Loading activation maps for group: {group_name}, contrast: {contrast_name}")
    group_df = subjects_df[subjects_df['group'] == group_name].copy()
    if group_df.empty:
        logging.warning(f"No subjects found for group '{group_name}'.")
        return None, None, None, None, None, []

    glm_output_dir = base_data_dir / f"{group_name}_glm_output"
    logging.info(f"  Looking for data in: {glm_output_dir}")

    img_data_list = []
    labels_list = []
    run_ids_list = []
    sub_names_list = []
    training_flags_list = []
    missing_runs = []

    # Check for required columns
    required_cols = [sub_col, label_col, train_col]
    if not all(col in group_df.columns for col in required_cols):
        logging.error(f"DataFrame missing required columns: {required_cols}")
        return None, None, None, None, None, []

    for _, row in group_df.iterrows():
        sub = row[sub_col]
        label = row[label_col]
        training = bool(row[train_col]) # Ensure boolean

        for r in range(1, 3): # Runs 1 and 2
            run_id_str = f"run-{r}"
            # Construct filename pattern (adjust if needed)
            # Example: data/raw/alcohol_glm_output/sub-M00000000_run-1/sub-M00000000_task-cuepres_contrast-tasteMinusWashout_stat-z_statmap.nii.gz
            file_path = glm_output_dir / f"sub-{sub}_{run_id_str}" / f"sub-{sub}_task-cuepres_contrast-{contrast_name}_stat-z_statmap.nii.gz"

            try:
                if file_path.is_file():
                     img = load_img(str(file_path))
                     img_flat = img.get_fdata().flatten()
                     img_data_list.append(img_flat)
                     labels_list.append(label)
                     run_ids_list.append(r - 1) # Store as 0 or 1
                     sub_names_list.append(sub)
                     training_flags_list.append(training)
                else:
                    # logging.warning(f"  File not found: {file_path}") # Can be verbose
                    missing_runs.append(f"{sub}_{run_id_str}")

            except Exception as e:
                 logging.error(f"  Error loading or processing {file_path}: {e}", exc_info=False) # Set exc_info=True for full traceback
                 missing_runs.append(f"{sub}_{run_id_str}")
                 continue # Skip to next run/subject on error

    if not img_data_list:
        logging.warning(f"No valid image data loaded for group '{group_name}'.")
        return None, None, None, None, None, missing_runs

    # Convert lists to numpy arrays
    img_matrix = np.vstack(img_data_list) # Stack flattened vectors into a matrix
    labels = np.array(labels_list)
    run_ids = np.array(run_ids_list)
    sub_names = np.array(sub_names_list)
    training_flags = np.array(training_flags_list)

    logging.info(f"  Loaded group '{group_name}'. Image matrix shape: {img_matrix.shape}")
    logging.info(f"  Missing runs ({len(missing_runs)}): {missing_runs[:5]}...") # Show first few missing

    return img_matrix, labels, run_ids, sub_names, training_flags, missing_runs


def run_pca_on_activation(
    img_matrix: np.ndarray,
    pca_type: str = 'full_rank', # 'full_rank', '70%var', '90%var', or integer n_components
    plot_scree: bool = True,
    standardize_before_pca: bool = True
) -> Optional[pd.DataFrame]:
    """
    Runs PCA on the activation data matrix.

    Args:
        img_matrix: Input data matrix (n_runs, n_voxels).
        pca_type: Method for selecting components ('full_rank', 'XX%var', int).
        plot_scree: Whether to generate a scree plot.
        standardize_before_pca: Whether to apply StandardScaler before PCA.

    Returns:
        DataFrame containing the reduced principal components, or None if PCA fails.
    """
    if img_matrix is None or img_matrix.size == 0:
        logging.error("Input image matrix is empty. Cannot run PCA.")
        return None

    logging.info(f"Running PCA (Type: {pca_type}, Standardize: {standardize_before_pca}) on matrix shape {img_matrix.shape}...")

    X = img_matrix
    if standardize_before_pca:
        logging.info("  Standardizing data before PCA...")
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            if np.any(~np.isfinite(X)):
                 logging.warning(" Non-finite values found after scaling. Replacing with 0.")
                 X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as scale_e:
             logging.error(f"  Standardization failed: {scale_e}. Proceeding without standardization.")
             X = img_matrix # Fallback to original

    # Determine number of components
    n_components_request: Optional[Union[int, float]] = None
    if isinstance(pca_type, int):
        n_components_request = min(pca_type, X.shape[0], X.shape[1]) # Cap components
        logging.info(f"  Using specified n_components: {n_components_request}")
    elif pca_type == 'full_rank':
        n_components_request = min(X.shape[0], X.shape[1]) # Max possible components
        logging.info(f"  Using full rank n_components: {n_components_request}")
    elif isinstance(pca_type, str) and pca_type.endswith('%var'):
        try:
            variance_threshold = float(pca_type[:-4]) / 100.0
            if not (0 < variance_threshold < 1): raise ValueError("Variance threshold must be between 0 and 100.")
            # Set n_components to variance threshold for PCA object
            n_components_request = variance_threshold
            logging.info(f"  Using n_components to explain {variance_threshold*100:.1f}% variance.")
        except ValueError as e:
            logging.error(f"  Invalid variance threshold format '{pca_type}': {e}. Using full rank.")
            n_components_request = min(X.shape[0], X.shape[1])
    else:
         logging.error(f"  Invalid pca_type '{pca_type}'. Using full rank.")
         n_components_request = min(X.shape[0], X.shape[1])


    try:
        pca = PCA(n_components=n_components_request, svd_solver='auto')
        reduced_matrix = pca.fit_transform(X)
        n_components_final = pca.n_components_ # Actual number of components kept

        logging.info(f"  PCA finished. Reduced matrix shape: {reduced_matrix.shape}")
        logging.info(f"  Explained variance by final {n_components_final} components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

        # Plotting
        if plot_scree:
            logging.info("  Generating Scree Plot...")
            PC_values = np.arange(n_components_final) + 1
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2, label='Individual Variance')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Proportion of Variance Explained', color='red')
            ax.tick_params(axis='y', labelcolor='red')
            ax.set_title('Scree Plot')

            ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis
            ax2.plot(PC_values, cumulative_variance, 'bo--', linewidth=2, label='Cumulative Variance')
            ax2.set_ylabel('Cumulative Proportion of Variance', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            # Add line/marker for variance threshold if applicable
            if isinstance(n_components_request, float):
                ax2.axhline(y=n_components_request, color='g', linestyle='--', label=f'{n_components_request*100:.0f}% Variance Threshold')
                # Mark the number of components chosen
                ax2.axvline(x=n_components_final, color='grey', linestyle=':', label=f'{n_components_final} Components')

            fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes)
            plt.grid(True, axis='y', linestyle=':')
            plt.show()


        return pd.DataFrame(data=reduced_matrix, columns=[f'PC{i+1}' for i in range(n_components_final)])

    except Exception as pca_e:
        logging.error(f"PCA failed: {pca_e}", exc_info=True)
        return None