# src/atlas_utils.py
import nilearn.datasets
import nilearn.image
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

def load_schaefer_atlas(
    n_rois: int = 100,
    yeo_networks: int = 7,
    resolution_mm: int = 2,
    data_dir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[object], Optional[List[str]], Optional[Dict[str, np.ndarray]]]:
    """
    Loads the Schaefer atlas using nilearn and extracts useful info.

    Args:
        n_rois: Number of ROIs (e.g., 100, 200, ...).
        yeo_networks: Number of Yeo networks (7 or 17).
        resolution_mm: Resolution (1 or 2).
        data_dir: Optional directory to store downloaded atlas.

    Returns:
        Tuple containing:
        - atlas_data: The raw object returned by nilearn.datasets.fetch_atlas_schaefer_2018.
        - parcel_names: Cleaned list of parcel names (e.g., 'Vis_1', 'SomMot_1').
        - network_indices: Dictionary mapping network names to parcel indices.
        Returns (None, None, None) on failure.
    """
    try:
        logging.info(f"Fetching Schaefer atlas: {n_rois} ROIs, {yeo_networks} networks, {resolution_mm}mm...")
        atlas_data = nilearn.datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=yeo_networks,
            resolution_mm=resolution_mm,
            data_dir=str(data_dir) if data_dir else None # Pass as string if Path provided
        )
        raw_labels = atlas_data['labels'].astype(str)
        # Clean names (e.g., remove '7Networks_' prefix)
        parcel_names = ['_'.join(label.split('_')[1:]) for label in raw_labels]
        logging.info(f"  Loaded {len(parcel_names)} parcel names.")

        # Create network indices mapping
        network_indices = {}
        for i, p_name in enumerate(parcel_names):
             try:
                 # Assumes format like 'Vis_1' after cleaning
                 network = p_name.split('_')[1]
                 if network in network_indices:
                     network_indices[network].append(i)
                 else:
                     network_indices[network] = [i]
             except IndexError:
                  logging.warning(f"Could not parse network from cleaned parcel name: '{p_name}'")
        # Convert lists to numpy arrays
        for network in network_indices:
            network_indices[network] = np.array(network_indices[network])
        logging.info(f"  Generated network indices for: {list(network_indices.keys())}")

        return atlas_data, parcel_names, network_indices

    except Exception as e:
        logging.error(f"Failed to load or process Schaefer atlas: {e}", exc_info=True)
        return None, None, None


def prepare_roi_images(atlas_image: object) -> List[Optional[object]]:
    """
    Creates a list of individual Nifti images, one for each ROI label in the atlas.

    Args:
        atlas_image: Nifti-like image object containing the atlas labels (e.g., from fetch_atlas_*).

    Returns:
        List of Nifti-like image objects, where each image is a binary mask for one ROI.
        Returns an empty list if input is invalid.
    """
    roi_images = []
    if not hasattr(atlas_image, 'get_fdata'):
        logging.error("Invalid atlas image provided for ROI extraction.")
        return roi_images

    logging.info("Preparing individual ROI Nifti images...")
    try:
        atlas_data = atlas_image.get_fdata()
        # Find unique non-zero label values (these are the ROI IDs)
        roi_labels = np.unique(atlas_data[atlas_data > 0]).astype(int)
        logging.info(f"  Found {len(roi_labels)} unique ROI labels in atlas.")

        for label_val in roi_labels:
            # Create binary mask for the current label
            # Use nilearn.image.math_img for safer image creation
            try:
                roi_img = nilearn.image.math_img(f"img == {label_val}", img=atlas_image)
                roi_images.append(roi_img)
            except Exception as math_e:
                 logging.error(f"    Failed to create image for ROI label {label_val}: {math_e}")
                 roi_images.append(None) # Append None on failure for this ROI

        logging.info(f"  Prepared {len([img for img in roi_images if img is not None])} valid ROI images.")
    except Exception as e:
        logging.error(f"Error during ROI image preparation: {e}", exc_info=True)
        return [] # Return empty list on major failure

    return roi_images