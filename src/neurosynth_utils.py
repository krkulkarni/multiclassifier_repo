# src/neurosynth_utils.py
import nilearn.image
import nilearn.plotting as nplotting
from nilearn.maskers import NiftiMasker
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Set, Optional, Union, Tuple

def plot_roi_neurosynth_overlap(
    neurosynth_map_path: Union[str, Path],
    all_roi_images: List[Optional[object]], # List of Nifti images for each ROI
    all_parcel_names: List[str],
    chosen_roi_names: List[str], # List of names for ROIs to highlight
    output_figure_path: Union[str, Path],
    bg_img: Optional[Union[str, object]] = 'MNI152', # Background anatomical
    display_mode: str = 'x',
    cut_coords: int = 7,
    cmap_neurosynth: object = nplotting.cm.black_red,
    cmap_chosen: object = nplotting.cm.black_green,
    cmap_other: object = nplotting.cm.brown_blue,
    linewidth: float = 0.5,
    title: Optional[str] = None
):
    """
    Visualizes overlap between chosen ROIs and a Neurosynth map.

    Args:
        neurosynth_map_path: Path to the Neurosynth statistical map (.nii.gz).
        all_roi_images: List containing Nifti images for ALL ROIs in the atlas. Order
                        must match all_parcel_names. Can contain None for invalid ROIs.
        all_parcel_names: List of ALL parcel names in the atlas. Order must match all_roi_images.
        chosen_roi_names: List of parcel names specifically chosen for highlighting.
        output_figure_path: Path to save the output figure (e.g., .svg, .png).
        bg_img: Background anatomical image (path or nilearn identifier).
        display_mode: Nilearn display mode ('x', 'y', 'z', 'ortho', etc.).
        cut_coords: Number of cuts or specific coordinates for display.
        cmap_neurosynth: Nilearn colormap for the Neurosynth map contours.
        cmap_chosen: Nilearn colormap for the chosen ROI contours.
        cmap_other: Nilearn colormap for the other ROI contours.
        linewidth: Linewidth for contours.
        title: Optional title for the plot.
    """
    output_figure_path = Path(output_figure_path)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    neurosynth_map_path = Path(neurosynth_map_path)

    if not neurosynth_map_path.is_file():
        logging.error(f"Neurosynth map not found: {neurosynth_map_path}")
        return
    if len(all_roi_images) != len(all_parcel_names):
         logging.error("Mismatch between number of ROI images and parcel names.")
         return

    logging.info(f"Generating overlap plot: {output_figure_path.name}")
    display = None # Initialize display object
    try:
        # Load background and Neurosynth map
        # anat_img = nilearn.image.load_img(bg_img) if isinstance(bg_img, (str, Path)) else bg_img
        ns_map_img = nilearn.image.load_img(str(neurosynth_map_path))

        # Start plotting
        display = nplotting.plot_anat(display_mode=display_mode, cut_coords=cut_coords, title=title, draw_cross=False)

        # Add Neurosynth map contours first (red)
        display.add_contours(ns_map_img, cmap=cmap_neurosynth, linewidths=linewidth)

        # Add ROI contours
        chosen_roi_names_set = set(chosen_roi_names) # For faster lookup
        plotted_chosen = 0
        plotted_other = 0
        for roi_name, roi_img in zip(all_parcel_names, all_roi_images):
            if roi_img is None: continue # Skip if ROI image failed to create

            if roi_name in chosen_roi_names_set:
                display.add_contours(roi_img, cmap=cmap_chosen, linewidths=linewidth)
                plotted_chosen += 1
            else:
                display.add_contours(roi_img, cmap=cmap_other, linewidths=linewidth)
                plotted_other += 1

        logging.info(f"  Plotted {plotted_chosen} chosen ROIs (green) and {plotted_other} other ROIs (blue).")
        display.savefig(str(output_figure_path))
        logging.info(f"  Saved plot to {output_figure_path}")

    except Exception as e:
        logging.error(f"Failed to generate or save plot {output_figure_path.name}: {e}", exc_info=True)
    # finally:
    #     # Close the display object to free memory, especially in loops
    #     if display is not None:
    #         try: display.close()
    #         except: pass # Ignore errors during close


def calculate_roi_neurosynth_overlap(
    neurosynth_map_path: Union[str, Path],
    all_roi_images: List[Optional[object]],
    all_parcel_names: List[str],
    chosen_roi_names: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Calculates which chosen ROIs overlap with positive values in a Neurosynth map.

    Args:
        neurosynth_map_path: Path to the Neurosynth statistical map (.nii.gz).
        all_roi_images: List containing Nifti images (binary masks) for ALL ROIs.
        all_parcel_names: List of ALL parcel names.
        chosen_roi_names: List of parcel names specifically chosen for analysis.

    Returns:
        Tuple containing:
        - overlap_rois: List of chosen ROI names that overlap with the map.
        - non_overlap_rois: List of chosen ROI names that do not overlap.
    """
    overlap_rois = []
    non_overlap_rois = []
    neurosynth_map_path = Path(neurosynth_map_path)

    if not neurosynth_map_path.is_file():
        logging.error(f"Neurosynth map not found: {neurosynth_map_path}")
        return [], []
    if len(all_roi_images) != len(all_parcel_names):
         logging.error("Mismatch between number of ROI images and parcel names.")
         return [], []

    logging.info(f"Calculating overlap for {len(chosen_roi_names)} chosen ROIs with {neurosynth_map_path.name}...")
    try:
        # Load neurosynth map data once
        ns_map_img = nilearn.image.load_img(str(neurosynth_map_path))

        chosen_roi_names_set = set(chosen_roi_names) # Use set for efficient checking
        for roi_name, roi_img in zip(all_parcel_names, all_roi_images):
            if roi_name in chosen_roi_names_set:
                 if roi_img is None:
                      logging.warning(f"  Skipping ROI '{roi_name}': Missing image data.")
                      non_overlap_rois.append(roi_name) # Count as non-overlapping if image missing
                      continue

                 try:
                     # Create a masker for this specific ROI
                     # Ensure labels_img and background_label=0 if using NiftiLabelsMasker
                     # Using NiftiMasker requires the ROI image to be a valid mask
                     masker = NiftiMasker(mask_img=roi_img, standardize=False)
                     # Extract data from the neurosynth map within this ROI mask
                     masked_data = masker.fit_transform(ns_map_img)

                     # Check if the mean of the extracted (positive?) values is > 0
                     # Original code used np.mean(masked_data) > 0
                     # Consider if thresholding NS map first is desired, or just check positive mean
                     if masked_data.size > 0 and np.mean(masked_data) > 1e-6: # Check mean is positive (small epsilon)
                         overlap_rois.append(roi_name)
                     else:
                         non_overlap_rois.append(roi_name)

                 except Exception as mask_e:
                      logging.error(f"  Failed to calculate overlap for ROI '{roi_name}': {mask_e}", exc_info=True)
                      non_overlap_rois.append(roi_name) # Count as non-overlapping on error

    except Exception as e:
        logging.error(f"Error during overlap calculation: {e}", exc_info=True)
        # Return empty lists on major failure
        return [], []

    logging.info(f"  Overlap results: {len(overlap_rois)} overlapping, {len(non_overlap_rois)} non-overlapping.")
    return sorted(overlap_rois), sorted(non_overlap_rois) # Return sorted lists