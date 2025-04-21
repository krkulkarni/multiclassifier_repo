# Multi-Classifier Brain Data Analysis Project

## Overview

This project performs machine learning classification tasks using brain data (functional connectivity and task activation) from different groups (Healthy Controls, Cannabis Users, Alcohol Users). It includes steps for data loading, preprocessing (including PCA for activation data), hyperparameter tuning, model training (binary and 3-way classification), evaluation, and subsequent network analysis to interpret the results using graph theory metrics and Neurosynth term maps.

The workflow is organized into several Jupyter notebooks, supported by reusable code modules in the `src/` directory.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/krkulkarni/multiclassifier_repo
    cd multiclassifier_repo
    ```

2.  **Create and activate a virtual environment:** (Recommended)
    *   Using Conda:
        ```bash
        conda env create -n <env_name> python=3.11
        conda activate <your-env-name>
        ```
    *   Using venv:
        ```bash
        python -m venv venv
        source venv/bin/activate # On Linux/macOS
        # venv\Scripts\activate # On Windows
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` lists all necessary packages like numpy, pandas, scikit-learn, nilearn, matplotlib, seaborn, plotly, networkx, joblib, PyYAML etc.)*

## Data

Input data is expected to be organized within the `data/` directory following standard conventions:

*   `data/raw/`: Contains immutable raw input data.
    *   `alcohol_glm_output/`: NIfTI files from GLM analysis for the alcohol group.
    *   `cannabis_glm_output/`: NIfTI files from GLM analysis for the cannabis group.
    *   `all_parcellations/`: Directory containing functional connectivity `.npy` files.
*   `data/metadata/` or `data/raw/`: Should contain the subject list file (e.g., `gender_controlled_splits.csv`).
*   `data/interim/`: Stores intermediate data generated during processing (e.g., combined FC data `.npz`, PCA outputs).
*   `analysis_results/neurosynth_maps/`: Expected location for downloaded Neurosynth statistical maps (`.nii.gz`).

Update paths in the notebook configuration cells if your data resides elsewhere.

## Project Structure
```
multiclassifier_repo/
├── data/
│ ├── raw/
│ ├── interim/
│ ├── processed/
│ └── metadata/
├── notebooks/
│ ├── 1_decoding.ipynb
│ ├── 1a_decoding_activation.ipynb
│ ├── 2_network_analysis.ipynb
│ ├── 3_community_classifier.ipynb
│ └── 4_neurosynth.ipynb
├── src/
│ ├── init.py
│ ├── activation_loading.py # Loading activation maps, PCA
│ ├── atlas_utils.py # Handling atlas info, ROI masks
│ ├── data_loading.py # Loading connectivity data
│ ├── file_utils.py # Pickle saving/loading
│ ├── matrix_utils.py # Matrix/vector conversions
│ ├── modeling.py # Core classification pipelines/evaluators
│ ├── modeling_utils.py # Helper functions for modeling (sigmoid, thresholds)
│ ├── network_analysis.py # Graph metrics, community detection logic
│ ├── network_utils.py # Graph creation class, network index helpers
│ ├── neurosynth_utils.py # Overlap calculation and plotting
│ ├── plotting_utils.py # Plotting functions (radar, violins, heatmaps)
│ └── stats_utils.py # Permutation testing helpers
├── models/ # Saved trained models (.pkl) and scalers (.joblib)
│ └── activation/ # Subdirectory for activation-based models
├── reports/
│ ├── figures/ # Saved plots and figures (.svg, .png)
│ │ ├── activation/ # Subdirectory for activation-based figures
│ │ └── network_analysis/ # Subdirectory for network figures
│ └── analysis_results/ # Saved metrics, data summaries (.csv, .pkl)
│ ├── degree_centrality/
│ ├── efficiency_analysis/
│ └── community_classifier/
├── environment.yml # Conda environment file (optional)
├── requirements.txt # Pip requirements file
└── README.md # This file
```

## Workflow and Notebook Roles

The notebooks are designed to be run sequentially, as later notebooks often depend on the outputs (saved models, data, results) of earlier ones.

1.  **`1_decoding.ipynb` - Decoding with Functional Connectivity**
    *   **Purpose:** Performs binary classification (e.g., User vs Control within Alcohol/Cannabis groups, cross-group prediction, aggregated prediction) using **functional connectivity vectors** as features.
    *   **Inputs:** Subject list CSV, functional connectivity `.npy` files.
    *   **Steps:**
        *   Loads and preprocesses connectivity data (`src/data_loading.py`).
        *   Performs hyperparameter grid search for SGD classifiers (`src/modeling.py: CustomGridSearch`).
        *   Trains and evaluates classifiers using K-Fold cross-validation (subject-aware) with integrated scaling (`src/modeling.py: EvaluationPipeline`, `CrossEvaluationPipeline`). Covers within-group, cross-group, and aggregated scenarios.
        *   Generates ROC curves and calculates performance metrics.
        *   Saves final trained models (Pipelines) and evaluation metrics.
    *   **Outputs:** Saved classifier models (`models/`), ROC plots (`reports/figures/`), metrics summaries (`reports/analysis_results/`), processed data (`data/interim/`).

2.  **`1a_decoding_activation.ipynb` - Decoding with Task Activation**
    *   **Purpose:** Mirrors the workflow of `1_decoding.ipynb` but uses **task activation maps (GLM betas/z-stats)** as input features instead of connectivity.
    *   **Inputs:** Subject list CSV, NIfTI activation maps.
    *   **Steps:**
        *   Loads and flattens NIfTI activation maps (`src/activation_loading.py`).
        *   Performs **PCA** on the activation data to reduce dimensionality (`src/activation_loading.py`).
        *   Performs hyperparameter grid search, model training/evaluation (within-group, cross-group, aggregated, 3-way) on the **PCA components** using the same pipeline classes from `src/modeling.py` as Notebook 1.
        *   Saves PCA-based models (`models/activation/`), plots, metrics, and processed PCA data.
    *   **Outputs:** Saved activation-based models, PCA components, plots, metrics.

3.  **`2_network_analysis.ipynb` - Network Analysis of Connectivity/Models**
    *   **Purpose:** Analyzes the results from the **connectivity-based** classification (Notebook 1) or potentially the 3-way activation models (Notebook 1a), focusing on interpreting classifier weights and identifying important network features using graph theory.
    *   **Inputs:** Saved classifier models (specifically the 3-way models like OVR/OVO), processed functional connectivity data (`.npz`), atlas information.
    *   **Steps:**
        *   Calculates weighted functional connectivity (WFC) by combining classifier weights and subject FC data.
        *   Creates run-level graphs from thresholded WFC (`src/network_utils.py: GraphFromConnectivity`).
        *   Calculates graph metrics like Degree Centrality and Efficiency (`src/network_analysis.py`).
        *   Performs community detection on average WFC graphs (`src/network_analysis.py`).
        *   Generates visualizations: WFC heatmaps, DC violin plots (Seaborn), Efficiency violin plots (Seaborn), network radar plot (Matplotlib), community connectomes (Nilearn).
    *   **Outputs:** Saved graph metrics (`reports/analysis_results/degree_centrality/`, `.../efficiency_analysis/`), community analysis results (`.../network_analysis/`), various plots (`reports/figures/network_analysis/`).

4.  **`3_community_classifier.ipynb` - Community-Based Classification**
    *   **Purpose:** Investigates the predictive power of feature subsets defined by the network communities identified in Notebook 2. Performs 3-way classification using features only from cumulatively added top communities.
    *   **Inputs:** Saved 3-way classifier models, combined 3-way activation/connectivity data, saved community definitions/rankings (from Notebook 2).
    *   **Steps:**
        *   Prepares 3-way train/test data splits.
        *   Loads community definitions (sorted by DC) for each group (HC, Cannabis, Alcohol).
        *   Iteratively adds communities based on a selected group's ranking.
        *   For each cumulative set of communities, identifies the corresponding feature indices.
        *   Uses an **approximation method**: zeros out non-selected features in the test data and predicts using the original full 3-way classifier.
        *   Calculates 3-way accuracy.
        *   Performs permutation testing to assess the significance of the accuracy at each step (`src/stats_utils.py`).
        *   Generates plots showing accuracy vs. number of communities included, highlighting significant points.
    *   **Outputs:** Accuracy results per community subset (`reports/analysis_results/community_classifier/`), plots (`reports/figures/community_classifier/`).

5.  **`4_neurosynth.ipynb` - Neurosynth Overlap Analysis**
    *   **Purpose:** Relates brain regions identified as important (e.g., high Degree Centrality ROIs from Notebook 2) to cognitive terms using Neurosynth meta-analytic maps.
    *   **Inputs:** Neurosynth statistical maps (`.nii.gz`), Schaefer atlas information, saved Degree Centrality summary results (to dynamically select ROIs).
    *   **Steps:**
        *   Loads Schaefer atlas and prepares individual ROI masks (`src/atlas_utils.py`).
        *   Loads DC results and dynamically selects the top N ROIs for each group (HC, Cannabis, Alcohol) based on specified model results.
        *   For each group and specified Neurosynth term map (e.g., 'craving', 'drug'):
            *   Calculates the quantitative overlap between the selected ROIs and the Neurosynth map (`src/neurosynth_utils.py`).
            *   Generates and saves anatomical plots showing the ROIs overlaid on the Neurosynth map contours (`src/neurosynth_utils.py`).
    *   **Outputs:** Overlap statistics (printed or saved), overlap plots (`reports/figures/neurosynth_overlap/`).

## Results

*   **Models:** Trained models are saved as pickle files (`.pkl`) in the `models/` directory (with subdirectories for activation vs connectivity). Scalers might be saved as `.joblib` files.
*   **Figures:** Plots generated by the notebooks are saved in `reports/figures/` (with relevant subdirectories).
*   **Metrics & Summaries:** DataFrames containing evaluation metrics, DC scores, community analysis results, etc., are saved as `.csv` or `.pkl` files in `reports/analysis_results/` (with relevant subdirectories).
