# src/modeling.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.base import clone, BaseEstimator, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay, accuracy_score, classification_report
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import SGDClassifier # Needed for type hint/logic
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union # For type hinting
import pickle # For saving models

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomGridSearch:
    """
    Performs GridSearchCV for an SGDClassifier wrapped in a Pipeline
    with StandardScaler. Finds the best alpha for each combo of loss/penalty.

    Expects param_grid keys to be prefixed (e.g., 'estimator__alpha', 'estimator__loss').
    """
    def __init__(self, param_grid: Dict[str, Any], random_state: int, max_iter: int):
        """
        Args:
            param_grid: Parameter grid for the 'estimator' step in the pipeline
                        (must include 'estimator__loss', 'estimator__penalty', 'estimator__alpha').
            random_state: Random state for SGDClassifier.
            max_iter: Max iterations for SGDClassifier.
        """
        # Validate required prefixed keys are intended in the grid
        required_keys = ['estimator__loss', 'estimator__penalty', 'estimator__alpha']
        if not all(key in param_grid for key in required_keys):
            logging.warning(f"Input param_grid might be missing required prefixed keys: {required_keys}")
            # Allow proceeding but results might be unexpected if keys mismatch pipeline structure

        self.param_grid = param_grid
        self.random_state = random_state
        self.max_iter = max_iter
        self.grid_search_: Optional[GridSearchCV] = None
        self.results_df_: Optional[pd.DataFrame] = None
        self.best_alphas_: Dict[str, float] = {} # Stores 'loss_penalty': best_alpha

    def run_search(self, X: np.ndarray, y: np.ndarray, **grid_kwargs: Any):
        """
        Run GridSearchCV on a Pipeline(StandardScaler, SGDClassifier).

        Args:
            X: Feature matrix.
            y: Target labels.
            **grid_kwargs: Additional arguments for GridSearchCV (e.g., cv=5, scoring='accuracy').
        """
        X, y = check_X_y(X, y) # Basic validation

        # Define the base estimator (will be cloned inside Pipeline)
        base_estimator = SGDClassifier(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight='balanced'
        )

        # Create the Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', base_estimator) # Name step 'estimator' to match param_grid prefix
        ])

        logging.info("Running GridSearchCV for Pipeline(StandardScaler, SGDClassifier)...")
        start_time = time.time()
        # Pass the pipeline and the PRE-PREFIXED param_grid
        self.grid_search_ = GridSearchCV(pipeline, self.param_grid, **grid_kwargs)
        try:
            self.grid_search_.fit(X, y) # Fit the pipeline
            self.results_df_ = pd.DataFrame(self.grid_search_.cv_results_)
            logging.info(f"GridSearchCV finished in {time.time() - start_time:.2f} seconds.")
            logging.info(f"Best overall pipeline score: {self.grid_search_.best_score_:.4f}")
            logging.info(f"Best overall pipeline params: {self.grid_search_.best_params_}")
            self._extract_best_alphas() # Extract best alpha per combo
        except Exception as e:
             logging.error(f"GridSearchCV failed: {e}", exc_info=True)
             self.grid_search_ = None
             self.results_df_ = None

    def _extract_best_alphas(self):
        """Extracts the best alpha for each loss/penalty combination from cv_results_."""
        if self.results_df_ is None:
            logging.warning("Cannot extract alphas, grid search did not run or failed.")
            return

        self.best_alphas_ = {}
        # Define prefixed column names expected in cv_results_
        loss_col = 'param_estimator__loss'
        penalty_col = 'param_estimator__penalty'
        alpha_col = 'param_estimator__alpha'
        rank_col = 'rank_test_score'
        score_col = 'mean_test_score'

        required_cols = [loss_col, penalty_col, alpha_col, rank_col, score_col]
        if not all(col in self.results_df_.columns for col in required_cols):
             missing = [col for col in required_cols if col not in self.results_df_.columns]
             logging.error(f"cv_results_ DataFrame missing required columns: {missing}. Cannot extract best alphas per combo.")
             return

        # Get unique values directly from the prefixed columns
        losses = self.results_df_[loss_col].unique()
        penalties = self.results_df_[penalty_col].unique()

        for loss in losses:
            for penalty in penalties:
                combo_key = f"{loss}_{penalty}" # User-friendly key for output dict
                # Filter results using prefixed column names
                df_combo = self.results_df_[
                    (self.results_df_[loss_col] == loss) &
                    (self.results_df_[penalty_col] == penalty)
                ].copy()

                if df_combo.empty:
                    logging.warning(f"No results found for combo: {combo_key}")
                    continue

                # Sort by rank (best = 1) and then score (higher is better)
                df_combo = df_combo.sort_values(by=[rank_col, score_col], ascending=[True, False])
                best_row = df_combo.iloc[0]

                # Extract the best alpha (will be in the prefixed column)
                best_alpha = best_row.get(alpha_col, None)

                if best_alpha is not None:
                    self.best_alphas_[combo_key] = best_alpha
                    logging.info(f"Best alpha for {combo_key}: {best_alpha:.5f} (score: {best_row[score_col]:.4f})")
                else:
                     logging.warning(f"Could not extract best alpha for {combo_key}")


    def get_best_alphas(self) -> Dict[str, float]:
        """Returns the dictionary mapping 'loss_penalty' to the best alpha found."""
        return self.best_alphas_

    def get_results_df(self) -> Optional[pd.DataFrame]:
        """Returns the full cv_results_ DataFrame."""
        return self.results_df_

    def get_best_overall_params(self) -> Dict[str, Any]:
         """Returns the single best parameter set found for the entire pipeline."""
         if self.grid_search_ and hasattr(self.grid_search_, 'best_params_'):
              return self.grid_search_.best_params_
         return {}

    def get_best_overall_pipeline(self) -> Optional[Pipeline]:
        """Returns the best pipeline instance refit on the whole dataset."""
        if self.grid_search_ and hasattr(self.grid_search_, 'best_estimator_'):
              return self.grid_search_.best_estimator_
        return None

class EvaluationPipeline:
    """
    Performs K-Fold Cross-Validation (with internal scaling via Pipeline)
    for multiple base classifiers and plots ROC curves.
    """
    def __init__(self, base_classifier_dict: Dict[str, BaseEstimator]):
        """
        Args:
            base_classifier_dict: Dictionary mapping classifier names to BASE
                                  scikit-learn estimator instances (e.g., SGDClassifier).
                                  These should be configured with optimal hyperparameters.
        """
        # Ensure input are base estimators
        if not all(is_classifier(clf) for clf in base_classifier_dict.values()):
            raise TypeError("Input estimators must be scikit-learn compatible classifiers.")
        self.base_classifier_dict = base_classifier_dict
        self.cv_results_: Dict[str, Dict[str, Any]] = {}
        # Stores the FITTED PIPELINE objects
        self.fitted_pipelines_: Dict[str, Optional[Pipeline]] = {}

    # --- Keep the _plot_single_roc_curve helper function (no changes needed) ---
    def _plot_single_roc_curve(self, ax: plt.Axes, mean_fpr: np.ndarray, tprs: List[np.ndarray], aucs: List[float], classifier_name: str):
        # (Keep the exact code from previous versions)
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color="b", label=rf"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})", lw=2, alpha=0.8)
        std_tpr = np.std(tprs, axis=0); tprs_upper = np.minimum(mean_tpr + std_tpr, 1); tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{classifier_name} ROC Curve")
        ax.legend(loc="lower right", fontsize="small")


    def run_cv_and_plot(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_ids: np.ndarray, # Required for subject_kfold
        num_folds: int = 5,
        cv_strategy: str = 'subject_kfold', # 'subject_kfold' or 'stratified_kfold'
        random_state_kfold: Optional[int] = None,
        pls_workaround: bool = False # If base estimator is PLS-like
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Optional[Pipeline]]]:
        """
        Performs K-Fold CV using Pipeline(StandardScaler, estimator).
        Evaluates on the hold-out fold within each split.
        Plots ROC curves. Fits final pipeline on the full training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target labels.
            train_ids: Subject IDs for training data (for 'subject_kfold').
            num_folds: Number of cross-validation folds.
            cv_strategy: 'subject_kfold' or 'stratified_kfold'.
            random_state_kfold: Random state for the KFold splitter.
            pls_workaround: If True, use RocCurveDisplay.from_predictions for models
                            with 'pls' in their name (assumes base estimator lacks predict_proba).

        Returns:
            A tuple containing:
            - fig: The matplotlib Figure object containing the ROC plots.
            - axes: The matplotlib Axes object array.
            - fitted_pipelines: Dictionary of final pipelines fitted on full X_train, y_train.
        """
        X_train, y_train = check_X_y(X_train, y_train) # Validate input
        if cv_strategy == 'subject_kfold' and (train_ids is None or len(train_ids) != len(y_train)):
            raise ValueError("train_ids must be provided and match y_train length for 'subject_kfold'.")

        n_classifiers = len(self.base_classifier_dict)
        fig, axes = plt.subplots(figsize=(7 * n_classifiers, 6), nrows=1, ncols=n_classifiers, squeeze=False)
        axes = axes.flatten()

        self.cv_results_ = {} # Reset results
        self.fitted_pipelines_ = {}

        # --- Setup CV Splitter ---
        if cv_strategy == 'subject_kfold':
            unique_ids = np.unique(train_ids)
            if len(unique_ids) < num_folds:
                logging.warning(f"Adjusting num_folds from {num_folds} to {len(unique_ids)} due to fewer unique subjects.")
                num_folds = len(unique_ids)
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state_kfold)
            cv_fold_indices = list(cv.split(unique_ids)) # Materialize folds
        elif cv_strategy == 'stratified_kfold':
            cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state_kfold)
            cv_fold_indices = list(cv.split(X_train, y_train)) # Materialize folds
        else:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

        # --- Iterate through classifiers ---
        for c, (name, base_classifier) in enumerate(self.base_classifier_dict.items()):
            logging.info(f"Running {num_folds}-fold CV for {name} (with scaling)...")

            fold_accuracies = np.zeros(num_folds)
            fold_conf_matrices = np.zeros((num_folds, 2, 2))
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # --- Cross-Validation Loop ---
            for k, (train_indices, test_indices) in enumerate(cv_fold_indices):

                # Determine actual train/test indices in the original X_train array
                if cv_strategy == 'subject_kfold':
                    fold_train_mask = np.in1d(train_ids, unique_ids[train_indices])
                    fold_test_mask = np.in1d(train_ids, unique_ids[test_indices])
                else: # stratified_kfold
                    fold_train_mask = train_indices
                    fold_test_mask = test_indices

                X_train_fold, y_train_fold = X_train[fold_train_mask], y_train[fold_train_mask]
                X_test_fold, y_test_fold = X_train[fold_test_mask], y_train[fold_test_mask]

                if X_test_fold.shape[0] == 0:
                    logging.warning(f"  Skipping fold {k+1} for {name}: No test samples in fold.")
                    fold_accuracies[k] = np.nan
                    continue

                # --- Create and Fit Pipeline for this Fold ---
                try:
                    base_estimator_clone = clone(base_classifier)
                    pipe_fold = Pipeline([
                        ('scaler', StandardScaler()),
                        ('estimator', base_estimator_clone)
                    ])
                    pipe_fold.fit(X_train_fold, y_train_fold)
                except Exception as fit_e:
                     logging.error(f"  Pipeline fitting failed for {name} fold {k+1}: {fit_e}", exc_info=True)
                     fold_accuracies[k] = np.nan
                     continue # Skip evaluation for this fold

                # --- Evaluate on Hold-out Fold using the Pipeline ---
                try:
                    predictions = pipe_fold.predict(X_test_fold)
                    fold_accuracies[k] = pipe_fold.score(X_test_fold, y_test_fold)
                    fold_conf_matrices[k, :, :] = confusion_matrix(y_test_fold, predictions, labels=[0, 1])

                    # --- ROC Curve Data ---
                    # Check the FITTED pipeline or its final estimator for probability/decision methods
                    viz = None
                    final_estimator_in_pipe = pipe_fold.named_steps['estimator']

                    # Check for PLS workaround based on BASE classifier name, but predict using pipe
                    if pls_workaround and "pls" in name.lower():
                        # This assumes pipe.predict() returns suitable scores for PLS-like models
                        probas_ = pipe_fold.predict(X_test_fold)
                        viz = RocCurveDisplay.from_predictions(y_test_fold, probas_, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    elif hasattr(pipe_fold, "predict_proba"): # Check pipeline first
                         viz = RocCurveDisplay.from_estimator(pipe_fold, X_test_fold, y_test_fold, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    elif hasattr(pipe_fold, "decision_function"): # Check pipeline second
                         probas_ = pipe_fold.decision_function(X_test_fold)
                         viz = RocCurveDisplay.from_predictions(y_test_fold, probas_, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    else:
                        logging.warning(f"  Cannot plot ROC for {name} fold {k+1}: Pipeline lacks predict_proba/decision_function.")

                    if viz:
                        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        aucs.append(viz.roc_auc)

                except Exception as eval_e:
                    logging.error(f"  Evaluation failed for {name} fold {k+1}: {eval_e}", exc_info=True)
                    fold_accuracies[k] = np.nan


            # --- Post-CV Calculations & Plotting for this classifier ---
            mean_accuracy = np.nanmean(fold_accuracies)
            std_accuracy = np.nanstd(fold_accuracies)
            total_conf_matrix = np.sum(fold_conf_matrices, axis=0)

            logging.info(f"  {name}: Mean CV Accuracy = {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
            logging.info(f"  {name}: Total Confusion Matrix (across folds):\n{total_conf_matrix}")

            # Plot mean ROC curve if data available
            if tprs and aucs:
                self._plot_single_roc_curve(axes[c], mean_fpr, tprs, aucs, name)
            else:
                 axes[c].set_title(f"{name} ROC Curve (Data Unavailable)")

            # Store results
            self.cv_results_[name] = {
                'mean_accuracy': mean_accuracy, 'std_accuracy': std_accuracy,
                'confusion_matrix_total': total_conf_matrix, 'mean_auc': np.mean(aucs) if aucs else np.nan,
                'std_auc': np.std(aucs) if aucs else np.nan, 'accuracies_folds': fold_accuracies
            }

            # --- Fit final PIPELINE on full training data ---
            try:
                final_base_estimator = clone(base_classifier)
                final_pipe = Pipeline([
                     ('scaler', StandardScaler()),
                     ('estimator', final_base_estimator)
                ])
                final_pipe.fit(X_train, y_train)
                self.fitted_pipelines_[name] = final_pipe # Store the fitted pipeline
                logging.info(f"  Fitted final pipeline for {name} on full training data.")
            except Exception as final_fit_e:
                 logging.error(f"  Failed to fit final pipeline for {name}: {final_fit_e}", exc_info=True)
                 self.fitted_pipelines_[name] = None


        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, axes, self.fitted_pipelines_ # Return dict of pipelines


    # --- Keep get_cv_metrics_summary (no changes needed) ---
    def get_cv_metrics_summary(self) -> pd.DataFrame:
        # (Keep the exact code from previous versions)
        summary_data = []; #... build summary_data list ...
        for name, results in self.cv_results_.items():
            conf_mat = results['confusion_matrix_total']; tn, fp, fn, tp = conf_mat.ravel() if conf_mat.size == 4 else (0,0,0,0)
            total = tn + fp + fn + tp; accuracy = (tp + tn) / total if total > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan; recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan; f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
            summary_data.append({'Estimator': name,'Mean CV Accuracy': results.get('mean_accuracy', np.nan),'Std CV Accuracy': results.get('std_accuracy', np.nan),'Mean CV AUC': results.get('mean_auc', np.nan),'Std CV AUC': results.get('std_auc', np.nan),'Total TP': tp,'Total FP': fp,'Total FN': fn,'Total TN': tn,'Overall Precision': precision,'Overall Recall (Sensitivity)': recall,'Overall Specificity': specificity,'Overall F1-Score': f1,})
        return pd.DataFrame(summary_data).round(4)

    # --- Modify save_fitted_models to reflect saving pipelines ---
    def save_fitted_models(self, file_path: Union[str, Path]):
        """Saves the dictionary of final fitted PIPELINES to a pickle file."""
        if not self.fitted_pipelines_:
            logging.warning("No fitted pipelines available to save.")
            return
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.fitted_pipelines_, f) # Save the dictionary of pipelines
            logging.info(f"Saved final fitted pipelines to {file_path}")
        except Exception as e:
            logging.error(f"Error saving pipelines to {file_path}: {e}", exc_info=True)

# Replace the existing CrossEvaluationPipeline in src/modeling.py

class CrossEvaluationPipeline:
    """
    Performs K-Fold Cross-Validation using Pipeline(StandardScaler, estimator).
    Models are trained on X_train folds and evaluated on a separate X_eval set.
    Plots ROC curves based on cross-dataset evaluation.
    """
    def __init__(self, base_classifier_dict: Dict[str, BaseEstimator]):
        """
        Args:
            base_classifier_dict: Dictionary mapping classifier names to BASE
                                  scikit-learn estimator instances (e.g., SGDClassifier).
        """
        if not all(is_classifier(clf) for clf in base_classifier_dict.values()):
            raise TypeError("Input estimators must be scikit-learn compatible classifiers.")
        self.base_classifier_dict = base_classifier_dict
        self.cv_results_: Dict[str, Dict[str, Any]] = {}
        # Stores the FITTED PIPELINE objects (fit on full X_train)
        self.fitted_pipelines_: Dict[str, Optional[Pipeline]] = {}

    # --- Keep the _plot_single_roc_curve helper function (no changes needed) ---
    def _plot_single_roc_curve(self, ax: plt.Axes, mean_fpr: np.ndarray, tprs: List[np.ndarray], aucs: List[float], classifier_name: str):
        # (Keep the exact code from previous versions)
        # ... (full plotting code as in EvaluationPipeline) ...
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color="b", label=rf"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})", lw=2, alpha=0.8)
        std_tpr = np.std(tprs, axis=0); tprs_upper = np.minimum(mean_tpr + std_tpr, 1); tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{classifier_name} ROC Curve")
        ax.legend(loc="lower right", fontsize="small")

    def run_cross_cv_and_plot(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray, # The dataset to evaluate on
        y_eval: np.ndarray, # Labels for the evaluation dataset
        num_folds: int = 5,
        random_state_kfold: Optional[int] = None, # For StratifiedKFold on X_train
        pls_workaround: bool = False
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Optional[Pipeline]]]:
        """
        Performs K-Fold CV on X_train using Pipeline(StandardScaler, estimator).
        In each fold, fits pipeline on train part and evaluates on X_eval/y_eval.
        Plots ROC curves. Fits final pipeline on full X_train/y_train.

        Args:
            X_train: Feature matrix for training the models.
            y_train: Target labels for training.
            X_eval: Feature matrix for evaluating the models in each fold.
            y_eval: Target labels for evaluation.
            num_folds: Number of cross-validation folds (on X_train/y_train).
            random_state_kfold: Random state for StratifiedKFold.
            pls_workaround: If True, use from_predictions for 'pls' models.

        Returns:
            A tuple containing:
            - fig: The matplotlib Figure object containing the ROC plots.
            - axes: The matplotlib Axes object array.
            - fitted_pipelines: Dictionary of final pipelines fitted on full X_train, y_train.
        """
        X_train, y_train = check_X_y(X_train, y_train)
        X_eval = check_array(X_eval) # Basic check on eval data
        y_eval = check_array(y_eval, ensure_2d=False, dtype=None)
        if X_train.shape[1] != X_eval.shape[1]:
             raise ValueError("Feature mismatch between X_train and X_eval.")

        n_classifiers = len(self.base_classifier_dict)
        fig, axes = plt.subplots(figsize=(7 * n_classifiers, 6), nrows=1, ncols=n_classifiers, squeeze=False)
        axes = axes.flatten()

        self.cv_results_ = {}
        self.fitted_pipelines_ = {}

        # --- Setup Stratified KFold on the Training data ---
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state_kfold)
        cv_fold_indices = list(cv.split(X_train, y_train)) # Materialize

        # --- Iterate through classifiers ---
        for c, (name, base_classifier) in enumerate(self.base_classifier_dict.items()):
            logging.info(f"Running {num_folds}-fold Cross-Evaluation for {name} (with scaling)...")

            fold_accuracies = np.zeros(num_folds)
            fold_conf_matrices = np.zeros((num_folds, 2, 2))
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # --- Cross-Validation Loop ---
            for k, (train_indices, _) in enumerate(cv_fold_indices): # Only need train indices
                X_train_fold, y_train_fold = X_train[train_indices], y_train[train_indices]

                # --- Create and Fit Pipeline for this Fold ---
                try:
                    base_estimator_clone = clone(base_classifier)
                    pipe_fold = Pipeline([
                        ('scaler', StandardScaler()),
                        ('estimator', base_estimator_clone)
                    ])
                    pipe_fold.fit(X_train_fold, y_train_fold)
                except Exception as fit_e:
                     logging.error(f"  Pipeline fitting failed for {name} fold {k+1}: {fit_e}", exc_info=True)
                     fold_accuracies[k] = np.nan
                     continue

                # --- Evaluate on the SEPARATE Evaluation Set (X_eval, y_eval) ---
                try:
                    predictions = pipe_fold.predict(X_eval)
                    fold_accuracies[k] = pipe_fold.score(X_eval, y_eval)
                    fold_conf_matrices[k, :, :] = confusion_matrix(y_eval, predictions, labels=[0, 1])

                    # --- ROC Curve Data ---
                    viz = None
                    # Check pipeline for methods, consider base estimator name for PLS workaround
                    if pls_workaround and "pls" in name.lower():
                        probas_ = pipe_fold.predict(X_eval) # Assume predict works for PLS scores
                        viz = RocCurveDisplay.from_predictions(y_eval, probas_, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    elif hasattr(pipe_fold, "predict_proba"):
                        viz = RocCurveDisplay.from_estimator(pipe_fold, X_eval, y_eval, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    elif hasattr(pipe_fold, "decision_function"):
                         probas_ = pipe_fold.decision_function(X_eval)
                         viz = RocCurveDisplay.from_predictions(y_eval, probas_, name=f"Fold {k+1}", alpha=0.3, lw=1, ax=axes[c])
                    else:
                        logging.warning(f"  Cannot plot ROC for {name} fold {k+1}: Pipeline lacks predict_proba/decision_function.")

                    if viz:
                        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        aucs.append(viz.roc_auc)

                except Exception as eval_e:
                    logging.error(f"  Cross-evaluation failed for {name} fold {k+1}: {eval_e}", exc_info=True)
                    fold_accuracies[k] = np.nan


            # --- Post-CV Calculations & Plotting for this classifier ---
            mean_accuracy = np.nanmean(fold_accuracies)
            std_accuracy = np.nanstd(fold_accuracies)
            total_conf_matrix = np.sum(fold_conf_matrices, axis=0)

            logging.info(f"  {name}: Mean Cross-Eval Accuracy = {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
            logging.info(f"  {name}: Total Cross-Eval Confusion Matrix (across folds):\n{total_conf_matrix}")

            if tprs and aucs:
                self._plot_single_roc_curve(axes[c], mean_fpr, tprs, aucs, name)
            else:
                 axes[c].set_title(f"{name} ROC Curve (Data Unavailable)")

            # Store results
            self.cv_results_[name] = {
                'mean_accuracy': mean_accuracy, 'std_accuracy': std_accuracy,
                'confusion_matrix_total': total_conf_matrix, 'mean_auc': np.mean(aucs) if aucs else np.nan,
                'std_auc': np.std(aucs) if aucs else np.nan, 'accuracies_folds': fold_accuracies
            }

            # --- Fit final PIPELINE on the FULL training data (X_train, y_train) ---
            try:
                final_base_estimator = clone(base_classifier)
                final_pipe = Pipeline([
                     ('scaler', StandardScaler()),
                     ('estimator', final_base_estimator)
                ])
                final_pipe.fit(X_train, y_train) # Fit on full X_train
                self.fitted_pipelines_[name] = final_pipe # Store the fitted pipeline
                logging.info(f"  Fitted final pipeline for {name} on full training data.")
            except Exception as final_fit_e:
                 logging.error(f"  Failed to fit final pipeline for {name}: {final_fit_e}", exc_info=True)
                 self.fitted_pipelines_[name] = None


        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, axes, self.fitted_pipelines_ # Return dict of pipelines


    # --- Keep get_cv_metrics_summary (no changes needed) ---
    def get_cv_metrics_summary(self) -> pd.DataFrame:
        # (Keep the exact code from previous versions)
        summary_data = []; #... build summary_data list ...
        for name, results in self.cv_results_.items(): #... append results ...
            conf_mat = results['confusion_matrix_total']; tn, fp, fn, tp = conf_mat.ravel() if conf_mat.size == 4 else (0,0,0,0)
            total = tn + fp + fn + tp; accuracy = (tp + tn) / total if total > 0 else np.nan; precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan; recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan; specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan; f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
            summary_data.append({'Estimator': name,'Mean CV Accuracy': results.get('mean_accuracy', np.nan),'Std CV Accuracy': results.get('std_accuracy', np.nan),'Mean CV AUC': results.get('mean_auc', np.nan),'Std CV AUC': results.get('std_auc', np.nan),'Total TP': tp,'Total FP': fp,'Total FN': fn,'Total TN': tn,'Overall Precision': precision,'Overall Recall (Sensitivity)': recall,'Overall Specificity': specificity,'Overall F1-Score': f1,})
        return pd.DataFrame(summary_data).round(4)

    # --- Modify save_fitted_models to reflect saving pipelines ---
    def save_fitted_models(self, file_path: Union[str, Path]):
        """Saves the dictionary of final fitted PIPELINES (trained on full X_train)"""
        if not self.fitted_pipelines_:
            logging.warning("No fitted pipelines available to save.")
            return
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.fitted_pipelines_, f) # Save the dictionary of pipelines
            logging.info(f"Saved final fitted pipelines to {file_path}")
        except Exception as e:
            logging.error(f"Error saving pipelines to {file_path}: {e}", exc_info=True)

class MultiClassEvaluator:
    """
    Performs Stratified K-Fold Cross-Validation for a dictionary of multi-class classifiers.
    Calculates average accuracy per classifier and fits final models on the full training data.
    """
    def __init__(self, classifier_dict: Dict[str, BaseEstimator]):
        """
        Args:
            classifier_dict: Dictionary mapping classifier names (e.g., 'OVR_L1_Logistic')
                             to scikit-learn compatible multi-class classifier instances
                             (e.g., OneVsRestClassifier(SGDClassifier(...))).
                             Estimators should be pre-configured with desired parameters.
        """
        if not isinstance(classifier_dict, dict):
             raise TypeError("Input must be a dictionary of classifiers.")
        for name, estimator in classifier_dict.items():
             if not is_classifier(estimator):
                 raise TypeError(f"Estimator '{name}' is not a scikit-learn compatible classifier.")
        self.classifier_dict = classifier_dict
        # Store results per classifier
        self.cv_results_: Dict[str, Dict[str, Any]] = {}
        self.fitted_estimators_: Dict[str, Optional[BaseEstimator]] = {} # Final models fit on all train data

    def evaluate_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_splits: int = 5,
        random_state_kfold: Optional[int] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[BaseEstimator]]]:
        """
        Performs Stratified K-Fold cross-validation for each classifier in the dictionary.

        Args:
            X_train: Training feature matrix.
            y_train: Training target labels (multi-class).
            n_splits: Number of cross-validation folds.
            random_state_kfold: Random state for StratifiedKFold.

        Returns:
            A tuple containing:
            - cv_results: Dictionary mapping classifier names to their CV results (mean/std accuracy, etc.).
            - fitted_estimators: Dictionary mapping classifier names to the final estimator
                                 fitted on the full X_train, y_train (or None if fitting failed).
        """
        X_train, y_train = check_X_y(X_train, y_train)
        self.cv_results_ = {} # Reset results
        self.fitted_estimators_ = {}

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_kfold)

        # --- Iterate through each classifier provided ---
        for name, estimator in self.classifier_dict.items():
            logging.info(f"Running {n_splits}-fold Stratified CV for {name}...")
            fold_accuracies = np.zeros(n_splits)
            cv_iterable = skf.split(X_train, y_train) # Recreate iterator for each classifier

            for k, (train_indices, test_indices) in enumerate(cv_iterable):
                X_train_fold, y_train_fold = X_train[train_indices], y_train[train_indices]
                X_test_fold, y_test_fold = X_train[test_indices], y_train[test_indices]

                estimator_clone = clone(estimator)
                try:
                    estimator_clone.fit(X_train_fold, y_train_fold)
                    preds = estimator_clone.predict(X_test_fold)
                    acc = accuracy_score(y_test_fold, preds)
                    fold_accuracies[k] = acc
                except Exception as e:
                    logging.error(f"Failed evaluating fold {k+1} for {name}: {e}", exc_info=True)
                    fold_accuracies[k] = np.nan

            mean_accuracy = np.nanmean(fold_accuracies)
            std_accuracy = np.nanstd(fold_accuracies)
            logging.info(f"  {name} CV finished. Mean Accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")

            self.cv_results_[name] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'fold_accuracies': fold_accuracies
            }

            # --- Fit final model on full training data ---
            try:
                final_estimator = clone(estimator)
                final_estimator.fit(X_train, y_train)
                self.fitted_estimators_[name] = final_estimator
                logging.info(f"  Fitted final model for {name} on full training data.")
            except Exception as e:
                logging.error(f"  Failed to fit final model for {name}: {e}", exc_info=True)
                self.fitted_estimators_[name] = None

        return self.cv_results_, self.fitted_estimators_

    def get_cv_metrics_summary(self) -> pd.DataFrame:
         """Returns a DataFrame summarizing the main CV results for each classifier."""
         summary_data = []
         for name, results in self.cv_results_.items():
             summary_data.append({
                 'Estimator': name,
                 'Mean CV Accuracy': results.get('mean_accuracy', np.nan),
                 'Std CV Accuracy': results.get('std_accuracy', np.nan),
             })
         return pd.DataFrame(summary_data).round(4)


    def save_models(self, file_path: Union[str, Path]):
        """Saves the dictionary of final fitted estimators to a pickle file."""
        # Note: Saves the dictionary containing all fitted models
        if not self.fitted_estimators_:
            logging.warning("No fitted models available to save.")
            return
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.fitted_estimators_, f) # Save the dictionary
            logging.info(f"Saved dictionary of final fitted multi-class models to {file_path}")
        except Exception as e:
            logging.error(f"Error saving multi-class models dictionary to {file_path}: {e}", exc_info=True)