import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_labels(labeled_parquet_path: str, ground_truth_parquet_path: str) -> dict:

    """

    Compares the Snorkel-generated probabilities against a ground truth set.

    """

    try:

        # Load the newly labeled data (has 'label_probability')

        labeled_df = pd.read_parquet(labeled_parquet_path)

        

        # Load the ground truth data (has the true 'income' label, for example)

        # We assume the ground truth file is a *subset* of the labeled file

        # or corresponds by index.

        truth_df = pd.read_parquet(ground_truth_parquet_path)

        

        # Try to align by index first, but if that fails, try to merge on common columns
        # This handles cases where ground truth is a subset
        
        # First, try index alignment
        if not labeled_df.index.equals(truth_df.index):
            # Try to merge on common columns (excluding label_probability and income)
            common_cols = [col for col in labeled_df.columns 
                          if col not in ['label_probability', 'income', 'income_binary'] 
                          and col in truth_df.columns]
            
            if common_cols:
                # Merge on common columns
                merged = labeled_df.merge(truth_df, on=common_cols, how='inner', suffixes=('', '_truth'))
                if merged.empty:
                    # Fallback: use first N rows where N is min of both
                    min_len = min(len(labeled_df), len(truth_df))
                    aligned_labeled = labeled_df.iloc[:min_len]
                    aligned_truth = truth_df.iloc[:min_len]
                else:
                    aligned_labeled = merged[['label_probability']]
                    # Get the income column (might be 'income' or 'income_truth')
                    income_col = 'income' if 'income' in merged.columns else 'income_truth'
                    aligned_truth = merged[[income_col]].rename(columns={income_col: 'income'})
            else:
                # No common columns, use first N rows
                min_len = min(len(labeled_df), len(truth_df))
                aligned_labeled = labeled_df.iloc[:min_len]
                aligned_truth = truth_df.iloc[:min_len]
        else:
            aligned_labeled, aligned_truth = labeled_df.align(truth_df, join='inner', axis=0)

        if aligned_truth.empty or len(aligned_truth) == 0:
            print("Warning: No matching rows found between labeled and ground truth data.")
            return {"roc_auc": 0.0}

        # Get the true labels - try 'income' first, then 'income_binary'
        if 'income' in aligned_truth.columns:
            true_labels = aligned_truth['income']
        elif 'income_binary' in aligned_truth.columns:
            true_labels = aligned_truth['income_binary']
        else:
            print("Warning: Could not find 'income' or 'income_binary' column in ground truth.")
            return {"roc_auc": 0.0} 

        predicted_probs = aligned_labeled['label_probability']

        auc = roc_auc_score(true_labels, predicted_probs)

        return {"roc_auc": auc}

    except Exception as e:

        print(f"Error during label evaluation: {e}")

        return {"roc_auc": 0.0}

