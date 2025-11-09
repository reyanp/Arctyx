import pandas as pd
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel


def create_labels(data_path, labeling_functions_list, output_path):
    """
    Wraps the entire Snorkel LabelModel workflow for weak supervision.
    Always outputs Parquet format for consistency and robustness.
    
    Args:
        data_path: Path to the raw, unlabeled CSV/Parquet file.
        labeling_functions_list: A list of Python functions (as LabelingFunction objects).
        output_path: Where to save the labeled data. Will be saved as Parquet format.
                    If extension is not .parquet, it will be added automatically.
    
    Returns:
        output_path: The string path to the new Parquet file with label probabilities.
    """
    # Load data (supports both CSV and Parquet input)
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Apply labeling functions
    applier = PandasLFApplier(lfs=labeling_functions_list)
    L_train = applier.apply(df=df)
    
    # Train label model (assuming binary classification)
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100)
    
    # Get probabilities
    probs = label_model.predict_proba(L=L_train)
    
    # Add label probability column (probability of the positive class)
    df['label_probability'] = probs[:, 1]
    
    # Ensure output is always Parquet format
    if not output_path.endswith('.parquet'):
        output_path = output_path.rsplit('.', 1)[0] + '.parquet'
    
    # Save labeled data as Parquet (robust format for internal use)
    df.to_parquet(output_path, index=False)
    
    return output_path

