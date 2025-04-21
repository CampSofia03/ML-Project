import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# --------------------
# Dataset Upload
# --------------------
def fetch_dataset(folder="dataset"):
    if os.path.exists(folder):
        X = pd.read_csv(os.path.join(folder, "X.csv"))
        y = pd.read_csv(os.path.join(folder, "y.csv"))
        variables = pd.read_csv(os.path.join(folder, "variables.csv"))
        
        # Rimuovi colonne con "Unnamed"
        X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
        y = y.loc[:, ~y.columns.str.contains('^Unnamed')]
        variables = variables.loc[:, ~variables.columns.str.contains('^Unnamed')]
        
        metadata = None
        return {"X": X, "y": y, "metadata": metadata, "variables": variables}

    secondary_mushroom = fetch_ucirepo(id=848)
    X = secondary_mushroom.data.features
    y = secondary_mushroom.data.targets

    dataset = {
        "X": X,
        "y": y,
        "metadata": secondary_mushroom.metadata,
        "variables": secondary_mushroom.variables,
    }

    os.makedirs(folder, exist_ok=True)
    X.to_csv(os.path.join(folder, "X.csv"), index=False)  # Ensure index=False
    y.to_csv(os.path.join(folder, "y.csv"), index=False)  # Ensure index=False
    dataset["variables"].to_csv(os.path.join(folder, "variables.csv"), index=False)  # Ensure index=False

    return dataset


# ------------------------
# Main Funciton
# ------------------------

def main():
    dataset_dict = fetch_dataset()

    dataset_dict["X"] = preprocess_data(
        dataset_dict["X"],
        dataset_dict["variables"],
        filepath="dataset/preprocessed.csv",
    )

    return dataset_dict
