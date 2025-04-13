import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# --------------------
# Dataset Upload
# --------------------

def fetch_dataset(folder="dataset"):
    # From dataset repo  https://shorturl.at/e1Mh1
    if os.path.exists(folder):
        X = pd.read_csv(os.path.join(folder, "X.csv"))
        y = pd.read_csv(os.path.join(folder, "y.csv"))
        variables = pd.read_csv(os.path.join(folder, "variables.csv"))
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
    X.to_csv(os.path.join(folder, "X.csv"), index=False)
    y.to_csv(os.path.join(folder, "y.csv"), index=False)
    dataset["variables"].to_csv(os.path.join(folder, "variables.csv"), index=False)

    return dataset

# --------------------
# Preprocessing
# --------------------

def remove_ifnan(x):
    return [x_ for x_ in x if isinstance(x_, str) or not np.isnan(x_)]

def preprocess_data(df, variables, filepath=None):
    if filepath is not None and not os.path.exists(filepath):
        variables = variables[variables.type == "Categorical"]
        variables = variables[variables.role != "Target"]

        CAT2IDX = {}
        for col in variables.name:
            uniques = remove_ifnan(df[col].unique())
            CAT2IDX[col] = {uniques[idx]: idx for idx in range(len(uniques))}
            if variables[variables.name == col].missing_values.values[0] == "yes":
                CAT2IDX[col][np.nan] = -1

        for idx in range(len(df)):
            for col in df.iloc[idx].index:
                if col in CAT2IDX.keys():
                    df.loc[idx, col] = CAT2IDX[col].get(df.loc[idx, col], -1)

        df.to_csv(filepath, index=False)
        return df

    return pd.read_csv(filepath)

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

# ------------------------
# Mail Function
# ------------------------

dataset_dict = main()

# Ora puoi usare dataset_dict
print("Dataset visualization:\n")
print(dataset_dict["X"].head())

print("\nNumber of samples:", len(dataset_dict["X"]))
print("\nInfo variabiles:")
print(dataset_dict["variables"].head())

print("\nMetadata:")
print(dataset_dict["metadata"])
