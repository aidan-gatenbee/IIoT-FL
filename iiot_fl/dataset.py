import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DROPS = ["Machine_ID", "Machine_Type"]
TARGETS = ["Remaining_Useful_Life_days", "Failure_Within_7_Days"]

FEATURES = [
    "Installation_Year",
    "Operational_Hours",
    "Temperature_C",
    "Vibration_mms",
    "Sound_dB",
    "Oil_Level_pct",
    "Coolant_Level_pct",
    "Power_Consumption_kW",
    "Last_Maintenance_Days_Ago",
    "Maintenance_History_Count",
    "Failure_History_Count",
    "AI_Supervision",
    "Error_Codes_Last_30_Days",
    "AI_Override_Events",
    "Laser_Intensity",
    "Hydraulic_Pressure_bar",
    "Coolant_Flow_L_min",
    "Heat_Index",
]

INPUT_DIM = len(FEATURES)


def preprocess_partition(df: pd.DataFrame) -> pd.DataFrame:
    # Drop 0-value columns
    # Replace missing machine-specific values with 0.0

    df = df.drop(columns=[c for c in DROPS if c in df.columns], errors="ignore")
    df = df.fillna(0.0)

    # Convert boolean to float (0 or 1)

    if "AI_Supervision" in df.columns:
        df["AI_Supervision"] = df["AI_Supervision"].astype(float)

    return df


class IoTDataset(Dataset):
    """Convert pandas.DataFrame to PyTorch Dataset"""

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        # Define our target columns

        self.rul = torch.tensor(
            df["Remaining_Useful_Life_days"].values,
            dtype=torch.float32,
        )
        self.failure = torch.tensor(
            df["Failure_Within_7_Days"].values, dtype=torch.float32
        )

        # Scale the features with numpy then convert into a pytorch tensor

        feature_data = df[FEATURES].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            feature_data = self.scaler.fit_transform(feature_data)
        elif scaler is not None:
            self.scaler = scaler
            feature_data = self.scaler.transform(feature_data)
        else:
            self.scaler = None

        self.x = torch.tensor(feature_data, dtype=torch.float32)

        # Address class imbalance in Failure_Within_7_Days
        n_pos = self.failure.sum().item()  # Number of failures in 7 days in the dataset
        n_neg = len(self.failure) - n_pos  # everything else is negative
        self.pos_weight = torch.tensor(
            [
                min(
                    max(
                        n_neg
                        / max(
                            n_pos, 1
                        ),  # Protect against /0 if there are no failures in a set
                        1.0,
                    ),  # Protect against extreme failure rate outlier
                    50.0,
                )
            ],  # Clamp max weight to 50, worth considering as a hyperparameter
            dtype=torch.float32,
        )

        logger.info(
            "Dataset: %d samples | %d failures (%.1f%%) | pos_weight=%.2f",
            len(self.rul),
            int(n_pos),
            100 * n_pos / max(len(self.rul), 1),
            self.pos_weight.item(),
        )

    def __len__(self) -> int:
        return len(self.rul)

    def __getitem__(self, idx):
        return self.x[idx], self.rul[idx], self.failure[idx]


def load_partition(
    data_dir: str,
    machine_type: str,
    batch_size: int = 512,
    train_frac: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    path = os.path.join(data_dir, machine_type, "data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Partition file not found: {path}\n"
            f"Expected structure: {{data_dir}}/{{machine_type}}/data.csv"
        )

    logger.info("Loading partition: %s", path)
    df = pd.read_csv(path)
    df = preprocess_partition(df)

    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Partition '{machine_type}' missing some columns: {missing}")

    train_df = df.sample(frac=train_frac, random_state=random_seed)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    logger.info(
        "Split: %d train / %d val for machine type '%s'",
        len(train_df),
        len(val_df),
        machine_type,
    )

    # Use the same scaler for training and validation, dont fit a new one
    train_ds = IoTDataset(train_df, fit_scaler=True)
    val_ds = IoTDataset(val_df, scaler=train_ds.scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
    )

    return train_loader, val_loader, train_ds.pos_weight
