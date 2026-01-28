import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.utils.modeling import FitRecDataset


def predict_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    dataset_args: dict,
    n_workers: int = 4,
    verbose: bool = True,
) -> np.ndarray:
    """Predicts heart rate using the provided model and dataset.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        df (pd.DataFrame): The dataset to use for predictions.
        n_workers (int, optional): The number of workers to use for data loading. Defaults to 4.

    Returns:
        np.ndarray: The predicted heart rates.
            shape: (n_workouts, n_samples)

    """
    test_dataset = FitRecDataset(
        df,
        **dataset_args,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(df),
        shuffle=False,
        num_workers=n_workers,
    )

    model.eval()
    with torch.no_grad():
        y_preds = []
        for x, y, u, s, g in tqdm(test_loader, disable=not verbose):
            y_preds.append(model(x, u, s, g))

        y_preds = torch.hstack(y_preds).squeeze().numpy()

    return y_preds
