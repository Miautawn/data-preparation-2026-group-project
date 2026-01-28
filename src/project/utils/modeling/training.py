from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, path: Path, patience: int = 3):
        """Early stopping helper class to save the best model during training.

        Args:
            patience (int, optional): Number of epochs with no improvement
                after which training will be stopped. Defaults to 3.
            path (str, optional): Path to save the best model. Defaults to "best_model.pth".
        """
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if early stopping condition is met.

        Args:
            val_loss (float): current validation loss
            model (torch.nn.Module): model to save

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            torch.save(model, self.path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def train_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_save_path: Path,
    batch_size: int = 2048,
    epochs: int = 5,
    learning_rate: float = 0.001,
    l2_norm: float = 0.001,
    early_stopping_patience: int = 3,
    n_workers: int = 4,
) -> torch.nn.Module:
    """Train a the FitRecModel model with early stopping and model checkpointing.

    Args:
        model (torch.nn.Module): model to train
        train_dataset (Dataset): training dataset
        val_dataset (Dataset): validation dataset
        model_save_path (Path): path to where model state dict will be saved
        batch_size (int, optional): batch size for training. Defaults to 2048.
        epochs (int, optional): number of training epochs. Defaults to 5.
        learning_rate (float, optional): learning rate for the optimizer. Defaults to 0.001.
        l2_norm (float, optional): L2 regularization strength. Defaults to 0.001.
        early_stopping_patience (int, optional): patience for early stopping. Defaults to 3.
        n_workers (int, optional): number of worker threads for data loading. Defaults to 4.

    Returns:
        torch.nn.Module: trained model (with best weights loaded from early stopping)
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_norm)
    criterion = torch.nn.MSELoss()
    stopper = EarlyStopping(patience=early_stopping_patience, path=model_save_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for x, y, user_idx, sport_idx, gender_idx in pbar:
            optimizer.zero_grad()

            outputs = model(x, user_idx, sport_idx, gender_idx)
            loss = criterion(outputs, y.view(-1, 1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_mse = 0

        pbar = tqdm(val_loader, desc=f"Validation Epoch: {epoch + 1}/{epochs}")
        with torch.no_grad():
            for x, y, u, s, g in pbar:
                outputs = model(x, u, s, g)
                val_mse += criterion(outputs, y.view(-1, 1)).item()

        avg_val_loss = val_mse / len(val_loader)
        avg_val_rmse = avg_val_loss**0.5

        print(f">> Epoch {epoch + 1} Results:")
        print(f"   Train MSE: {avg_train_loss:.2f}")
        print(f"   Val   MSE: {avg_val_loss:.2f} | RMSE: {avg_val_rmse:.2f} BPM")
        print("\n")

        # --- EARLY STOPPING CHECK ---
        if stopper(avg_val_loss, model):
            print(
                f"\nEarly stopping triggered. No improvement for {stopper.patience} epochs."
            )
            model = model.load(weights_only=False, path=model_save_path)
            break

    return model
