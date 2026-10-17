"""
src/neural_network.py - PyTorch & Scikit-Learn Neural Network Architectures
Implements:
1. SentimentNeuralNetwork: Deep feedforward neural network in PyTorch with Batch Normalization and Dropout.
2. PyTorchSentimentClassifier: Scikit-learn compatible estimator wrapping the PyTorch model.
3. Training routines, checkpoint serialization, and probability inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASSES,
    LABEL_MAPPING,
    MAX_FEATURES,
    NGRAM_RANGE,
    PYTORCH_MODEL_FILE,
    RANDOM_STATE,
    VECTORIZER_FILE,
)
from src.utils import logger


# ============================================================================
# 1. PyTorch Dataset & Neural Network Architecture
# ============================================================================

class SparseTextDataset(Dataset):
    """PyTorch Dataset wrapper for sparse or dense feature matrices."""

    def __init__(self, X: Union[sp.spmatrix, np.ndarray], y: Optional[np.ndarray] = None):
        self.is_sparse = sp.issparse(X)
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.is_sparse:
            # Convert single sparse row to dense float32 tensor
            row = torch.from_numpy(self.X[idx].toarray().astype(np.float32).squeeze(0))
        else:
            row = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))

        if self.y is not None:
            return row, self.y[idx]
        return row


class SentimentNeuralNetwork(nn.Module):
    """
    Multi-Layer Deep Neural Network for 3-class Sentiment Polarity Classification.
    Architecture:
      Input (TF-IDF Features) -> Dense (128) -> BatchNorm -> ReLU -> Dropout (0.3)
                              -> Dense (64)  -> BatchNorm -> ReLU -> Dropout (0.2)
                              -> Dense (3)   -> Softmax Logits
    """

    def __init__(
        self,
        input_dim: int = 6000,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        # Output Layer
        self.out = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        if out.size(0) > 1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        logits = self.out(out)
        return logits


# ============================================================================
# 2. Scikit-Learn Compatible PyTorch Classifier Wrapper
# ============================================================================

class PyTorchSentimentClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible estimator wrapping the PyTorch SentimentNeuralNetwork.
    Implements fit, predict, and predict_proba for seamless benchmarking and evaluation.
    """

    def __init__(
        self,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 8,
        batch_size: int = 64,
        dropout_rate: float = 0.3,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self.model: Optional[SentimentNeuralNetwork] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx_: Dict[str, int] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: Union[sp.spmatrix, np.ndarray], y: Union[List[str], np.ndarray]):
        """Fit PyTorch neural network on training feature matrix and labels."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Determine classes
        y_arr = np.array(y)
        self.classes_ = np.unique(y_arr)
        self.class_to_idx_ = {cls_name: idx for idx, cls_name in enumerate(self.classes_)}
        y_indices = np.array([self.class_to_idx_[label] for label in y_arr], dtype=np.int64)

        input_dim = X.shape[1]
        num_classes = len(self.classes_)

        # Instantiate model
        self.model = SentimentNeuralNetwork(
            input_dim=input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=num_classes,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # DataLoader
        dataset = SparseTextDataset(X, y_indices)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Compute balanced class weights for loss function
        class_counts = np.bincount(y_indices, minlength=num_classes)
        total_samples = len(y_indices)
        class_weights = total_samples / (num_classes * np.maximum(class_counts, 1))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / total_samples
            logger.debug(f"[PyTorch NN] Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

        return self

    def predict_proba(self, X: Union[sp.spmatrix, np.ndarray]) -> np.ndarray:
        """Compute class posterior probabilities via softmax."""
        if self.model is None or self.classes_ is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        dataset = SparseTextDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict(self, X: Union[sp.spmatrix, np.ndarray]) -> np.ndarray:
        """Predict sentiment class labels."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]

    def save_checkpoint(self, path: Union[str, Path] = PYTORCH_MODEL_FILE) -> None:
        """Save PyTorch neural network state dict and metadata."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "num_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "class_to_idx": self.class_to_idx_,
        }
        torch.save(checkpoint, path)
        logger.info(f"[INFO] PyTorch Neural Network checkpoint saved to: {path}")

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path] = PYTORCH_MODEL_FILE) -> "PyTorchSentimentClassifier":
        """Load PyTorch neural network from saved checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            hidden_dim1=int(checkpoint["hidden_dim1"]),
            hidden_dim2=int(checkpoint["hidden_dim2"]),
        )
        instance.classes_ = np.array(checkpoint["classes"])
        instance.class_to_idx_ = checkpoint["class_to_idx"]
        instance.model = SentimentNeuralNetwork(
            input_dim=checkpoint["input_dim"],
            hidden_dim1=checkpoint["hidden_dim1"],
            hidden_dim2=checkpoint["hidden_dim2"],
            num_classes=checkpoint["num_classes"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance
