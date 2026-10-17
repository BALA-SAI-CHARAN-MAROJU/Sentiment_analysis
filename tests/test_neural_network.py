"""
tests/test_neural_network.py - Unit tests for PyTorch and Multi-Layer Perceptron neural network models.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.neural_network import MLPClassifier

from src.neural_network import (
    PyTorchSentimentClassifier,
    SentimentNeuralNetwork,
    SparseTextDataset,
)


def test_pytorch_neural_network_architecture():
    """Verify PyTorch model forward pass, batchnorm, and output dimensions."""
    input_dim = 150
    batch_size = 8
    num_classes = 3

    model = SentimentNeuralNetwork(
        input_dim=input_dim,
        hidden_dim1=64,
        hidden_dim2=32,
        num_classes=num_classes,
        dropout_rate=0.2,
    )

    x = torch.randn(batch_size, input_dim)
    output = model(x)

    assert output.shape == (batch_size, num_classes)
    assert not torch.isnan(output).any()


def test_sparse_text_dataset():
    """Verify SparseTextDataset handles sparse csr_matrix correctly."""
    X_sparse = sp.csr_matrix([[0.0, 1.5, 0.0], [2.1, 0.0, 0.5]])
    y = np.array([0, 1])

    dataset = SparseTextDataset(X_sparse, y)
    assert len(dataset) == 2

    x0, y0 = dataset[0]
    assert isinstance(x0, torch.Tensor)
    assert x0.shape == (3,)
    assert torch.isclose(x0[1], torch.tensor(1.5, dtype=torch.float32))
    assert y0.item() == 0


def test_pytorch_sentiment_classifier_fit_and_predict():
    """Verify PyTorchSentimentClassifier conforms to scikit-learn estimator interface."""
    np.random.seed(42)
    # 30 samples, 20 features
    X_dense = np.random.randn(30, 20).astype(np.float32)
    y = np.array(["negative", "neutral", "positive"] * 10)

    clf = PyTorchSentimentClassifier(
        hidden_dim1=32,
        hidden_dim2=16,
        epochs=3,
        batch_size=10,
        random_state=42,
    )
    clf.fit(X_dense, y)

    assert clf.classes_ is not None
    assert set(clf.classes_) == {"negative", "neutral", "positive"}

    # Predict probabilities
    probs = clf.predict_proba(X_dense[:5])
    assert probs.shape == (5, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-4)

    # Predict labels
    preds = clf.predict(X_dense[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in clf.classes_


def test_pytorch_checkpoint_save_and_load(tmp_path):
    """Verify serialization and restoration of PyTorch model state."""
    X = np.random.randn(20, 15).astype(np.float32)
    y = np.array(["negative", "neutral", "positive"] * 6 + ["positive", "neutral"])

    clf = PyTorchSentimentClassifier(
        hidden_dim1=16,
        hidden_dim2=8,
        epochs=2,
        batch_size=10,
        random_state=42,
    )
    clf.fit(X, y)

    checkpoint_file = tmp_path / "test_model.pt"
    clf.save_checkpoint(checkpoint_file)
    assert checkpoint_file.exists()

    loaded_clf = PyTorchSentimentClassifier.load_checkpoint(checkpoint_file)
    assert loaded_clf.model is not None
    assert np.array_equal(loaded_clf.classes_, clf.classes_)

    probs_orig = clf.predict_proba(X[:3])
    probs_loaded = loaded_clf.predict_proba(X[:3])
    np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-5)


def test_mlp_classifier_integration():
    """Verify scikit-learn MLPClassifier trains and outputs valid probabilities."""
    X = sp.csr_matrix(np.random.randn(20, 25).astype(np.float32))
    y = np.array(["negative", "neutral", "positive"] * 6 + ["positive", "neutral"])

    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        max_iter=10,
        random_state=42,
    )
    mlp.fit(X, y)

    probs = mlp.predict_proba(X[:3])
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-4)
