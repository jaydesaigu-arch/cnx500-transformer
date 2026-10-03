"""Compact CPU models with causal, continuous next-patch pretraining."""
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summary_features(x):
    pieces = [x[:, -1], x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1)]
    for window in (5, 20, 60):
        if window <= x.shape[1]:
            pieces.extend([x[:, -window:].mean(axis=1), x[:, -window:].std(axis=1)])
    return np.concatenate(pieces, axis=1).astype(np.float32)


class CompactTransformer(nn.Module):
    def __init__(self, features, context, width=32, layers=2, heads=4, dropout=.1):
        super().__init__()
        self.patch = 4
        self.tokens = context // self.patch
        self.embed = nn.Linear(features * self.patch, width)
        self.position = nn.Parameter(torch.zeros(1, self.tokens, width))
        nn.init.normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width*2,
                                          dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 15)
        self.next_patch = nn.Linear(width, features * self.patch)
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.tokens, self.tokens, dtype=torch.bool), diagonal=1))

    def patches(self, x):
        return x.reshape(x.shape[0], self.tokens, -1)

    def encode(self, x):
        embedded = self.embed(self.patches(x)) + self.position
        return self.norm(self.encoder(embedded, mask=self.causal_mask))

    def forward(self, x):
        return self.classifier(self.encode(x)[:, -1]).reshape(-1, 3, 5)

    def pretraining_loss(self, x):
        # Token j predicts token j+1; causal mask forbids attention to j+1.
        predicted = self.next_patch(self.encode(x)[:, :-1])
        return nn.functional.smooth_l1_loss(predicted, self.patches(x)[:, 1:])


class SmallMLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Dropout(.15),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 15))

    def forward(self, x):
        return self.net(x).reshape(-1, 3, 5)


def make_neural(config, x):
    if config["model"] == "mlp":
        return SmallMLP(summary_features(x).shape[1])
    return CompactTransformer(x.shape[2], x.shape[1])


def model_inputs(config, x):
    return summary_features(x) if config["model"] == "mlp" else x


def pretrain(model, x, epochs, seed, batch_size=128):
    if epochs <= 0:
        return []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    history = []
    for _ in range(epochs):
        model.train()
        losses = []
        for (xb,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.pretraining_loss(xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def neural_probabilities(model, config, x):
    model.eval()
    inp = torch.from_numpy(model_inputs(config, x))
    outputs = []
    with torch.inference_mode():
        for xb in inp.split(256):
            outputs.append(model(xb).softmax(dim=-1).numpy())
    return np.concatenate(outputs)


def fit_neural(config, x, y, seed, epochs=20, tuning=None, pretrain_epochs=4):
    seed_all(seed)
    model = make_neural(config, x)
    pre_history = pretrain(model, x, pretrain_epochs if config["model"] == "pretrained_transformer" else 0, seed)
    train_x = model_inputs(config, x)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y)),
                        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=.01)
    history, best_state, best_loss, best_epoch, stale = [], None, float("inf"), epochs, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 5), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "training_cross_entropy": float(np.mean(losses))}
        if tuning is not None:
            xp, yp = tuning
            probs = neural_probabilities(model, config, xp)
            val_loss = float(-np.log(np.clip(np.take_along_axis(probs, yp[..., None], axis=-1)[..., 0], 1e-9, 1)).mean())
            record["tuning_cross_entropy"] = val_loss
            if val_loss < best_loss - .001:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(record)
        if tuning is not None and stale >= 4:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    details = {"best_epoch": best_epoch, "pretraining_history": pre_history, "training_history": history,
               "supervised_seconds": time.perf_counter() - started,
               "parameters": sum(p.numel() for p in model.parameters())}
    return model, details


class Baseline:
    def __init__(self, name, seed=42):
        self.name, self.seed, self.models, self.priors = name, seed, [], []

    def fit(self, x, y):
        features = summary_features(x)
        for k in range(3):
            counts = np.bincount(y[:, k], minlength=5)
            self.priors.append((counts + 1) / (counts.sum() + 5))
            if self.name == "majority":
                continue
            if self.name == "logistic":
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500, random_state=self.seed))
            elif self.name == "histboost":
                model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, learning_rate=.05,
                                                       l2_regularization=10., early_stopping=False,
                                                       random_state=self.seed)
            else:
                raise ValueError(self.name)
            model.fit(features, y[:, k])
            self.models.append(model)
        return self

    def predict_proba(self, x):
        if self.name == "majority":
            return np.broadcast_to(np.array(self.priors)[None], (len(x), 3, 5)).copy()
        features = summary_features(x)
        out = np.full((len(x), 3, 5), 1e-9)
        for k, model in enumerate(self.models):
            out[:, k, model.classes_] = model.predict_proba(features)
        return out / out.sum(axis=-1, keepdims=True)
