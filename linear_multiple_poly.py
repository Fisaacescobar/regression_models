#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Regressions: simple linear, multiple linear, and polynomial (degree=3 on X1).
# ASCII-only strings to avoid encoding issues in some editors.
# Requires: numpy, pandas, matplotlib, scikit-learn

from pathlib import Path
from dataclasses import dataclass
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------- utils -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class FitResult:
    name: str
    r2_train: float
    r2_test: float
    mae_test: float
    rmse_test: float
    cv_r2_mean: float | None = None
    cv_r2_std: float | None = None


# ------------------------- synthetic dataset -------------------------

def make_dataset(n: int = 400, noise: float = 1.0, seed: int = 2025) -> pd.DataFrame:
    """
    Generate X1, X2, X3 ~ N(0,1). Target has linear part + nonlinear terms in X1.
    So: simple linear (X1) underfits, multiple improves, polynomial (X1^2, X1^3) captures nonlinearity.
    """
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)
    eps = rng.normal(0, noise, n)

    y = (
        1.0
        + 3.0 * X1
        - 2.0 * X2
        + 1.25 * X3
        + 0.60 * (X1 ** 2)
        - 0.20 * (X1 ** 3)
        + eps
    )

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "y": y})
    return df


# ------------------------- models -------------------------

def fit_linear_simple(df: pd.DataFrame, test_size: float = 0.25, seed: int = 2025):
    X = df[["X1"]].to_numpy()
    y = df["y"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv = cross_val_score(model, X, y, cv=kf, scoring="r2")

    res = FitResult(
        name="linear_simple_X1",
        r2_train=r2_score(y_tr, y_tr_hat),
        r2_test=r2_score(y_te, y_te_hat),
        mae_test=mean_absolute_error(y_te, y_te_hat),
        rmse_test=rmse(y_te, y_te_hat),
        cv_r2_mean=float(cv.mean()),
        cv_r2_std=float(cv.std())
    )
    return res, X_tr, y_tr, X_te, y_te_hat


def fit_linear_multiple(df: pd.DataFrame, test_size: float = 0.25, seed: int = 2025):
    X = df[["X1", "X2", "X3"]].to_numpy()
    y = df["y"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv = cross_val_score(model, X, y, cv=kf, scoring="r2")

    res = FitResult(
        name="linear_multiple_X1_X2_X3",
        r2_train=r2_score(y_tr, y_tr_hat),
        r2_test=r2_score(y_te, y_te_hat),
        mae_test=mean_absolute_error(y_te, y_te_hat),
        rmse_test=rmse(y_te, y_te_hat),
        cv_r2_mean=float(cv.mean()),
        cv_r2_std=float(cv.std())
    )
    return res, y_te, y_te_hat


def fit_poly_X1_degree3(df: pd.DataFrame, test_size: float = 0.25, seed: int = 2025):
    X = df[["X1"]].to_numpy()
    y = df["y"].to_numpy()
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, test_size=test_size, random_state=seed)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_tr_hat = model.predict(X_tr)
    y_te_hat = model.predict(X_te)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv = cross_val_score(model, X_poly, y, cv=kf, scoring="r2")

    res = FitResult(
        name="poly_X1_degree3",
        r2_train=r2_score(y_tr, y_tr_hat),
        r2_test=r2_score(y_te, y_te_hat),
        mae_test=mean_absolute_error(y_te, y_te_hat),
        rmse_test=rmse(y_te, y_te_hat),
        cv_r2_mean=float(cv.mean()),
        cv_r2_std=float(cv.std())
    )
    return res, X, y, model.predict(poly.transform(X))


# ------------------------- plots -------------------------

def plot_simple_fit(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te_hat: np.ndarray, outpath: Path) -> None:
    plt.figure()
    plt.scatter(X_tr[:, 0], y_tr, s=12)
    order = np.argsort(X_te[:, 0])
    plt.plot(X_te[order, 0], y_te_hat[order])
    plt.title("Linear simple: y vs X1")
    plt.xlabel("X1")
    plt.ylabel("y / y_hat")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.title(title)
    plt.xlabel("y true")
    plt.ylabel("y pred")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


def plot_poly_curve(X: np.ndarray, y: np.ndarray, y_hat_all: np.ndarray, outpath: Path) -> None:
    plt.figure()
    plt.scatter(X[:, 0], y, s=12)
    order = np.argsort(X[:, 0])
    plt.plot(X[order, 0], y_hat_all[order])
    plt.title("Polynomial (deg=3) on X1")
    plt.xlabel("X1")
    plt.ylabel("y / y_hat")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


# ------------------------- save metrics -------------------------

def save_metrics(results, out_csv: Path) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regressions: simple/multiple/polynomial (demo)")
    p.add_argument("--n", type=int, default=400, help="n samples")
    p.add_argument("--noise", type=float, default=1.0, help="gaussian noise sigma")
    p.add_argument("--seed", type=int, default=2025, help="RNG seed")
    p.add_argument("--demo", action="store_true", help="run end-to-end and save figures/metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(__file__).parent / "figs"
    ensure_dir(outdir)

    if not args.demo:
        print("Nothing to do. Use --demo (and optional --n/--noise/--seed).")
        return

    df = make_dataset(n=args.n, noise=args.noise, seed=args.seed)

    # 1) simple
    r1, X_tr, y_tr, X_te, y_te_hat = fit_linear_simple(df, seed=args.seed)
    plot_simple_fit(X_tr, y_tr, X_te, y_te_hat, outdir / "simple_fit.png")

    # 2) multiple
    r2, y_te_m, y_hat_m = fit_linear_multiple(df, seed=args.seed)
    plot_pred_vs_true(y_te_m, y_hat_m, outdir / "multiple_pred_vs_true.png", "Multiple linear: y vs y_hat")

    # 3) polynomial
    r3, X_all, y_all, y_hat_all = fit_poly_X1_degree3(df, seed=args.seed)
    plot_poly_curve(X_all, y_all, y_hat_all, outdir / "poly_fit.png")

    # save metrics
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"metrics_{ts}.csv"
    save_metrics([r1, r2, r3], csv_path)

    print("[OK] figures ->", outdir)
    print("[OK] metrics ->", csv_path)


if __name__ == "__main__":
    main()
