"""Prueba rápida de TabPFN frente a XGBoost, con la misma división que el notebook.

Uso (desde la carpeta del proyecto):
    .venv/bin/python experiments/tabpfn_quick.py
    .venv/bin/python experiments/tabpfn_quick.py --n-context 20000 --device mps

Requisito previo de TabPFN (una sola vez):
    1. Entrar en https://ux.priorlabs.ai, registrarse y aceptar la licencia ("Licenses").
    2. Guardar la API key (https://ux.priorlabs.ai/account) en la variable TABPFN_TOKEN
       o en el archivo ~/.cache/tabpfn/auth_token.
    Los datos no salen del ordenador: la clave solo sirve para descargar los pesos.

Qué hace (en 3 etapas, cada una en su propio proceso):
    1. features: misma división train_train / train_test que el notebook (random_state=1928),
       mismas features (make_tree_preprocessor, con cross-fitting) y XGBoost de referencia.
       Guarda las matrices y las predicciones en experiments/.cache/.
    2. tabpfn: TabPFN con una submuestra de train_train como contexto (--n-context filas).
    3. comparación: RMSE de cada uno, ensambles y correlación de errores.

Por qué procesos separados: en macOS, XGBoost y PyTorch (que usa TabPFN) cargan cada uno su
propia copia de libomp (OpenMP). En el mismo proceso se bloquean entre sí (deadlock).

--dry-run sustituye TabPFN por un modelo sencillo para comprobar el script sin licencia.
"""
import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / ".cache"
RANDOM_STATE = 1928
warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "features", "tabpfn"], default="all", help=argparse.SUPPRESS)
    p.add_argument("--n-context", type=int, default=10_000, help="filas de train_train usadas como contexto")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--n-estimators", type=int, default=4, help="nº de estimadores internos de TabPFN")
    p.add_argument("--batch", type=int, default=2_000, help="filas de train_test predichas por lote")
    p.add_argument("--dry-run", action="store_true", help="no usa TabPFN (prueba el script sin licencia)")
    return p.parse_args()


def rmse(y, p):
    return float(np.sqrt(np.mean((y - np.clip(p, 0, 100)) ** 2)))


def stage_features():
    """Etapa 1: features + XGBoost. No importa PyTorch."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    sys.path.insert(0, str(ROOT / "src"))
    from spotify.features import ROW_FEATURES, add_group_features, make_tree_preprocessor

    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    train, test = add_group_features(train, test)
    X, y = train.drop(columns="popularity"), train["popularity"]
    X_tt, X_th, y_tt, y_th = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    prep = Pipeline([("rows", ROW_FEATURES), ("prep", make_tree_preprocessor(RANDOM_STATE))])
    A = prep.fit_transform(X_tt, y_tt)          # cross-fitting: sin fuga en train_train
    B = prep.transform(X_th)
    print(f"Features: {A.shape[1]} | train_train: {len(A)} | train_test: {len(B)}")

    t0 = time.time()
    xgb = XGBRegressor(n_estimators=3000, learning_rate=0.015, max_depth=9, subsample=0.7,
                       colsample_bytree=0.6, min_child_weight=3, tree_method="hist",
                       random_state=RANDOM_STATE, n_jobs=-1).fit(A, y_tt)
    p_xgb = xgb.predict(B)
    print(f"XGBoost   RMSE {rmse(y_th.to_numpy(), p_xgb):.3f}   ({time.time() - t0:.0f} s)")

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "tabpfn_inputs.npz", A=A, B=B, y_tt=y_tt.to_numpy(float),
             y_th=y_th.to_numpy(float), p_xgb=p_xgb)


def stage_tabpfn(args):
    """Etapa 2: TabPFN. No importa XGBoost."""
    d = np.load(CACHE / "tabpfn_inputs.npz")
    A, B, y_tt = d["A"], d["B"], d["y_tt"]
    idx = np.random.default_rng(RANDOM_STATE).choice(len(A), size=min(args.n_context, len(A)), replace=False)

    if args.dry_run:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model, label = HistGradientBoostingRegressor(random_state=RANDOM_STATE), "DRY-RUN (HistGB)"
    else:
        import torch
        from tabpfn import TabPFNRegressor
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = TabPFNRegressor(n_estimators=args.n_estimators, device=device,
                                random_state=RANDOM_STATE, ignore_pretraining_limits=True)
        label = f"TabPFN ({device}, contexto {len(idx)})"

    t0 = time.time()
    model.fit(A[idx], y_tt[idx])
    preds = []
    for i in range(0, len(B), args.batch):
        preds.append(model.predict(B[i:i + args.batch]))
        print(f"  {label}: {min(i + args.batch, len(B))}/{len(B)} filas predichas ({time.time() - t0:.0f} s)",
              flush=True)
    p_tab = np.concatenate(preds)
    np.save(CACHE / "tabpfn_pred.npy", p_tab)
    print(f"{label}   RMSE {rmse(d['y_th'], p_tab):.3f}   ({time.time() - t0:.0f} s)")


def compare():
    d = np.load(CACHE / "tabpfn_inputs.npz")
    y, p_xgb, p_tab = d["y_th"], d["p_xgb"], np.load(CACHE / "tabpfn_pred.npy")
    print("\n=== Comparación en train_test ===")
    print(f"XGBoost                      RMSE {rmse(y, p_xgb):.3f}")
    print(f"TabPFN                       RMSE {rmse(y, p_tab):.3f}")
    for w in (0.2, 0.3, 0.5):
        print(f"{1 - w:.1f}·XGB + {w:.1f}·TabPFN          RMSE {rmse(y, (1 - w) * p_xgb + w * p_tab):.3f}")
    print(f"Correlación de errores XGB vs TabPFN: {np.corrcoef(y - p_xgb, y - p_tab)[0, 1]:.3f}")


def main():
    args = parse_args()
    if args.stage == "features":
        return stage_features()
    if args.stage == "tabpfn":
        return stage_tabpfn(args)

    # Etapa "all": lanza cada parte en su propio proceso
    base = [sys.executable, "-u", __file__]
    extra = ["--n-context", str(args.n_context), "--device", args.device,
             "--n-estimators", str(args.n_estimators), "--batch", str(args.batch)]
    if args.dry_run:
        extra.append("--dry-run")
    subprocess.run(base + ["--stage", "features"], check=True)
    subprocess.run(base + ["--stage", "tabpfn"] + extra, check=True)
    compare()


if __name__ == "__main__":
    main()
