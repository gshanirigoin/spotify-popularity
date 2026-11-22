"""TabPFN en un proceso aparte, con caché en disco.

Por qué un proceso aparte: en macOS, XGBoost y PyTorch (que usa TabPFN) cargan cada uno su
propia copia de libomp (OpenMP); en el mismo proceso se bloquean entre sí (deadlock). Desde el
notebook, `tabpfn_predict` guarda las matrices en disco, lanza este módulo con otro intérprete
de Python (que no importa XGBoost) y recoge las predicciones.

Caché: si ya se calcularon predicciones con exactamente los mismos datos y parámetros, se
reutilizan sin volver a ejecutar TabPFN (puede tardar más de 20 minutos).

Requisito: licencia de TabPFN aceptada y clave en TABPFN_TOKEN o ~/.cache/tabpfn/auth_token.
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]


def _fingerprint(*arrays, **params):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:16]


def tabpfn_predict(A_fit, y_fit, A_apply, n_context=40_000, device="auto", n_estimators=4,
                   batch=2_000, random_state=1928, cache_dir="experiments/.cache"):
    """Entrena TabPFN con una submuestra de n_context filas de (A_fit, y_fit) y predice A_apply.

    Se ejecuta en un subproceso. Devuelve un array con una predicción por fila de A_apply.
    """
    params = dict(n_context=n_context, device=device, n_estimators=n_estimators,
                  random_state=random_state)
    key = _fingerprint(A_fit, y_fit, A_apply, **params)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    inp, out = cache / f"tabpfn_in_{key}.npz", cache / f"tabpfn_out_{key}.npy"
    if out.exists():
        print(f"TabPFN: predicciones en caché ({out.name})")
        return np.load(out)

    np.savez(inp, A_fit=A_fit, y_fit=np.asarray(y_fit, dtype=float), A_apply=A_apply)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_DIR), os.environ.get("PYTHONPATH")]))}
    cmd = [sys.executable, "-u", "-m", "spotify.tabpfn_runner", str(inp), str(out),
           str(n_context), device, str(n_estimators), str(batch), str(random_state)]
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    inp.unlink(missing_ok=True)
    print(f"TabPFN: {len(A_apply)} predicciones en {time.time() - t0:.0f} s")
    return np.load(out)


def _main(inp, out, n_context, device, n_estimators, batch, random_state):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from tabpfn import TabPFNRegressor

    d = np.load(inp)
    A, y, B = d["A_fit"], d["y_fit"], d["A_apply"]
    idx = np.random.default_rng(random_state).choice(len(A), size=min(n_context, len(A)), replace=False)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = TabPFNRegressor(n_estimators=n_estimators, device=device, random_state=random_state,
                            ignore_pretraining_limits=True)
    model.fit(A[idx], y[idx])
    t0, preds = time.time(), []
    for i in range(0, len(B), batch):
        preds.append(model.predict(B[i:i + batch]))
        print(f"  TabPFN ({device}, contexto {len(idx)}): {min(i + batch, len(B))}/{len(B)} filas "
              f"({time.time() - t0:.0f} s)", flush=True)
    np.save(out, np.concatenate(preds))


if __name__ == "__main__":
    a = sys.argv[1:]
    _main(a[0], a[1], int(a[2]), a[3], int(a[4]), int(a[5]), int(a[6]))
