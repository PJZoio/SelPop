"""Emparelhamento caso-controlo por etapas (exato + calibres), guloso global, sem reposição.

Unidade de controlo = doente-dia candidato; cada doente-controlo só pode ser usado uma vez.
Custo de um par = Σ |diferença| / calibre, para as variáveis com calibre (0 = igual, 1 = no limite).
Em cada etapa ordenam-se todos os pares admissíveis por custo (desempate aleatório reprodutível) e
atribuem-se enquanto o caso precisar de controlos e o doente-controlo estiver livre. Casos que não
completaram a razão passam à etapa seguinte (critérios mais permissivos), sem perder os já atribuídos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def match(cases: pd.DataFrame, controls: pd.DataFrame, spec: dict, idc: str, rng: np.random.Generator) -> pd.DataFrame:
    ratio = int(spec.get("razao", 1))
    n_sup = int(spec.get("suplentes_por_caso", 0))
    need_total = ratio + n_sup
    stages = spec["etapas"]
    got: dict = {cid: [] for cid in cases[idc]}
    used: set = set()
    rows = []
    ctrl = controls.reset_index().rename(columns={"index": "_row"})
    for si, st in enumerate(stages, 1):
        exact, cal = st.get("exato", []) or [], st.get("calibre", {}) or {}
        pend = cases[cases[idc].map(lambda x: len(got[x]) < need_total)]
        if pend.empty:
            break
        avail = ctrl[~ctrl[idc].isin(used)]
        if avail.empty:
            break
        pairs = []
        for _, c in pend.iterrows():
            m = np.ones(len(avail), bool)
            for v in exact:
                m &= (avail[v].values == c[v])
            cost = np.zeros(len(avail))
            for v, w in cal.items():
                d = np.abs(avail[v].values.astype(float) - float(c[v]))
                m &= np.nan_to_num(d, nan=np.inf) <= w
                cost += d / w
            if m.any():
                sub = avail[m]
                for ci, (r, pid, k) in enumerate(zip(sub["_row"].values, sub[idc].values, cost[m])):
                    pairs.append((k, rng.random(), c[idc], pid, r))
        pairs.sort(key=lambda x: (x[0], x[1]))
        for k, _, cid, pid, r in pairs:
            if len(got[cid]) >= need_total or pid in used or pid in [p for p, _ in got[cid]]:
                continue
            got[cid].append((pid, r))
            used.add(pid)
            rows.append({"caso_id": cid, "controlo_id": pid, "_row": r, "etapa": si, "custo": round(float(k), 3),
                         "criterio": _describe(exact, cal)})
    res = pd.DataFrame(rows, columns=["caso_id", "controlo_id", "_row", "etapa", "custo", "criterio"])
    if res.empty:
        res["papel"] = []
        return res
    res = res.sort_values(["caso_id", "etapa", "custo"])
    res["ordem"] = res.groupby("caso_id").cumcount() + 1
    res["papel"] = np.where(res.ordem <= ratio, "principal", "suplente")
    return res


def _describe(exact, cal):
    parts = []
    if exact:
        parts.append("exato: " + ", ".join(exact))
    if cal:
        parts.append("calibre: " + ", ".join(f"{k} ±{v:g}" for k, v in cal.items()))
    return "; ".join(parts) or "sem restrições"
