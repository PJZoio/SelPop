"""Emparelhamento caso-controlo por etapas (exato + calibres), sem reposição.

Unidade de controlo = doente-dia candidato; cada doente-controlo só pode ser usado uma vez.
Custo de um par = Σ |diferença| / calibre, para as variáveis com calibre (0 = igual, 1 = no limite).

Algoritmos (`emparelhamento.algoritmo`):
- "otimo" (por omissão): por RONDAS — 1.º controlo para todos os casos, depois o 2.º, …, depois os
  suplentes; em cada ronda e etapa, atribuição ótima (algoritmo húngaro): o maior número possível de casos
  recebe um controlo e, entre essas soluções, a de menor custo total.
- "rondas": as mesmas rondas, mas cada ronda é gulosa (pares por ordem de custo).
- "guloso": versão até à v0.6 — guloso global; um caso pode completar 2 controlos + suplente antes de outros
  casos terem algum. Mantido para reproduzir seleções antigas.
Em todos, casos sem controlo numa etapa passam à seguinte (critérios mais permissivos). Desempates
aleatórios reprodutíveis (semente do estudo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ALGORITMOS = {"otimo": "por rondas, atribuição ótima", "rondas": "por rondas, guloso", "guloso": "guloso global (v0.6)"}


def match(cases: pd.DataFrame, controls: pd.DataFrame, spec: dict, idc: str, rng: np.random.Generator) -> pd.DataFrame:
    alg = spec.get("algoritmo", "otimo")
    if alg not in ALGORITMOS:
        raise ValueError(f"emparelhamento.algoritmo deve ser um de {list(ALGORITMOS)}")
    if alg != "guloso":
        return _match_rounds(cases, controls, spec, idc, rng, optimal=(alg == "otimo"))
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


def _cost(c, avail, exact, cal):
    m = np.ones(len(avail), bool)
    for v in exact:
        m &= (avail[v].values == c[v])
    cost = np.zeros(len(avail))
    for v, w in cal.items():
        d = np.abs(avail[v].values.astype(float) - float(c[v]))
        m &= np.nan_to_num(d, nan=np.inf) <= w
        cost += d / w
    return m, cost


def _match_rounds(cases, controls, spec, idc, rng, optimal=True):
    ratio = int(spec.get("razao", 1))
    need_total = ratio + int(spec.get("suplentes_por_caso", 0))
    ctrl = controls.reset_index().rename(columns={"index": "_row"})
    got: dict = {cid: [] for cid in cases[idc]}
    used: set = set()
    rows = []
    for k in range(1, need_total + 1):                      # ronda k: k-ésimo controlo de cada caso
        for si, st in enumerate(spec["etapas"], 1):
            exact, cal = st.get("exato", []) or [], st.get("calibre", {}) or {}
            pend = cases[cases[idc].map(lambda x: len(got[x]) < k)]
            avail = ctrl[~ctrl[idc].isin(used)]
            if pend.empty or avail.empty:
                continue
            pids = avail[idc].unique()
            col_of = {p: j for j, p in enumerate(pids)}
            C = np.full((len(pend), len(pids)), np.inf)
            R = np.full((len(pend), len(pids)), -1, dtype=np.int64)
            pid_v, row_v = avail[idc].values, avail["_row"].values
            for i, (_, c) in enumerate(pend.iterrows()):
                m, cost = _cost(c, avail, exact, cal)
                mine = {p for p, _ in got[c[idc]]}
                for j0 in np.flatnonzero(m):                 # melhor dia de cada doente-controlo para este caso
                    p = pid_v[j0]
                    if p in mine:
                        continue
                    j = col_of[p]
                    if cost[j0] < C[i, j]:
                        C[i, j], R[i, j] = cost[j0], row_v[j0]
            if not np.isfinite(C).any():
                continue
            J = C + rng.random(C.shape) * 1e-9               # desempate reprodutível
            if optimal:
                BIG = 1e6
                r_ix, c_ix = linear_sum_assignment(np.where(np.isfinite(J), J, BIG))
                pairs = [(i, j) for i, j in zip(r_ix, c_ix) if np.isfinite(C[i, j])]
            else:
                fin = np.argwhere(np.isfinite(J))
                pairs, ri, cj = [], set(), set()
                for i, j in sorted(map(tuple, fin), key=lambda ij: J[ij]):
                    if i not in ri and j not in cj:
                        pairs.append((i, j)); ri.add(i); cj.add(j)
            for i, j in pairs:
                cid, pid = pend[idc].iloc[i], pids[j]
                got[cid].append((pid, int(R[i, j])))
                used.add(pid)
                rows.append({"caso_id": cid, "controlo_id": pid, "_row": int(R[i, j]), "etapa": si,
                             "custo": round(float(C[i, j]), 3), "criterio": _describe(exact, cal), "ordem": k})
    res = pd.DataFrame(rows, columns=["caso_id", "controlo_id", "_row", "etapa", "custo", "criterio", "ordem"])
    if res.empty:
        res["papel"] = []
        return res
    res = res.sort_values(["caso_id", "ordem"]).reset_index(drop=True)
    res["papel"] = np.where(res.ordem <= ratio, "principal", "suplente")
    return res


def _describe(exact, cal):
    parts = []
    if exact:
        parts.append("exato: " + ", ".join(exact))
    if cal:
        parts.append("calibre: " + ", ".join(f"{k} ±{v:g}" for k, v in cal.items()))
    return "; ".join(parts) or "sem restrições"
