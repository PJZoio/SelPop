"""Orquestração: carregar → derivar → grupos → emparelhar → Tabela 1 → reequilibrar → resultado."""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .cohort import add_expressions, build_groups
from .derive import derive, quality_checks
from .io import load
from .match import match
from .stats import table1


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _balance_vars(cfg):
    return [v for v in cfg["equilibrio"]["variaveis"]]


def _assemble(cases, controls, m, idc, policy):
    prin = m[m.papel == "principal"]
    matched_ids = set(prin.caso_id)
    sel_cases = cases[cases[idc].isin(matched_ids)] if policy == "excluir" else cases
    sel_ctrl = controls.loc[prin._row.values].copy()
    sel_ctrl["_caso"] = prin.caso_id.values
    return sel_cases, sel_ctrl


def run(cfg: dict, base_dir: Path, log=print) -> dict:
    seed = int(cfg["estudo"].get("semente", 12345))
    rng = np.random.default_rng(seed)
    idc = cfg["dados"]["id"]
    alpha = float(cfg["equilibrio"].get("alfa", 0.05))

    raw, prov = load(cfg, base_dir)
    log(f"Base: {len(raw)} linhas, {raw[idc].nunique()} doentes")
    df = derive(raw, cfg)
    df = add_expressions(df, cfg.get("derivados"))
    qc = quality_checks(df, cfg)
    cases, controls, funil = build_groups(df, cfg, rng)
    log(f"Casos elegíveis: {len(cases)} | controlos candidatos: {controls[idc].nunique()} doentes / {len(controls)} dias")

    names = (cfg["grupos"]["casos"].get("nome", "Casos"), cfg["grupos"]["controlos"].get("nome", "Controlos"))
    bvars = _balance_vars(cfg)

    # equilíbrio antes do emparelhamento (controlos: 1.º dia elegível por doente)
    ctrl_first = controls.groupby(idc, group_keys=False).head(1)
    t_before = table1(cases, ctrl_first, bvars, alpha, np.random.default_rng(seed), names)

    spec = copy.deepcopy(cfg.get("emparelhamento"))
    iters, results = [], []
    if spec:
        reeq = spec.get("reequilibrar", {}) or {}
        cands = reeq.get("candidatas", []) if reeq.get("ativo") else []
        pool = pd.concat([cases, controls])
        for it in range(int(reeq.get("max_iter", 0)) + 1 if cands else 1):
            m = match(cases, controls, spec, idc, np.random.default_rng(seed))
            sc, sk = _assemble(cases, controls, m, idc, spec.get("casos_sem_controlo", "excluir"))
            t = table1(sc, sk, bvars, alpha, np.random.default_rng(seed), names)
            tested = t[t.p.notna()]
            label_of = {v.get("nome", v["col"]): v for v in bvars}
            sig = [label_of[l] for l in tested[tested.p < alpha]["Variável"] if l in label_of]
            results.append((m, sc, sk, t, copy.deepcopy(spec)))
            iters.append({"Iteração": it, "Casos emparelhados": int(sc[idc].nunique()), "Controlos (principais)": len(sk),
                          "Variáveis com p < α": ", ".join(v.get("nome", v["col"]) for v in sig) or "nenhuma",
                          "N.º variáveis com p < α": len(sig),
                          "Emparelhamento usado": " | ".join(_stage_txt(s) for s in spec["etapas"])})
            todo = [v for v in sig if v["col"] in cands]
            if not todo or it == int(reeq.get("max_iter", 0)):
                break
            worst = min(todo, key=lambda v: float(tested.loc[tested["Variável"] == v.get("nome", v["col"]), "p"].iloc[0]))
            col = worst["col"]
            for s in spec["etapas"]:
                if worst.get("tipo", "continua") == "continua":
                    sd = float(pd.to_numeric(pool[col], errors="coerce").std())
                    w = max(round(float(reeq.get("fator_calibre", 0.5)) * sd, 2), 1e-6)
                    s.setdefault("calibre", {})
                    s["calibre"] = {**(s["calibre"] or {}), col: w}
                else:
                    s["exato"] = list(dict.fromkeys((s.get("exato") or []) + [col]))
            log(f"Reequilíbrio: iteração {it + 1} acrescenta {col} ao emparelhamento")
        # melhor iteração: menos variáveis significativas, depois mais casos emparelhados, depois a mais cedo
        best = min(range(len(results)), key=lambda i: (iters[i]["N.º variáveis com p < α"], -iters[i]["Casos emparelhados"], i))
        for i, r in enumerate(iters):
            r["Escolhida"] = "Sim" if i == best else ""
        m, sel_cases, sel_ctrl, t_after, spec_used = results[best]
    else:  # sem emparelhamento: controlos pelo dia-índice próprio
        from .cohort import pick_index
        rule = cfg["grupos"]["controlos"].get("indice")
        sel_ctrl = controls.groupby(idc, group_keys=False).apply(lambda g: pick_index(g, rule, rng))
        sel_ctrl["_caso"] = None
        sel_cases = cases
        m = pd.DataFrame(columns=["caso_id", "controlo_id", "_row", "etapa", "custo", "criterio", "papel"])
        t_after = table1(sel_cases, sel_ctrl, bvars, alpha, np.random.default_rng(seed), names)
        spec_used = None

    funil.append({"Grupo": names[0], "Passo": "Selecionados (com ≥1 controlo principal)" if spec else "Selecionados",
                  "Doentes": int(sel_cases[idc].nunique()), "Dias/amostras": len(sel_cases)})
    funil.append({"Grupo": names[1], "Passo": "Selecionados (controlos principais)", "Doentes": int(sel_ctrl[idc].nunique()),
                  "Dias/amostras": len(sel_ctrl)})

    meta = {"estudo": cfg["estudo"].get("nome", ""), "descricao": cfg["estudo"].get("descricao", ""),
            "gerado_em": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "versao": __version__, "semente": seed,
            "alfa": alpha, "proveniencia": prov}
    return {"df": df, "cases": cases, "controls": controls, "match": m, "sel_cases": sel_cases, "sel_ctrl": sel_ctrl,
            "t_before": t_before, "t_after": t_after, "funil": pd.DataFrame(funil), "iters": pd.DataFrame(iters),
            "spec_used": spec_used, "meta": meta, "qc": qc, "names": names, "cfg": cfg}


def _stage_txt(s):
    ex = ", ".join(s.get("exato") or []) or "—"
    ca = ", ".join(f"{k}±{v:g}" for k, v in (s.get("calibre") or {}).items()) or "—"
    return f"exato[{ex}] calibre[{ca}]"
