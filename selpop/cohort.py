"""Definição de grupos (casos/controlos), funil de elegibilidade e escolha do dia-índice."""
from __future__ import annotations

import numpy as np
import pandas as pd


def q(df: pd.DataFrame, expr: str | None) -> pd.Series:
    """Avalia uma expressão do ficheiro de configuração (sintaxe pandas.query, motor python)."""
    if not expr or str(expr).strip() in ("", "True", "true"):
        return pd.Series(True, index=df.index)
    m = df.eval(expr, engine="python")
    if not isinstance(m, pd.Series):
        m = pd.Series(bool(m), index=df.index)
    return m.fillna(False).astype(bool)


def _criteria(spec):
    """Aceita uma expressão única, uma lista de expressões ou uma lista de {nome, expr}.
    Com lista, o funil mostra o efeito cumulativo de cada critério."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, str):
        return [(spec, spec)]
    out = []
    for item in spec:
        if isinstance(item, dict):
            out.append((f"{item.get('nome', item['expr'])}  ({item['expr']})", item["expr"]))
        else:
            out.append((item, item))
    return out


def add_expressions(df: pd.DataFrame, exprs: dict | None) -> pd.DataFrame:
    for name, expr in (exprs or {}).items():
        df[name] = df.eval(expr, engine="python")
    return df


def pick_index(g: pd.DataFrame, rule: dict | None, rng: np.random.Generator) -> pd.Series:
    """Escolhe o dia-índice de um doente entre os dias elegíveis `g`."""
    rule = rule or {"regra": "primeiro"}
    r, col = rule.get("regra", "primeiro"), rule.get("coluna")
    if r == "primeiro":
        return g.iloc[0]
    if r == "ultimo":
        return g.iloc[-1]
    if r == "minimo":
        return g.loc[g[col].idxmin()]
    if r == "maximo":
        return g.loc[g[col].idxmax()]
    if r == "alvo":
        d = (g[col] - rule["alvo"]).abs()
        return g.loc[d.idxmin()]
    if r == "aleatorio":
        return g.iloc[int(rng.integers(len(g)))]
    raise ValueError(f"Regra de índice desconhecida: {r}")


def build_groups(df: pd.DataFrame, cfg: dict, rng: np.random.Generator):
    """Devolve (casos_indice, controlos_candidatos, funil). Casos: 1 linha por doente (dia-índice).
    Controlos: todas as linhas (doente-dia) elegíveis; o dia é escolhido no emparelhamento."""
    idc = cfg["dados"]["id"]
    G = cfg["grupos"]
    funil = []

    def stage(label, grupo, mask):
        sub = df[mask]
        funil.append({"Grupo": grupo, "Passo": label, "Doentes": int(sub[idc].nunique()), "Dias/amostras": int(len(sub))})

    out = {}
    for key in ("casos", "controlos"):
        gdef = G[key]
        nome = gdef.get("nome", key)
        m_all = pd.Series(True, index=df.index)
        stage("Base completa", nome, m_all)
        m = m_all
        for nivel in ("doente", "dia"):
            for lab, expr in _criteria(gdef.get(nivel)):
                m = m & q(df, expr)
                stage(f"[{nivel}] {lab}", nome, m)
        out[key] = df[m].copy()

    # exclusividade: um doente não pode ser caso e controlo
    both = set(out["casos"][idc]) & set(out["controlos"][idc])
    if both:
        out["controlos"] = out["controlos"][~out["controlos"][idc].isin(both)]
        funil.append({"Grupo": G["controlos"].get("nome", "controlos"), "Passo": "Removidos por também serem casos",
                      "Doentes": int(out["controlos"][idc].nunique()), "Dias/amostras": int(len(out["controlos"]))})

    out["casos"]["_linha"] = out["casos"].index          # linha original da base (rastreabilidade)
    cases = out["casos"].groupby(idc, group_keys=False).apply(lambda g: pick_index(g, G["casos"].get("indice"), rng))
    cases = cases.reset_index(drop=True) if idc in cases.columns else cases.reset_index()
    funil.append({"Grupo": G["casos"].get("nome", "casos"),
                  "Passo": f"Dia-índice ({(G['casos'].get('indice') or {}).get('regra', 'primeiro')}) — 1 por doente",
                  "Doentes": int(cases[idc].nunique()), "Dias/amostras": int(len(cases))})
    return cases, out["controlos"], funil
