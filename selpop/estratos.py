"""Seleção estratificada: repete a seleção completa (grupos → emparelhamento → reequilíbrio → tabela)
dentro de cada estrato de uma variável (ex.: COVID-19 sim/não), com tabela e testes próprios por estrato.

Configuração (opcional):

    estratificar:
      col: COVID
      estratos:                      # opcional; por omissão, os rótulos da variável em `equilibrio`
        - {valor: 1, rotulo: "COVID-19", rotulo_en: "COVID-19", sufixo: COVpos}
        - {valor: 0, rotulo: "sem COVID-19", rotulo_en: "non-COVID-19", sufixo: COVneg}

ou, na linha de comandos, `--estratificar COVID`.
"""
from __future__ import annotations

import copy
import re

import pandas as pd


def _var_of(cfg: dict, col: str) -> dict:
    return next((v for v in cfg.get("equilibrio", {}).get("variaveis", []) if v["col"] == col), {"col": col})


def _lit(v):
    """Literal para a expressão de filtro (números sem aspas, texto com aspas)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return repr(int(v)) if float(v).is_integer() else repr(float(v))
    return repr(str(v))


def _ref(col: str) -> str:
    return col if re.fullmatch(r"[A-Za-z_]\w*", col) else f"`{col}`"


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)) or "x"


def strata(cfg: dict, df: pd.DataFrame | None = None) -> list[dict]:
    """Lista de estratos {valor, rotulo, rotulo_en, sufixo}."""
    spec = cfg.get("estratificar")
    if not spec:
        return []
    if isinstance(spec, str):
        spec = {"col": spec}
    col = spec["col"]
    v = _var_of(cfg, col)
    nome, nome_en = v.get("nome", col), v.get("nome_en", v.get("nome", col))
    labs, labs_en = v.get("rotulos") or {}, v.get("rotulos_en") or {}
    if spec.get("estratos"):
        items = [dict(e) for e in spec["estratos"]]
    else:
        vals = list(labs.keys())
        if not vals and df is not None and col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist(), key=lambda z: (str(type(z)), z))
        if not vals:
            raise ValueError(f"Estratificar por '{col}': indique os estratos (valores) ou rótulos da variável.")
        items = [{"valor": x} for x in vals]
    yes_no_en = {"Sim": "Yes", "Não": "No"}
    for e in items:
        x = e["valor"]
        lab = labs.get(x, labs.get(str(x), x))
        lab_en = labs_en.get(x, labs_en.get(str(x), yes_no_en.get(str(lab), lab)))
        e.setdefault("rotulo", f"{nome}: {lab}")
        e.setdefault("rotulo_en", f"{nome_en}: {lab_en}")
        e.setdefault("sufixo", f"{_slug(col)}{_slug(x)}")
        e["col"] = col
    return items


def cfg_for(cfg: dict, e: dict) -> dict:
    """Configuração do estrato `e`: critério de estrato nos dois grupos; a variável de estrato sai
    da tabela (é constante) e dos candidatos a reequilíbrio."""
    c = copy.deepcopy(cfg)
    c.pop("estratificar", None)
    col = e["col"]
    crit = {"nome": f"Estrato: {e['rotulo']}", "expr": f"{_ref(col)} == {_lit(e['valor'])}"}
    for g in ("casos", "controlos"):
        gg = c["grupos"][g]
        d = gg.get("doente")
        if d is None or d == "":
            d = []
        elif isinstance(d, str):
            d = [d]
        gg["doente"] = [crit] + list(d)
    eq = c.setdefault("equilibrio", {})
    eq["variaveis"] = [v for v in eq.get("variaveis", []) if v["col"] != col]
    reeq = (c.get("emparelhamento") or {}).get("reequilibrar")
    if reeq and reeq.get("candidatas"):
        reeq["candidatas"] = [x for x in reeq["candidatas"] if x != col]
    est = c.setdefault("estudo", {})
    est["nome"] = f"{est.get('nome', 'Estudo')} — {e['rotulo']}"
    est["estrato"] = {k: e[k] for k in ("col", "valor", "rotulo", "rotulo_en", "sufixo")}
    return c


def expand(cfg: dict, df: pd.DataFrame | None = None) -> list[tuple[str | None, dict]]:
    """[(sufixo, cfg)] — um elemento por estrato, ou [(None, cfg)] sem estratificação."""
    S = strata(cfg, df)
    return [(e["sufixo"], cfg_for(cfg, e)) for e in S] if S else [(None, cfg)]


def with_strata(cfg: dict, col: str | None) -> dict:
    """Acrescenta `estratificar: col` (linha de comandos), respeitando uma definição já existente."""
    if not col:
        return cfg
    c = copy.deepcopy(cfg)
    if not (isinstance(c.get("estratificar"), dict) and c["estratificar"].get("col") == col):
        c["estratificar"] = {"col": col}
    return c
