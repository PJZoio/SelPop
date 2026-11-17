"""Carregamento da base principal e de ficheiros auxiliares (ex.: IMC), com registo de proveniência."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_table(buf, name: str, sheet):
    """Excel (.xlsx/.xls) ou texto delimitado (.csv/.tsv/.txt; separador detetado automaticamente)."""
    if str(name).lower().endswith((".csv", ".tsv", ".txt")):
        return pd.read_csv(buf, sep=None, engine="python")
    return pd.read_excel(buf, sheet_name=sheet)


def _read(src, sheet):
    """`src` é um caminho ou um par (nome, bytes) — este último para ficheiros carregados na interface."""
    if isinstance(src, (str, Path)):
        p = Path(src).resolve()
        return _read_table(p, p.name, sheet), p.name, sha256(p)
    name, data = src
    return _read_table(io.BytesIO(data), name, sheet), name, hashlib.sha256(data).hexdigest()


def load_sources(cfg: dict, main_src, aux_srcs: list | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Lê a base principal e junta os auxiliares por ID. `aux_srcs[i]` corresponde a `dados.auxiliares[i]`;
    None salta esse auxiliar. Devolve (dados, proveniência)."""
    src = cfg["dados"]
    df, name, h = _read(main_src, src.get("folha", 0))
    prov = [{"ficheiro": name, "folha": str(src.get("folha", 0)), "linhas": len(df), "sha256": h}]

    # normalizar "N.A" e afins para NaN nas colunas indicadas como numéricas
    for col in src.get("colunas_numericas_com_texto", []) or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    id_col = src["id"]
    auxs = src.get("auxiliares", []) or []
    aux_srcs = aux_srcs if aux_srcs is not None else [None] * len(auxs)
    for aux, asrc in zip(auxs, aux_srcs):
        if asrc is None:
            continue
        a, aname, ah = _read(asrc, aux.get("folha", 0))
        a = a.rename(columns={aux["id"]: id_col})
        keep = [id_col] + list(aux["colunas"].keys())
        a = a[keep].rename(columns=aux["colunas"]).drop_duplicates(id_col)
        df = df.merge(a, on=id_col, how="left")
        prov.append({"ficheiro": aname, "folha": str(aux.get("folha", 0)), "linhas": len(a), "sha256": ah,
                     "nota": f"junção por {aux['id']} → {id_col}; colunas: {', '.join(aux['colunas'].values())}"})
    return df, prov


def load(cfg: dict, base_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Versão de linha de comandos: caminhos relativos à pasta do ficheiro de configuração."""
    src = cfg["dados"]
    main = base_dir / src["ficheiro"]
    aux = [base_dir / a["ficheiro"] for a in (src.get("auxiliares", []) or [])]
    return load_sources(cfg, main, aux)


def ensure_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


__all__ = ["load", "sha256", "ensure_datetime", "to_num", "np"]
