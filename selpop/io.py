"""Carregamento da base principal e de ficheiros auxiliares (ex.: IMC), com registo de proveniência."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(cfg: dict, base_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Lê a base principal e junta os ficheiros auxiliares por ID. Devolve (dados, proveniência)."""
    src = cfg["dados"]
    prov = []
    p = (base_dir / src["ficheiro"]).resolve()
    df = pd.read_excel(p, sheet_name=src.get("folha", 0))
    prov.append({"ficheiro": str(p.name), "folha": str(src.get("folha", 0)), "linhas": len(df), "sha256": sha256(p)})

    # normalizar "N.A" e afins para NaN nas colunas indicadas como numéricas
    for col in src.get("colunas_numericas_com_texto", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    id_col = src["id"]
    for aux in src.get("auxiliares", []) or []:
        pa = (base_dir / aux["ficheiro"]).resolve()
        a = pd.read_excel(pa, sheet_name=aux.get("folha", 0))
        a = a.rename(columns={aux["id"]: id_col})
        keep = [id_col] + list(aux["colunas"].keys())
        a = a[keep].rename(columns=aux["colunas"]).drop_duplicates(id_col)
        df = df.merge(a, on=id_col, how="left")
        prov.append({"ficheiro": str(pa.name), "folha": str(aux.get("folha", 0)), "linhas": len(a), "sha256": sha256(pa),
                     "nota": f"junção por {aux['id']} → {id_col}; colunas: {', '.join(aux['colunas'].values())}"})
    return df, prov


def ensure_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


__all__ = ["load", "sha256", "ensure_datetime", "to_num", "np"]
