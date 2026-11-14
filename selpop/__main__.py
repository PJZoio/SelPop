"""Linha de comandos: python -m selpop configuracao.yaml [--saida resultado.xlsx]"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd
from pathlib import Path

from .pipeline import load_config, run
from .report import write


def main(argv=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    ap = argparse.ArgumentParser(prog="selpop", description="Seleção reprodutível de casos e controlos a partir da base clínica.")
    ap.add_argument("config", help="ficheiro YAML de configuração do estudo")
    ap.add_argument("--saida", help="ficheiro Excel de saída (por omissão: o definido na configuração)")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = load_config(cfg_path)
    R = run(cfg, cfg_path.parent)
    out = Path(a.saida) if a.saida else cfg_path.parent / cfg.get("saida", {}).get("ficheiro", "selecao.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    info = write(R, out)
    t = R["t_after"]
    sig = t[(t["Diferença significativa"] == "Sim") & t["Papel"].astype(str).str.startswith("Confundidor")]["Variável"].tolist()
    pred = t[(t["Diferença significativa"] == "Sim") & ~t["Papel"].astype(str).str.startswith("Confundidor")]["Variável"].tolist()
    print(f"\nResultado: {out}")
    print(f"  {R['names'][0]}: {R['sel_cases'].shape[0]} | {R['names'][1]} (principais): {R['sel_ctrl'].shape[0]} | suplentes: {info['suplentes']} | casos sem controlo: {info['casos_sem_controlo']}")
    print(f"  Confundidores com p < {R['meta']['alfa']}: {', '.join(sig) if sig else 'nenhum'}")
    if pred:
        print(f"  Preditores/interferentes com p < {R['meta']['alfa']}: {', '.join(pred)}")


if __name__ == "__main__":
    main()
