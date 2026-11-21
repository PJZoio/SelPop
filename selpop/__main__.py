"""Linha de comandos: python -m selpop configuracao.yaml [--saida resultado.xlsx]"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd
from pathlib import Path

from .estratos import expand, with_strata
from .io import load
from .pipeline import load_config, prepare, select
from .report import write


def main(argv=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    ap = argparse.ArgumentParser(prog="selpop", description="Seleção reprodutível de casos e controlos a partir da base clínica.")
    ap.add_argument("config", help="ficheiro YAML de configuração do estudo")
    ap.add_argument("--saida", help="ficheiro Excel de saída (por omissão: o definido na configuração)")
    ap.add_argument("--lingua", choices=["en", "pt"], default="en", help="língua da tabela de publicação (por omissão: en)")
    ap.add_argument("--publicacao", action="store_true", help="gravar também a tabela de características em Word (.docx)")
    ap.add_argument("--sav", action="store_true", help="gravar também os dados em SPSS (.sav) + sintaxe de validação (.sps)")
    ap.add_argument("--estratificar", metavar="COLUNA", help="repetir a seleção dentro de cada estrato desta coluna (ex.: COVID); um Excel por estrato")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = with_strata(load_config(cfg_path), a.estratificar)
    raw, prov = load(cfg, cfg_path.parent)
    print(f"Base: {len(raw)} linhas, {raw[cfg['dados']['id']].nunique()} doentes")
    df, qc = prepare(cfg, raw)
    out0 = Path(a.saida) if a.saida else cfg_path.parent / cfg.get("saida", {}).get("ficheiro", "selecao.xlsx")
    out0.parent.mkdir(parents=True, exist_ok=True)
    for suf, c in expand(cfg, df):
        if suf:
            print(f"\n=== Estrato: {c['estudo']['estrato']['rotulo']} ===")
        R = select(c, df.copy(), qc, prov)
        out = out0.with_name(f"{out0.stem}_{suf}{out0.suffix}") if suf else out0
        _write_all(R, out, a)


def _write_all(R, out, a):
    info = write(R, out, pub_lang=a.lingua)
    from .publish import write_docx, write_sav, write_sps
    if a.publicacao:
        print(f"  Tabela de características (Word): {write_docx(R, out.with_name(out.stem + f'_tabela_caracteristicas_{a.lingua}.docx'), lang=a.lingua)}")
    if a.sav:
        sav = write_sav(R, out.with_suffix('.sav'), lang=a.lingua)
        print(f"  SPSS: {sav} + {write_sps(R, out.with_name(out.stem + '_validacao.sps'), Path(sav).name, lang=a.lingua)}")
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
