"""Corre vários estudos sobre a mesma base e junta-os num só Excel, com a lista única de amostras a pedir.

    python -m selpop.consolidar estudos/prognostico_lra.yaml estudos/diagnostico_lra.yaml estudos/diagnostico_drc.yaml \
        --rotulos M1 M2 M3 --saida resultados/Selecao_amostras_biobanco_UCI_miniNIR.xlsx
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

from .io import load
from .pipeline import load_config, prep_key_of, prepare, select
from .report import FILL_IN, FLAG, METODOS, F, _sel_rows, _style

FILL_ROW = {"caso": PatternFill("solid", fgColor="FCE4D6"), "controlo": PatternFill("solid", fgColor="DDEBF7")}


def run_many(cfg_paths: list[Path], log=print) -> list[dict]:
    """Corre os estudos reaproveitando a base preparada quando a preparação é igual."""
    cache, out = {}, []
    for p in cfg_paths:
        cfg = load_config(p)
        k = prep_key_of(cfg)
        if k not in cache:
            raw, prov = load(cfg, p.parent)
            df, qc = prepare(cfg, raw)
            cache[k] = (df, qc, prov)
        df, qc, prov = cache[k]
        log(f"A correr {cfg['estudo'].get('nome', p.stem)}…")
        out.append(select(cfg, df.copy(), qc, prov, log=lambda *_: None))
    return out


def _stage_roles(spec) -> dict:
    """Variável → descrição do papel no emparelhamento final (por etapa)."""
    roles = {}
    for i, s in enumerate((spec or {}).get("etapas", []), 1):
        for c in s.get("exato") or []:
            roles.setdefault(c, []).append(f"E{i}: exato")
        for c, w in (s.get("calibre") or {}).items():
            roles.setdefault(c, []).append(f"E{i}: ±{w:g}")
    return {k: "; ".join(v) for k, v in roles.items()}


def summary(Rs, labels) -> pd.DataFrame:
    rows = []
    for lab, R in zip(labels, Rs):
        t = R["t_after"]
        conf = t["Papel"].astype(str).str.startswith("Confundidor")
        sig = t["Diferença significativa"] == "Sim"
        m = R["match"]
        idc = R["cfg"]["dados"]["id"]
        rows.append({
            "Modelo": lab, "Estudo": R["meta"]["estudo"],
            "Casos": R["names"][0], "Controlos": R["names"][1],
            "Casos elegíveis": len(R["cases"]), "Casos emparelhados": int(R["sel_cases"][idc].nunique()),
            "Controlos principais": len(R["sel_ctrl"]), "Suplentes": int((m.papel == "suplente").sum()) if len(m) else 0,
            "Casos sem controlo": len(R["cases"]) - int(R["sel_cases"][idc].nunique()) if R["cfg"].get("emparelhamento") else 0,
            "Confundidores com p < α": ", ".join(t.loc[conf & sig, "Variável"]) or "nenhum",
            "Preditores/interferentes com p < α": ", ".join(t.loc[~conf & sig & t["Variável"].ne(""), "Variável"]) or "nenhum",
            "Emparelhamento final": " | ".join(
                f"E{i}: exato[{', '.join(s.get('exato') or [])}] calibre[{', '.join(f'{k}±{v:g}' for k, v in (s.get('calibre') or {}).items())}]"
                for i, s in enumerate((R["spec_used"] or {}).get("etapas", []), 1)),
            "α": R["meta"]["alfa"], "Semente": R["meta"]["semente"],
        })
    return pd.DataFrame(rows)


def variables(Rs, labels) -> pd.DataFrame:
    """Uma linha por variável; uma coluna por modelo com o papel nesse modelo."""
    from .stats import papel
    names, order = {}, []
    per = {}
    for lab, R in zip(labels, Rs):
        roles = _stage_roles(R["spec_used"])
        d = {}
        for v in R["cfg"]["equilibrio"]["variaveis"]:
            col = v["col"]
            if col not in names:
                names[col] = v.get("nome", col)
                order.append(col)
            r = papel(v)
            if col in roles:
                r = f"Emparelhamento ({roles[col]}) + {r.split(' (')[0].lower()}"
            d[col] = r
        for col, r in roles.items():
            if col not in d:
                if col not in names:
                    names[col] = col
                    order.append(col)
                d[col] = f"Emparelhamento ({r}); não está na Tabela 1"
        per[lab] = d
    rows = [{"Variável": names[c], "Coluna": c, **{lab: per[lab].get(c, "—") for lab in labels}} for c in order]
    return pd.DataFrame(rows)


def request_list(Rs, labels) -> pd.DataFrame:
    """Lista única de amostras (todas as que algum modelo usa), com o papel em cada modelo."""
    parts = []
    for lab, R in zip(labels, Rs):
        for kind in ("principal", "suplente"):
            s = _sel_rows(R, kind)
            if not len(s):
                continue
            s = s[["ID", "Nº amostra", "Data colheita", "Papel", "Conjunto (ID do caso)"]].copy()
            s["Modelo"] = lab
            parts.append(s)
    a = pd.concat(parts, ignore_index=True)
    a["uso"] = a.Modelo + ": " + a.Papel.str.replace("controlo principal", "controlo")
    prio = a.Papel.map({"caso": 0, "controlo principal": 1, "controlo": 1, "controlo suplente": 2}).fillna(3)
    a["_prio"] = prio
    g = a.sort_values(["Modelo"]).groupby("Nº amostra")
    out = g.agg(ID=("ID", "first"), Data=("Data colheita", "first"), Usos=("uso", lambda x: "; ".join(x)),
                Prioridade=("_prio", "min"), N_modelos=("Modelo", "nunique")).reset_index()
    out["Prioridade"] = out.Prioridade.map({0: "1 — caso", 1: "2 — controlo principal", 2: "3 — suplente"})
    for lab in labels:
        out[lab] = [next((u.split(": ", 1)[1] for u in us.split("; ") if u.startswith(lab + ":")), "") for us in out.Usos]
    out = out.sort_values(["Prioridade", "ID", "Data"]).reset_index(drop=True)
    out.insert(0, "Seq", [f"P{i + 1:03d}" for i in range(len(out))])
    out = out.rename(columns={"Data": "Data colheita", "N_modelos": "N.º modelos"}).drop(columns="Usos")
    for k in ("Disponível no biobanco?", "Volume (µL)", "Medido miniNIR?", "Medido FT-NIR?", "Notas"):
        out[k] = ""
    return out


def write_many(Rs, labels, out_path: Path, intro: list[tuple[str, str]] | None = None) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    S, V, P = summary(Rs, labels), variables(Rs, labels), request_list(Rs, labels)
    meta = Rs[0]["meta"]
    leia = [("Objetivo", "Seleção das amostras do biobanco UCI a medir no miniNIR (e FT-NIR) para os modelos abaixo, "
                         "com os respetivos controlos, a Tabela 1 de cada modelo e a lista única de amostras a pedir.")]
    if intro is None:
        intro = []
        for lab, R in zip(labels, Rs):
            e = R["cfg"]["estudo"]
            intro.append((e.get('nome', lab), e.get("descricao", "")))
            for n in e.get("notas", []) or []:
                intro.append((f"{lab} — nota", n))
    leia += intro
    leia += [("Gerado em", meta["gerado_em"]), ("Versão selpop", meta["versao"])]
    for p in meta["proveniencia"]:
        leia.append(("Ficheiro de entrada", f"{p['ficheiro']} [{p['linhas']} linhas] SHA-256 {p['sha256'][:16]}… {p.get('nota', '')}"))
    leia.append(("Folhas", "Resumo · Variaveis (papel de cada variável em cada modelo) · Pedido_biobanco (amostras únicas, "
                           "por prioridade) · Qualidade_dados · por modelo: Mx_Funil, Mx_Tabela1, Mx_Selecao, Mx_Suplentes, Mx_Config"))
    leia.append(("Papéis das variáveis", "Confundidor: deve estar equilibrado entre grupos (conta para o reequilíbrio). "
                                         "Preditor candidato: testado mas NÃO emparelhado — emparelhar anularia o sinal a estudar. "
                                         "Interferente analítico: pode alterar o espetro NIR (hemólise, bilirrubina, lesão muscular); monitorizar. "
                                         "Descritiva: posterior à amostra ou só informativa."))
    leia += [("", "")] + METODOS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
            pd.DataFrame(leia, columns=["Item", "Descrição"]).to_excel(xw, sheet_name="LEIA-ME", index=False)
            S.to_excel(xw, sheet_name="Resumo", index=False)
            V.to_excel(xw, sheet_name="Variaveis", index=False)
            P.to_excel(xw, sheet_name="Pedido_biobanco", index=False)
            Rs[0]["qc"].to_excel(xw, sheet_name="Qualidade_dados", index=False)
            for lab, R in zip(labels, Rs):
                R["funil"].to_excel(xw, sheet_name=f"{lab}_Funil", index=False)
                R["t_after"].to_excel(xw, sheet_name=f"{lab}_Tabela1", index=False, startrow=2)
                _sel_rows(R, "principal").to_excel(xw, sheet_name=f"{lab}_Selecao", index=False)
                sup = _sel_rows(R, "suplente")
                (sup if len(sup) else pd.DataFrame({"Nota": ["Sem suplentes"]})).to_excel(xw, sheet_name=f"{lab}_Suplentes", index=False)
                pd.DataFrame({"Configuração (YAML)": yaml.safe_dump(R["cfg"], allow_unicode=True, sort_keys=False).splitlines()}) \
                    .to_excel(xw, sheet_name=f"{lab}_Config", index=False)

    wb = load_workbook(out_path)
    _style(wb["LEIA-ME"], widths={"A": 30, "B": 130}, wrap=("A", "B"))
    _style(wb["Resumo"], widths={"A": 8, "B": 36, "C": 20, "D": 24, "E": 10, "F": 11, "G": 11, "H": 10, "I": 10,
                                 "J": 30, "K": 40, "L": 70, "M": 6, "N": 10}, wrap=("B", "J", "K", "L"))
    wv = wb["Variaveis"]
    _style(wv, widths={"A": 46, "B": 26, **{chr(ord("C") + i): 44 for i in range(len(labels))}}, wrap=tuple(chr(ord("C") + i) for i in range(len(labels))))
    for row in wv.iter_rows(min_row=2):
        for c in row[2:]:
            v = str(c.value or "")
            if v.startswith("Emparelhamento"):
                c.fill = PatternFill("solid", fgColor="DDEBF7")
            elif v.startswith("Preditor"):
                c.fill = PatternFill("solid", fgColor="E2EFDA")
            elif v.startswith("Interferente"):
                c.fill = PatternFill("solid", fgColor="FFF2CC")
    wp = wb["Pedido_biobanco"]
    _style(wp, widths={"A": 6, "B": 11, "C": 8, "D": 12, "E": 20, "F": 9, **{chr(ord("G") + i): 18 for i in range(len(labels))}})
    wp.auto_filter.ref = wp.dimensions
    hdr = [c.value for c in wp[1]]
    for r in range(2, wp.max_row + 1):
        wp.cell(r, hdr.index("Data colheita") + 1).number_format = "dd-mm-yyyy"
        for k in ("Disponível no biobanco?", "Volume (µL)", "Medido miniNIR?", "Medido FT-NIR?", "Notas"):
            wp.cell(r, hdr.index(k) + 1).fill = FILL_IN
    _style(wb["Qualidade_dados"], widths={"A": 48, "B": 12, "C": 100}, wrap=("C",))
    for lab, R in zip(labels, Rs):
        _style(wb[f"{lab}_Funil"], widths={"A": 26, "B": 80, "C": 10, "D": 14}, wrap=("B",))
        ws = wb[f"{lab}_Tabela1"]
        ws["A1"] = f"{lab} — {R['meta']['estudo']}: {R['names'][0]} vs {R['names'][1]} (após emparelhamento). α = {R['meta']['alfa']}"
        ws["A1"].font = Font(name=F, bold=True, size=12, color="1F3864")
        _style(ws, header_row=3, widths={"A": 40, "B": 18, "C": 19, "D": 19, "E": 16, "F": 8, "G": 8, "H": 11, "I": 30, "J": 11, "K": 11})
        ws.freeze_panes = "A4"
        h = {c.value: c.column for c in ws[3]}
        for row in ws.iter_rows(min_row=4):
            pc, sc = row[h["p"] - 1], row[h["SMD"] - 1]
            if isinstance(pc.value, float):
                pc.number_format = "0.000"
                if pc.value < R["meta"]["alfa"]:
                    for c in row:
                        c.fill = FLAG
            if isinstance(sc.value, float):
                sc.number_format = "0.00"
        ws.cell(ws.max_row + 2, 1, "A laranja: p < α. Para preditores candidatos, uma diferença é esperada e desejada (não foram emparelhados).").font = Font(name=F, size=9, italic=True)
        for name in (f"{lab}_Selecao", f"{lab}_Suplentes"):
            ws = wb[name]
            _style(ws, widths={"A": 6, "B": 24, "C": 18, "D": 10, "E": 8, "F": 10, "G": 11})
            ws.auto_filter.ref = ws.dimensions
            hh = [c.value for c in ws[1]]
            for r in range(2, ws.max_row + 1):
                if "Data colheita" in hh:
                    ws.cell(r, hh.index("Data colheita") + 1).number_format = "dd-mm-yyyy"
                for k in ("Disponível no biobanco?", "Volume (µL)", "Medido miniNIR?", "Notas"):
                    if k in hh:
                        ws.cell(r, hh.index(k) + 1).fill = FILL_IN
        _style(wb[f"{lab}_Config"], widths={"A": 120})
    wb.save(out_path)
    return {"amostras_unicas": len(P), "por_prioridade": P["Prioridade"].value_counts().to_dict()}


def main(argv=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    ap = argparse.ArgumentParser(prog="selpop.consolidar", description="Junta vários estudos num só Excel.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--rotulos", nargs="+", help="rótulos curtos (ex.: M1 M2 M3)")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args(argv)
    paths = [Path(c).resolve() for c in a.configs]
    labels = a.rotulos or [f"M{i + 1}" for i in range(len(paths))]
    Rs = run_many(paths)
    info = write_many(Rs, labels, Path(a.saida))
    print(f"Resultado: {a.saida}")
    print(summary(Rs, labels)[["Modelo", "Casos emparelhados", "Controlos principais", "Suplentes", "Confundidores com p < α"]].to_string(index=False))
    print(f"Amostras únicas a pedir: {info['amostras_unicas']} {info['por_prioridade']}")


if __name__ == "__main__":
    main()
