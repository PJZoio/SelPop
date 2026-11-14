"""Exportação para Excel: seleção, Tabela 1, equilíbrio antes/depois, funil, iterações e configuração."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

F = "Arial"
HDR = PatternFill("solid", fgColor="1F3864")
TH = Side(style="thin", color="BFBFBF")
BRD = Border(left=TH, right=TH, top=TH, bottom=TH)
FLAG = PatternFill("solid", fgColor="F8CBAD")
FILL_IN = PatternFill("solid", fgColor="FFF2CC")

METODOS = [
    ("Objetivo", "Selecionar, de forma reprodutível, a população-alvo (casos) e os controlos a partir da base, com o dia/amostra-índice de cada doente, e documentar o equilíbrio entre grupos."),
    ("Pipeline", "1) carregar base e ficheiros auxiliares; 2) derivar variáveis por dia e por doente; 3) aplicar critérios de doente e de dia a cada grupo; 4) escolher o dia-índice dos casos; 5) emparelhar controlos por etapas; 6) Tabela 1 com testes e SMD; 7) opcionalmente reequilibrar acrescentando ao emparelhamento a variável com menor p; 8) exportar."),
    ("Emparelhamento", "Guloso global sem reposição: todos os pares caso–(doente-dia controlo) admissíveis (igualdade nas variáveis 'exato', diferença ≤ calibre nas restantes) ordenados por custo = Σ|diferença|/calibre, com desempate aleatório reprodutível (semente). Cada doente-controlo é usado uma vez. Casos que não completam a razão passam à etapa seguinte, mais permissiva. Os primeiros controlos de cada caso (até à razão) são 'principais'; os seguintes são 'suplentes'."),
    ("Testes (como nos artigos do grupo)", "Contínuas: mediana (P25–P75); Shapiro–Wilk em cada grupo → t de Student se ambos normais, senão Mann–Whitney U. Categóricas: n (%); qui-quadrado, ou exato de Fisher se alguma frequência esperada < 5. Bilateral; α definido na configuração. Sem correção para comparações múltiplas (Tabela 1 descritiva)."),
    ("SMD", "Diferença média padronizada (para categóricas com >2 níveis, o maior |SMD| entre níveis). Não depende de n: com grupos pequenos, 'p ≥ α' não demonstra equilíbrio, e o SMD ajuda a ver diferenças relevantes que o teste não deteta. Referência habitual: |SMD| < 0,1."),
    ("TFGe", "CKD-EPI 2021 sem coeficiente de raça. Em UCI a creatinina não está em estado estacionário e a perda muscular sobrestima a TFGe: usar como descritiva. 'tfge_basal_proxy' usa a creatinina mínima do internamento."),
    ("LRA (KDIGO, critério de creatinina)", "Subida: creatinina máx. do dia ≥1,5× o mínimo dos 7 dias anteriores (incl. o dia) ou ≥0,3 mg/dL acima do mínimo das 48 h anteriores. Presente na admissão: nos primeiros dias, creatinina ≥1,5× o nadir dos 7 dias seguintes. Estádio 1/2/3 pelo rácio face ao basal (1,5–1,9 / 2,0–2,9 / ≥3,0 ou ≥4,0 mg/dL). Aproximação: não há débito urinário nem creatinina pré-admissão; comparações entre valores diários."),
    ("Limitações", "Seleção condicionada aos dados registados; variáveis não registadas (ex.: medicação, débito urinário) não são equilibradas. Reequilibrar por p-valores aumenta o número de testes; a decisão final sobre as variáveis de emparelhamento deve ser científica, não só estatística."),
]


def _style(ws, header_row=1, widths=None, wrap=()):
    for c in ws[header_row]:
        c.font = Font(name=F, bold=True, color="FFFFFF", size=10)
        c.fill = HDR
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        c.border = BRD
    for row in ws.iter_rows(min_row=header_row + 1):
        for c in row:
            c.font = Font(name=F, size=10)
            c.border = BRD
            c.alignment = Alignment(vertical="top", wrap_text=c.column_letter in wrap)
    for col, w in (widths or {}).items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    ws.sheet_view.showGridLines = False


def _sel_rows(R, papel_filter):
    cfg, idc = R["cfg"], R["cfg"]["dados"]["id"]
    c = cfg["colunas"]
    names = R["names"]
    extra = [x for x in cfg.get("saida", {}).get("colunas", []) if x in R["df"].columns]
    bcols = [v["col"] for v in cfg["equilibrio"]["variaveis"] if v["col"] in R["df"].columns]
    cols = list(dict.fromkeys(extra + bcols))
    m = R["match"]
    rows = []
    if papel_filter == "principal":
        for _, r in R["sel_cases"].iterrows():
            rows.append({"Grupo": names[0], "Papel": "caso", "Conjunto (ID do caso)": r[idc], **_base(r, idc, c, cols)})
    sub = m[m.papel == papel_filter] if len(m) else m
    for _, mm in sub.iterrows():
        r = R["controls"].loc[mm._row]
        rows.append({"Grupo": names[1], "Papel": f"controlo {papel_filter}", "Conjunto (ID do caso)": mm.caso_id,
                     "Etapa": mm.etapa, "Critério": mm.criterio, "Custo": mm.custo, **_base(r, idc, c, cols)})
    if papel_filter == "principal" and not len(m) and "_caso" in R["sel_ctrl"]:
        for _, r in R["sel_ctrl"].iterrows():
            rows.append({"Grupo": names[1], "Papel": "controlo", "Conjunto (ID do caso)": None, **_base(r, idc, c, cols)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Conjunto (ID do caso)", "Papel"], key=lambda s: s.astype(str)).reset_index(drop=True)
        out.insert(0, "Seq", [f"{'S' if papel_filter == 'principal' else 'X'}{i + 1:03d}" for i in range(len(out))])
        for k in ("Disponível no biobanco?", "Volume (µL)", "Medido miniNIR?", "Notas"):
            out[k] = ""
    return out


def _base(r, idc, c, cols):
    d = {"ID": r[idc], "Nº amostra": r[c["n_amostra"]], "Data colheita": pd.to_datetime(r[c["data_colheita"]]).date()}
    for k in cols:
        v = r[k]
        d[k] = (v.item() if hasattr(v, "item") else v)
    return d


def write(R: dict, out_path: str | Path):
    out_path = Path(out_path)
    cfg, meta = R["cfg"], R["meta"]
    names = R["names"]
    idc = cfg["dados"]["id"]
    sel = _sel_rows(R, "principal")
    sup = _sel_rows(R, "suplente")
    m = R["match"]
    unmatched = R["cases"][~R["cases"][idc].isin(set(m[m.papel == "principal"].caso_id) if len(m) else set())] if cfg.get("emparelhamento") else R["cases"].iloc[0:0]

    tb, ta = R["t_before"].copy(), R["t_after"].copy()
    key = ["Variável", "Nível"]
    tb["_k"] = tb["Variável"].replace("", np.nan).ffill() + "|" + tb["Nível"]
    ta["_k"] = ta["Variável"].replace("", np.nan).ffill() + "|" + ta["Nível"]
    eq = ta[key + ["_k", "p", "SMD"]].merge(tb[["_k", "p", "SMD"]], on="_k", how="left", suffixes=(" depois", " antes"))
    eq = eq[key + ["p antes", "SMD antes", "p depois", "SMD depois"]]

    leia = [("Estudo", meta["estudo"]), ("Descrição", meta["descricao"]), ("Gerado em", meta["gerado_em"]),
            ("Versão selpop", meta["versao"]), ("Semente aleatória", meta["semente"]), ("α", meta["alfa"]),
            ("Casos", names[0]), ("Controlos", names[1])]
    for p in meta["proveniencia"]:
        leia.append(("Ficheiro de entrada", f"{p['ficheiro']} [folha {p['folha']}; {p['linhas']} linhas] SHA-256 {p['sha256'][:16]}… {p.get('nota', '')}"))
    leia += [("", "")] + METODOS

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(leia, columns=["Item", "Descrição"]).to_excel(xw, sheet_name="LEIA-ME", index=False)
        R["funil"].to_excel(xw, sheet_name="Funil", index=False)
        R["qc"].to_excel(xw, sheet_name="Qualidade_dados", index=False)
        ta.drop(columns="_k").to_excel(xw, sheet_name="Tabela1", index=False, startrow=2)
        eq.to_excel(xw, sheet_name="Equilibrio_antes_depois", index=False)
        sel.to_excel(xw, sheet_name="Selecao", index=False)
        (sup if len(sup) else pd.DataFrame({"Nota": ["Sem suplentes"]})).to_excel(xw, sheet_name="Suplentes", index=False)
        if len(unmatched):
            u = unmatched[[idc, cfg["colunas"]["n_amostra"], cfg["colunas"]["data_colheita"]] +
                          [v["col"] for v in cfg["equilibrio"]["variaveis"] if v["col"] in unmatched.columns][:8]]
            u.to_excel(xw, sheet_name="Casos_sem_controlo", index=False)
        if len(R["iters"]):
            R["iters"].to_excel(xw, sheet_name="Iteracoes", index=False)
        pd.DataFrame({"Configuração (YAML) usada nesta execução": yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).splitlines()}).to_excel(xw, sheet_name="Config", index=False)

    wb = load_workbook(out_path)
    _style(wb["LEIA-ME"], widths={"A": 30, "B": 120}, wrap=("A", "B"))
    _style(wb["Qualidade_dados"], widths={"A": 48, "B": 12, "C": 100}, wrap=("C",))
    _style(wb["Funil"], widths={"A": 26, "B": 90, "C": 10, "D": 14}, wrap=("B",))
    ws = wb["Tabela1"]
    ws["A1"] = f"Tabela 1 — {names[0]} vs {names[1]} (após emparelhamento). α = {meta['alfa']}"
    ws["A1"].font = Font(name=F, bold=True, size=12, color="1F3864")
    _style(ws, header_row=3, widths={"A": 34, "B": 20, "C": 18, "D": 18, "E": 16, "F": 9, "G": 8, "H": 12, "I": 12, "J": 12})
    ws.freeze_panes = "A4"
    hdr = {c.value: c.column for c in ws[3]}
    for row in ws.iter_rows(min_row=4):
        pcell = row[hdr["p"] - 1]
        if isinstance(pcell.value, float):
            pcell.number_format = "0.000"
            if pcell.value < meta["alfa"]:
                for c in row:
                    c.fill = FLAG
        smd = row[hdr["SMD"] - 1]
        if isinstance(smd.value, float):
            smd.number_format = "0.00"
    ws.cell(row=ws.max_row + 2, column=1, value="Linhas a laranja: p < α. Para variáveis binárias mostra-se o nível de referência (ex.: 'Sim'/'Feminino'); o p e o SMD referem-se à variável inteira.").font = Font(name=F, size=9, italic=True)
    _style(wb["Equilibrio_antes_depois"], widths={"A": 34, "B": 20, "C": 10, "D": 10, "E": 10, "F": 10})
    for row in wb["Equilibrio_antes_depois"].iter_rows(min_row=2):
        for c in row[2:]:
            if isinstance(c.value, float):
                c.number_format = "0.000" if "p" in str(wb["Equilibrio_antes_depois"].cell(1, c.column).value) else "0.00"
    for name in ("Selecao", "Suplentes"):
        ws = wb[name]
        _style(ws, widths={"A": 6, "B": 22, "C": 18, "D": 10, "E": 8, "F": 10, "G": 11})
        ws.auto_filter.ref = ws.dimensions
        hdrs = [c.value for c in ws[1]]
        for k in ("Disponível no biobanco?", "Volume (µL)", "Medido miniNIR?", "Notas"):
            if k in hdrs:
                j = hdrs.index(k) + 1
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, j).fill = FILL_IN
        for j, h in enumerate(hdrs, 1):
            if h == "Data colheita":
                for r in range(2, ws.max_row + 1):
                    ws.cell(r, j).number_format = "dd-mm-yyyy"
    for name in ("Casos_sem_controlo", "Iteracoes"):
        if name in wb.sheetnames:
            _style(wb[name], widths={"A": 12, "B": 14, "C": 16, "D": 40, "E": 12, "F": 70, "G": 10}, wrap=("D", "F"))
    _style(wb["Config"], widths={"A": 120})
    wb.save(out_path)
    return {"selecao": len(sel), "suplentes": len(sup), "casos_sem_controlo": len(unmatched)}
