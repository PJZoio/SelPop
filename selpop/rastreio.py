"""Folha de rastreabilidade: todas as linhas da base (doente-dia/amostra), com as colunas usadas nos
cálculos e filtros (KDIGO passo a passo, janelas temporais) e, por estudo, o destino de cada linha:
o primeiro critério em que falhou, ou o papel que teve na seleção."""
from __future__ import annotations

import pandas as pd

from .cohort import _criteria, q

# (coluna, descrição) — só entram as que existem na base
DERIVADAS = [
    ("dias_desde_admissao", "Dias desde a admissão na UCI (0 = dia da admissão)"),
    ("amostra", "Há amostra no biobanco neste dia (nº de amostra preenchido)"),
    ("em_ecmo_dia", "Em ECMO neste dia"),
    ("em_vmi_dia", "Em VMI neste dia"),
    ("cr_min_dia", "Creatinina mínima do dia (mg/dL)"),
    ("kdigo_cr_min_7d_antes", "KDIGO: mínimo da creatinina nos 7 dias até este dia (inclusive)"),
    ("kdigo_razao_7d", "KDIGO: creatinina máxima do dia ÷ mínimo dos 7 dias (critério se ≥ 1,5)"),
    ("kdigo_cr_min_48h_antes", "KDIGO: mínimo da creatinina nas 48 h até este dia"),
    ("kdigo_subida_48h", "KDIGO: creatinina máxima do dia − mínimo das 48 h, mg/dL (critério se ≥ 0,3)"),
    ("lra_dia", "Critério KDIGO de creatinina cumprido neste dia (razão ≥ 1,5 ou subida ≥ 0,3)"),
    ("lra_estadio_dia", "Estádio KDIGO neste dia (1: ≥1,5×; 2: ≥2×; 3: ≥3× ou ≥4,0 mg/dL)"),
    ("kdigo_adm_nadir_7d_seguintes", "LRA na admissão: nadir da creatinina neste dia e nos 7 seguintes (só nos primeiros dias)"),
    ("kdigo_adm_razao", "LRA na admissão: creatinina máxima do dia ÷ nadir dos 7 dias seguintes (critério se ≥ 1,5)"),
    ("lra_subida", "Doente: algum dia com critério KDIGO de subida no internamento"),
    ("lra_admissao", "Doente: LRA já presente na admissão (kdigo_adm_razao ≥ 1,5 num dos primeiros dias)"),
    ("lra_qualquer", "Doente: lra_subida ou lra_admissao"),
    ("lra_inicio", "Doente: data do primeiro dia com critério KDIGO (início da LRA)"),
    ("lra_inicio_dias_admissao", "Doente: dia de internamento em que a LRA começou"),
    ("lra_estadio_max", "Doente: estádio KDIGO máximo"),
    ("dias_ate_lra", "Dias deste dia até ao início da LRA (positivo = antes; 0 = dia do início; negativo = depois)"),
    ("sem_lra_janela", "Nenhum dia com critério KDIGO na janela de ± dias (dia estável)"),
    ("dias_obs_apos", "Dias com creatinina registada depois deste dia (seguimento)"),
    ("cr_min_internamento", "Doente: creatinina mínima do internamento (mg/dL)"),
    ("tfge_basal_proxy", "Doente: TFGe CKD-EPI 2021 com a creatinina mínima do internamento (proxy basal)"),
    ("tfge_dia", "TFGe CKD-EPI 2021 do dia (creatinina com valor único)"),
]


def _status(R, df, grupo: str) -> pd.Series:
    cfg, idc = R["cfg"], R["cfg"]["dados"]["id"]
    gdef = cfg["grupos"][grupo]
    out = pd.Series("", index=df.index, dtype=object)
    alive = pd.Series(True, index=df.index)
    for nivel in ("doente", "dia"):
        for lab, expr in _criteria(gdef.get(nivel)):
            ok = q(df, expr).reindex(df.index).fillna(False).astype(bool)
            out[alive & ~ok] = f"excluído: [{nivel}] {lab}"
            alive &= ok
    if grupo == "casos":
        out[alive] = "elegível (outro dia foi o dia-índice)"
        row = (lambda d: d["_linha"]) if "_linha" in R["cases"].columns else (lambda d: d.index)
        out[out.index.isin(row(R["cases"]))] = "dia-índice, sem controlo"
        out[out.index.isin(row(R["sel_cases"]))] = "CASO SELECIONADO (dia-índice)"
    else:
        out[alive] = "elegível (não escolhido)"
        out[alive & ~out.index.isin(R["controls"].index)] = "removido (o doente é caso)"
        m = R["match"]
        for papel, txt in (("suplente", "SUPLENTE"), ("principal", "CONTROLO PRINCIPAL")):
            mm = m[m.papel == papel]
            out.loc[mm._row.values] = [f"{txt} (caso {c})" for c in mm.caso_id.values]
    return out


def trace_frame(Rs: list[dict], labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabela, dicionário de colunas). Usa a base preparada do primeiro estudo; os estudos com outra
    preparação são calculados sobre a sua própria base e alinhados pelo índice da linha."""
    R0 = Rs[0]
    df, cfg = R0["df"], R0["cfg"]
    c, idc = cfg.get("colunas", {}) or {}, cfg["dados"]["id"]
    ident = [idc] + [c[k] for k in ("n_amostra", "data_colheita", "creatinina_max", "creatinina_min", "sofa_renal")
                     if c.get(k) and c[k] in df.columns]
    extra = [x for x in ("DRC", "TRANSPLANTADO", "COVID") if x in df.columns and x not in ident]
    ident = list(dict.fromkeys(ident))
    der = [(col, d) for col, d in DERIVADAS if col in df.columns and col not in ident + extra]
    T = df[ident + extra + [col for col, _ in der]].copy()
    dic = [(col, "Coluna da base original") for col in ident + extra] + der
    for lab, R in zip(labels, Rs):
        base = R["df"] if not R["df"].index.equals(df.index) else df
        for g, nome in (("casos", "casos"), ("controlos", "controlos")):
            col = f"{lab} {nome}".strip()
            T[col] = _status(R, base, g).reindex(T.index).fillna("")
            dic.append((col, f"Destino da linha no estudo {R['cfg']['estudo'].get('nome', lab)} — grupo {R['names'][0 if g == 'casos' else 1]}: "
                             "primeiro critério que falhou, ou papel na seleção"))
    for col in T.columns:
        if T[col].dtype == bool:
            T[col] = T[col].map({True: "Sim", False: "Não"})
    return T, pd.DataFrame(dic, columns=["Coluna", "Significado"])


def write_trace(xw, Rs, labels, sheet="Rastreabilidade"):
    """Grava a folha (com filtros e painel fixo) e o dicionário das colunas."""
    T, D = trace_frame(Rs, labels)
    T.to_excel(xw, sheet_name=sheet, index=False)
    D.to_excel(xw, sheet_name=f"{sheet}_colunas"[:31], index=False)
    ws = xw.sheets[sheet]
    from openpyxl.styles import Alignment, Font, PatternFill
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    from openpyxl.utils import get_column_letter
    for i, col in enumerate(T.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, len(str(col)) * 0.9), 45)
    xw.sheets[f"{sheet}_colunas"[:31]].column_dimensions["A"].width = 32
    xw.sheets[f"{sheet}_colunas"[:31]].column_dimensions["B"].width = 110
    return T
