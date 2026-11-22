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
    ("cr_max_dia", "Creatinina máxima do dia (mg/dL; numérica)"),
    ("cr_min_dia", "Creatinina mínima do dia (mg/dL; numérica)"),
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
    ("tfge_dia", "TFGe CKD-EPI 2021 com a creatinina mínima do dia"),
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
    c, idc = dict(cfg.get("colunas", {}) or {}), cfg["dados"]["id"]
    if c.get("creatinina") and not c.get("creatinina_max"):        # uma só coluna serve de máximo e mínimo
        c["creatinina_max"] = c["creatinina_min"] = c["creatinina"]
    ident = [idc] + [c[k] for k in ("n_amostra", "data_colheita", "data_admissao", "idade", "sexo", "vmi", "vmi_inicio", "vmi_fim",
                                    "ecmo", "ecmo_inicio", "ecmo_fim", "creatinina_max", "creatinina_min", "sofa_renal")
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


def _formulas(cfg: dict, cols: list[str], n: int) -> dict:
    """Fórmulas Excel por coluna derivada (só as que têm todas as colunas de entrada na folha).
    Reproduzem o código de derive.py; os limiares comparam valores arredondados a 6 casas, como no código."""
    from openpyxl.utils import get_column_letter
    c = cfg.get("colunas", {}) or {}
    for base in ("creatinina",):
        if c.get(base) and not c.get(f"{base}_max"):
            c = {**c, f"{base}_max": c[base], f"{base}_min": c[base]}
    lra = cfg.get("lra", {}) or {}
    dadm, k = int(lra.get("dias_admissao", 2)), int(lra.get("janela_estavel_dias", 2))
    pos = {col: get_column_letter(i + 1) for i, col in enumerate(cols)}
    name = {"id": cfg["dados"]["id"], **{key: c.get(key) for key in ("n_amostra", "data_colheita", "data_admissao", "idade", "sexo", "vmi",
            "vmi_inicio", "vmi_fim", "ecmo", "ecmo_inicio", "ecmo_fim", "creatinina_max", "creatinina_min")}}
    fem = c.get("sexo_feminino", 1)
    fem = f'"{fem}"' if isinstance(fem, str) else fem

    def L(x):            # letra da coluna (lógica ou derivada) ou None
        col = name.get(x, x)
        return pos.get(col) if col else None

    def C(x):            # célula da linha {r}
        return f"{L(x)}{{r}}"

    def RG(x):           # intervalo absoluto da coluna
        return f"${L(x)}$2:${L(x)}${n + 1}"

    EPS = "0.000001"     # ~0,1 s: o critério em texto ("<="&data) arredonda a 15 dígitos; evita perder a própria linha
    F = {}
    need = lambda *xs: all(L(x) for x in xs)
    ID, DT = "id", "data_colheita"
    same = lambda: f"{RG(ID)},{C(ID)}"
    if need(DT, "data_admissao"):
        F["dias_desde_admissao"] = f'=IF(OR({C(DT)}="",{C("data_admissao")}=""),"",INT({C(DT)}-{C("data_admissao")}))'
    if need("n_amostra"):
        F["amostra"] = f'=IF({C("n_amostra")}<>"","Sim","Não")'
    for flag, a0, a1, out in (("vmi", "vmi_inicio", "vmi_fim", "em_vmi_dia"), ("ecmo", "ecmo_inicio", "ecmo_fim", "em_ecmo_dia")):
        if need(DT, a0, a1):
            fl = f"{C(flag)}=1," if L(flag) else ""
            F[out] = (f'=IF(AND({fl}{C(a0)}<>"",{C(DT)}<>""),IF(AND({C(DT)}>={C(a0)},'
                      f'{C(DT)}<=IF({C(a1)}="",DATE(2100,1,1),{C(a1)})),"Sim","Não"),"Não")')
    for raw, out in (("creatinina_max", "cr_max_dia"), ("creatinina_min", "cr_min_dia")):
        if need(raw) and name[raw] != out:
            F[out] = f'=IF(ISNUMBER({C(raw)}),{C(raw)},"")'
    HI, LO = "cr_max_dia", "cr_min_dia"
    if need(ID, DT, HI, LO):
        def win_min(lo_expr, hi_expr):
            crit = f'{same()},{RG(DT)},">="&({lo_expr}-{EPS}),{RG(DT)},"<="&({hi_expr}+{EPS})'
            return f'IF(COUNTIFS({crit},{RG(LO)},">=0")=0,"",MINIFS({RG(LO)},{crit}))'
        F["kdigo_cr_min_7d_antes"] = f'=IF(OR({C(HI)}="",{C(DT)}=""),"",{win_min(f"{C(DT)}-7", C(DT))})'
        F["kdigo_razao_7d"] = f'=IF(N({C("kdigo_cr_min_7d_antes")})=0,"",ROUND({C(HI)}/{C("kdigo_cr_min_7d_antes")},6))'
        F["kdigo_cr_min_48h_antes"] = f'=IF(OR({C(HI)}="",{C(DT)}=""),"",{win_min(f"{C(DT)}-2", C(DT))})'
        F["kdigo_subida_48h"] = f'=IF({C("kdigo_cr_min_48h_antes")}="","",ROUND({C(HI)}-{C("kdigo_cr_min_48h_antes")},6))'
        r7, d2 = C("kdigo_razao_7d"), C("kdigo_subida_48h")
        F["lra_dia"] = (f'=IF(AND({C(HI)}<>"",OR(AND({r7}<>"",N({r7})>=1.5),AND({d2}<>"",N({d2})>=0.3))),"Sim","Não")')
        F["lra_estadio_dia"] = f'=IF({C("lra_dia")}="Não",0,IF(OR(N({r7})>=3,{C(HI)}>=4),3,IF(N({r7})>=2,2,1)))'
        F["kdigo_adm_nadir_7d_seguintes"] = (f'=IF(OR({C(HI)}="",{C(DT)}=""),"",IF({C(DT)}>MINIFS({RG(DT)},{same()})+{dadm - 1},"",'
                                             f'{win_min(C(DT), f"{C(DT)}+7")}))')
        F["kdigo_adm_razao"] = (f'=IF(N({C("kdigo_adm_nadir_7d_seguintes")})=0,"",'
                                f'ROUND({C(HI)}/{C("kdigo_adm_nadir_7d_seguintes")},6))')
        F["lra_subida"] = f'=IF(COUNTIFS({same()},{RG("lra_dia")},"Sim")>0,"Sim","Não")'
        F["lra_admissao"] = f'=IF(COUNTIFS({same()},{RG("kdigo_adm_razao")},">=1.5")>0,"Sim","Não")'
        F["lra_qualquer"] = f'=IF(OR({C("lra_subida")}="Sim",{C("lra_admissao")}="Sim"),"Sim","Não")'
        F["lra_inicio"] = f'=IF({C("lra_subida")}="Não","",MINIFS({RG(DT)},{same()},{RG("lra_dia")},"Sim"))'
        if L("dias_desde_admissao"):
            F["lra_inicio_dias_admissao"] = (f'=IF({C("lra_inicio")}="","",MINIFS({RG("dias_desde_admissao")},{same()},'
                                             f'{RG(DT)},{C("lra_inicio")},{RG("lra_dia")},"Sim"))')
        F["lra_estadio_max"] = f'=MAXIFS({RG("lra_estadio_dia")},{same()})'
        F["dias_ate_lra"] = f'=IF(OR({C("lra_inicio")}="",{C(DT)}=""),"",{C("lra_inicio")}-{C(DT)})'
        F["sem_lra_janela"] = (f'=IF({C(DT)}="","Sim",IF(COUNTIFS({same()},{RG(DT)},">="&({C(DT)}-{k}-{EPS}),{RG(DT)},"<="&({C(DT)}+{k}+{EPS}),'
                               f'{RG("lra_dia")},"Sim")=0,"Sim","Não"))')
        F["dias_obs_apos"] = (f'=IF(OR({C(DT)}="",COUNTIFS({same()},{RG(HI)},">=0")=0),"",'
                              f'MAXIFS({RG(DT)},{same()},{RG(HI)},">=0")-{C(DT)})')
    if need(ID, LO):
        F["cr_min_internamento"] = f'=IF(COUNTIFS({same()},{RG(LO)},">=0")=0,"",MINIFS({RG(LO)},{same()}))'
    if need("idade", "sexo", LO):
        def ckd(scr):
            f = f"({C('sexo')}={fem})"
            kk, aa = f"IF({f},0.7,0.9)", f"IF({f},-0.241,-0.302)"
            return (f'=IF(OR(N({scr})=0,{C("idade")}=""),"",142*MIN({scr}/{kk},1)^{aa}*MAX({scr}/{kk},1)^(-1.2)'
                    f'*0.9938^{C("idade")}*IF({f},1.012,1))')
        F["tfge_dia"] = ckd(C(LO))
        if L("cr_min_internamento"):
            F["tfge_basal_proxy"] = ckd(C("cr_min_internamento"))
    raw = {v for v in name.values() if v}
    # só colunas presentes, sem referências em falta e sem sobrepor uma coluna de dados com o mesmo nome
    return {col: f for col, f in F.items() if col in pos and col not in raw and "None" not in f}


def write_trace(xw, Rs, labels, sheet="Rastreabilidade", formulas=True):
    """Grava a folha (com filtros e painel fixo) e o dicionário das colunas. Com `formulas`, as colunas
    derivadas ficam como fórmulas Excel (recalculadas ao abrir) em vez de valores."""
    T, D = trace_frame(Rs, labels)
    T.to_excel(xw, sheet_name=sheet, index=False)
    ws = xw.sheets[sheet]
    F = _formulas(Rs[0]["cfg"], list(T.columns), len(T)) if formulas else {}
    if F:
        from openpyxl.utils import get_column_letter
        idx = {col: i + 1 for i, col in enumerate(T.columns)}
        dates = {"lra_inicio"}
        for col, f in F.items():
            j = idx[col]
            for r in range(2, len(T) + 2):
                cell = ws.cell(row=r, column=j)
                # MINIFS/MAXIFS são funções "novas" no formato .xlsx: gravam-se com o prefixo _xlfn. (sem ele: #NAME?)
                cell.value = f.replace("{r}", str(r)).replace("MINIFS(", "_xlfn.MINIFS(").replace("MAXIFS(", "_xlfn.MAXIFS(")
                if col in ("dias_desde_admissao", "dias_obs_apos", "dias_ate_lra", "lra_inicio_dias_admissao", "lra_estadio_dia", "lra_estadio_max"):
                    cell.number_format = "0"
                elif col in dates:
                    cell.number_format = "yyyy-mm-dd"
                elif col.startswith("kdigo_razao") or col.startswith("kdigo_adm_razao") or col.startswith("tfge"):
                    cell.number_format = "0.0000" if col.startswith("kdigo") else "0.0"
        D["Fórmula (linha 2)"] = D.Coluna.map(lambda col: F[col].replace("{r}", "2") if col in F else "")
        xw.book.calculation.fullCalcOnLoad = True
    derived = {x for x, _ in DERIVADAS}
    status = set(T.columns[-2 * len(Rs):])
    D["Tipo"] = D.Coluna.map(lambda col: "fórmula" if col in F else "destino na seleção (valor calculado pelo SelPop)" if col in status
                             else "derivada (valor)" if col in derived else "dados originais")
    D.to_excel(xw, sheet_name=f"{sheet}_colunas"[:31], index=False)
    from openpyxl.styles import Alignment, Font, PatternFill
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E7D32" if cell.value in F else "305496")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    from openpyxl.utils import get_column_letter
    for i, col in enumerate(T.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, len(str(col)) * 0.9), 45)
    wd = xw.sheets[f"{sheet}_colunas"[:31]]
    for L_, w in zip("ABCD", (32, 90, 120, 26)):
        wd.column_dimensions[L_].width = w
    return T
