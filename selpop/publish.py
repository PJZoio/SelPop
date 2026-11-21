"""Saídas para publicação e para validação externa.

- Tabela de características pronta para artigo (inglês por omissão; português opcional), em Word (.docx) e como DataFrame.
- Dados das amostras selecionadas em SPSS (.sav), com rótulos de variáveis e de valores
  (abre no SPSS, PSPP, jamovi e JASP), e sintaxe SPSS/PSPP para reproduzir os testes da tabela de comparação.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import comparar_categorica, comparar_continua, smd_cont, smd_prop

# ---------------------------------------------------------------- textos
SECTIONS = {
    "demografia": {"en": "Demographics", "pt": "Demografia"},
    "comorbilidades": {"en": "Comorbidities", "pt": "Comorbilidades"},
    "internamento": {"en": "ICU stay and severity at sampling", "pt": "Internamento e gravidade à data da amostra"},
    "preditores": {"en": "Candidate predictors (not used for matching)", "pt": "Preditores candidatos (não usados no emparelhamento)"},
    "laboratorio": {"en": "Laboratory (potential spectral interferents)", "pt": "Laboratório (potenciais interferentes espetrais)"},
    "desfechos": {"en": "Outcomes (after sampling; not tested)", "pt": "Desfechos (posteriores à amostra; não testados)"},
    "outras": {"en": "Other", "pt": "Outras"},
}
SECTION_ORDER = list(SECTIONS)

# Nomes em inglês e secção por omissão para as variáveis da base; a configuração pode sobrepor com
# `nome_en`, `secao` e `rotulos_en`.
DEFAULTS = {
    "IDADE": ("Age (years)", "demografia"), "SEXO": ("Sex", "demografia"), "IMC": ("BMI (kg/m²)", "demografia"),
    "COVID": ("COVID-19", "demografia"), "VAGA": ("Epidemic wave", "demografia"),
    "MOTIVO_ADMISSAO": ("Reason for ICU admission", "demografia"),
    "COMORBILIDADES": ("Any comorbidity", "comorbilidades"), "HTA": ("Hypertension", "comorbilidades"),
    "DIABETES": ("Diabetes mellitus", "comorbilidades"), "OBESIDADE": ("Obesity", "comorbilidades"),
    "DISLIPIDEMIA": ("Dyslipidaemia", "comorbilidades"), "CARDIOPATIA_ISQUEMICA": ("Ischaemic heart disease", "comorbilidades"),
    "ICC": ("Heart failure", "comorbilidades"), "DPCO_ASMA_ENFISEMA": ("COPD/asthma/emphysema", "comorbilidades"),
    "NEOPLASIA_SOLIDA": ("Solid neoplasm", "comorbilidades"), "DHC": ("Chronic liver disease", "comorbilidades"),
    "VMI": ("Invasive mechanical ventilation during stay", "internamento"),
    "em_vmi_dia": ("On invasive mechanical ventilation at sampling", "internamento"),
    "ECMO": ("ECMO during stay", "internamento"),
    "dias_desde_admissao": ("Days from ICU admission to sampling", "internamento"),
    "dias_vmi_ate_amostra": ("Days on mechanical ventilation at sampling", "internamento"),
    "sofa_nao_renal_nao_cv_dia": ("SOFA excluding renal and cardiovascular components", "internamento"),
    "sofa_nao_renal_dia": ("Non-renal SOFA", "internamento"), "sofa_total_dia": ("Total SOFA", "internamento"),
    "sofa_cv_dia": ("Cardiovascular SOFA", "preditores"),
    "troponina_dia": ("High-sensitivity troponin I (pg/mL)", "preditores"), "lactato_dia": ("Lactate (mmol/L)", "preditores"),
    "ldh_dia": ("Lactate dehydrogenase (U/L)", "laboratorio"), "bilirrubina_dia": ("Total bilirubin (mg/dL)", "laboratorio"),
    "ck_dia": ("Creatine kinase (U/L)", "laboratorio"), "tfge_dia": ("eGFR at sampling (mL/min/1.73 m²)", "laboratorio"),
    "duracao_uci_dias": ("ICU length of stay (days)", "desfechos"), "OBITO_UCI": ("ICU mortality", "desfechos"),
    "lra_estadio_max": ("Maximum KDIGO AKI stage", "desfechos"), "lra_qualquer": ("AKI at any time during stay", "desfechos"),
}
LABEL_EN = {"Sim": "Yes", "Não": "No", "Masculino": "Male", "Feminino": "Female"}
TESTS = {  # nome interno → (chave, texto en, texto pt)
    "Mann–Whitney U": ("a", "Mann–Whitney U test", "teste U de Mann–Whitney"),
    "t de Student": ("b", "Student's t-test", "teste t de Student"),
    "t de Welch": ("g", "Welch's t-test", "teste t de Welch"),
    "Qui-quadrado": ("c", "Pearson's χ² test", "teste do χ² de Pearson"),
    "Fisher exato": ("d", "Fisher's exact test", "teste exato de Fisher"),
    "Fisher (Monte Carlo)": ("e", "Fisher's exact test (Monte Carlo, 9,999 resamples)", "teste exato de Fisher (Monte Carlo, 9 999 reamostragens)"),
    "Qui-quadrado (permutação)": ("f", "χ² permutation test", "teste do χ² por permutação"),
}
TXT = {
    "en": {"var": "Variable", "p": "p", "smd": "SMD", "median": "median [IQR]", "n": "n (%)",
           "title": "Characteristics of {a} and matched {b}", "title_nm": "Characteristics of {a} and {b}",
           "data": "Data are median [interquartile range] or n (%). p-values are two-sided.",
           "match": "Controls were matched to cases ({ratio}:1) on {crit}.",
           "relax": " Cases without a control at this stage were matched with relaxed criteria: {crit}.",
           "pred": "Candidate predictors were not used for matching; differences between groups are expected.",
           "smdn": "SMD, standardised mean difference (log scale for skewed laboratory variables).",
           "miss": "Missing data (cases/controls): {lst}.", "nt": "—", "exact": "exact", "within": "±{w:g}"},
    "pt": {"var": "Variável", "p": "p", "smd": "SMD", "median": "mediana [IIQ]", "n": "n (%)",
           "title": "Características dos {a} e dos {b} emparelhados", "title_nm": "Características dos {a} e dos {b}",
           "data": "Dados em mediana [intervalo interquartil] ou n (%). Valores de p bilaterais.",
           "match": "Os controlos foram emparelhados aos casos ({ratio}:1) por {crit}.",
           "relax": " Os casos sem controlo nesta etapa foram emparelhados com critérios mais permissivos: {crit}.",
           "pred": "Os preditores candidatos não foram usados no emparelhamento; são esperadas diferenças entre grupos.",
           "smdn": "SMD, diferença média padronizada (escala log para variáveis laboratoriais assimétricas).",
           "miss": "Dados em falta (casos/controlos): {lst}.", "nt": "—", "exact": "exato", "within": "±{w:g}"},
}


def _vinfo(v: dict, lang: str):
    en, sec = DEFAULTS.get(v["col"], (v.get("nome", v["col"]), None))
    name = v.get("nome_en", en) if lang == "en" else v.get("nome", v["col"])
    if v.get("secao"):
        sec = v["secao"]
    if sec is None:
        sec = {"preditor": "preditores", "interferente": "laboratorio"}.get(v.get("papel"), "desfechos" if not v.get("testar", True) else "outras")
    labs = dict(v.get("rotulos") or {})
    if lang == "en":
        labs = {k: LABEL_EN.get(str(x), x) for k, x in labs.items()}
        labs.update(v.get("rotulos_en") or {})
    return name, sec, labs


def _dec(s: pd.Series, v: dict) -> int:
    if "decimais" in v:
        return int(v["decimais"])
    s = s.dropna()
    if not len(s):
        return 1
    if np.allclose(s, np.round(s)):
        return 0
    m = abs(float(s.median()))
    return 0 if m >= 100 else 1 if m >= 1 else 2


def _num(x, d, lang):
    t = f"{x:.{d}f}"
    return t.replace(".", ",") if lang == "pt" else t


def _p(p, lang):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    t = "<0.001" if p < 0.001 else f"{p:.3f}"
    return t.replace(".", ",") if lang == "pt" else t


def group_names(R, lang):
    g = R["cfg"]["grupos"]
    if lang == "en":
        return g["casos"].get("nome_en", R["names"][0]), g["controlos"].get("nome_en", R["names"][1])
    return R["names"]


def build_table(R: dict, lang: str = "en", smd: bool = False) -> tuple[pd.DataFrame, dict]:
    """Devolve (linhas, meta). Cada linha: tipo ('secao'|'var'|'nivel'), Variável, casos, controlos, p, chave do teste, SMD."""
    T = TXT[lang]
    a, b = R["sel_cases"], R["sel_ctrl"]
    rng = np.random.default_rng(int(R["meta"]["semente"]))
    by_sec = {s: [] for s in SECTION_ORDER}
    used_tests, missing = {}, []
    for v in R["cfg"]["equilibrio"]["variaveis"]:
        col = v["col"]
        if col not in a.columns:
            continue
        name, sec, labs = _vinfo(v, lang)
        test = v.get("testar", True)
        xa, xb = a[col], b[col]
        na, nb = int(xa.isna().sum()), int(xb.isna().sum())
        if na or nb:
            missing.append(f"{name} {na}/{nb}")
        rows = by_sec.setdefault(sec, [])
        if v.get("tipo", "continua") == "continua":
            xa, xb = pd.to_numeric(xa, errors="coerce"), pd.to_numeric(xb, errors="coerce")
            d = _dec(pd.concat([xa, xb]), v)
            f = lambda s: "" if s.notna().sum() == 0 else f"{_num(s.median(), d, lang)} [{_num(s.quantile(.25), d, lang)}–{_num(s.quantile(.75), d, lang)}]"
            p, tn = comparar_continua(xa, xb, forced=v.get("teste")) if test else (None, None)
            if v.get("log"):
                sm = smd_cont(np.log(xa.where(xa > 0)), np.log(xb.where(xb > 0)))
            else:
                sm = smd_cont(xa, xb)
            k = TESTS.get(tn, (None,))[0] if tn else None
            if k:
                used_tests[k] = tn
            rows.append({"tipo": "var", "Variável": f"{name}, {T['median']}", "casos": f(xa), "controlos": f(xb),
                         "p": _p(p, lang) if test else T["nt"], "teste": k, "SMD": sm})
        else:
            xa_, xb_ = xa.dropna(), xb.dropna()
            levels = sorted(set(xa_.unique()) | set(xb_.unique()), key=lambda z: (str(type(z)), z))
            pos = next((k for k in labs if (k is True or k == 1) and str(labs[k]) in ("Yes", "Sim")), None)
            if pos is not None and len(levels) == 1 and levels[0] != pos:   # ex.: ninguém fez ECMO → mostrar "Sim: 0 (0%)"
                levels = levels + [pos]
            tab = np.array([[int((xa_ == l).sum()) for l in levels], [int((xb_ == l).sum()) for l in levels]])
            p, tn = comparar_categorica(tab, rng, forced=v.get("teste")) if test and len(levels) > 1 else (None, None)
            k = TESTS.get(tn, (None,))[0] if tn else None
            if k:
                used_tests[k] = tn
            tot = tab.sum(1).clip(min=1)
            cell = lambda i, j: f"{tab[i, j]} ({_num(100 * tab[i, j] / tot[i], 1, lang)})"
            smds = [smd_prop(tab[0, j] / tot[0], tab[1, j] / tot[1]) for j in range(len(levels))]
            sm = (max(smds, key=abs) if len(levels) > 2 else smds[-1]) if smds else np.nan
            ptxt = _p(p, lang) if test else T["nt"]
            lab = lambda l: str(labs.get(l, labs.get(str(l), l)))
            if len(levels) <= 2:
                j = len(levels) - 1
                for jj, l in enumerate(levels):   # mostrar 'Sim'/'Feminino' quando existir (convenção habitual)
                    if lab(l) in ("Yes", "Sim", "Female", "Feminino"):
                        j = jj
                lv = lab(levels[j]) if levels else ""
                suffix = "" if lv in ("Yes", "Sim", "1", "True", "true") else f" — {lv}"
                rows.append({"tipo": "var", "Variável": f"{name}{suffix}, {T['n']}", "casos": cell(0, j) if levels else "",
                             "controlos": cell(1, j) if levels else "", "p": ptxt, "teste": k, "SMD": sm})
            else:
                rows.append({"tipo": "var", "Variável": f"{name}, {T['n']}", "casos": "", "controlos": "", "p": ptxt, "teste": k, "SMD": sm})
                for j, l in enumerate(levels):
                    rows.append({"tipo": "nivel", "Variável": lab(l), "casos": cell(0, j), "controlos": cell(1, j), "p": "", "teste": None, "SMD": None})
    out = []
    for s in [s for s in SECTION_ORDER if by_sec.get(s)] + [s for s in by_sec if s not in SECTION_ORDER and by_sec[s]]:
        out.append({"tipo": "secao", "Variável": SECTIONS.get(s, {}).get(lang, s), "casos": "", "controlos": "", "p": "", "teste": None, "SMD": None})
        out.extend(by_sec[s])
    df = pd.DataFrame(out)
    an, bn = group_names(R, lang)
    title = (T["title"] if R["cfg"].get("emparelhamento") else T["title_nm"]).format(a=an, b=bn)
    est = R["cfg"].get("estudo", {}).get("estrato")
    if est:
        title += f" — {est['rotulo_en'] if lang == 'en' else est['rotulo']}"
    meta = {"title": title,
            "col_a": f"{an[:1].upper() + an[1:]} (n = {len(a)})", "col_b": f"{bn[:1].upper() + bn[1:]} (n = {len(b)})", "notes": _notes(R, lang, used_tests, missing, smd,
                                                                                                any(r["tipo"] == "secao" and r["Variável"] == SECTIONS["preditores"][lang] for r in out))}
    if not smd:
        df = df.drop(columns="SMD")
    return df, meta


def _lower_first(t: str) -> str:
    """'Days from ICU…' → 'days from ICU…'; mantém siglas ('COVID-19', 'BMI')."""
    return t[0].lower() + t[1:] if len(t) > 1 and t[1].islower() else t


def _crit_text(st, lang, names):
    T = TXT[lang]
    parts = []
    for c in st.get("exato") or []:
        parts.append(_lower_first(names.get(c, c).split(" (")[0]))
    for c, w in (st.get("calibre") or {}).items():
        unit = re.search(r"\(([^)]*)\)\s*$", names.get(c, ""))
        u = f" {unit.group(1)}" if unit and unit.group(1) in ("years", "anos") else (" days" if "days" in names.get(c, "").lower() and lang == "en" else (" dias" if "dias" in names.get(c, "").lower() else ""))
        parts.append(f"{_lower_first(names.get(c, c).split(' (')[0])} ({T['within'].format(w=w)}{u})")
    return ", ".join(parts)


def _notes(R, lang, used_tests, missing, smd, has_pred):
    T = TXT[lang]
    notes = [T["data"]]
    spec = R.get("spec_used")
    names = {v["col"]: _vinfo(v, lang)[0] for v in R["cfg"]["equilibrio"]["variaveis"]}
    for c, (en, _) in DEFAULTS.items():
        names.setdefault(c, en if lang == "en" else c)
    if spec and spec.get("etapas"):
        st = spec["etapas"]
        txt = T["match"].format(ratio=spec.get("razao", 1), crit=_crit_text(st[0], lang, names))
        if len(st) > 1:
            txt += T["relax"].format(crit="; ".join(_crit_text(s, lang, names) for s in st[1:]))
        notes.append(txt)
    if used_tests:
        idx = 1 if lang == "en" else 2
        notes.append("; ".join(f"{k} {TESTS[n][idx]}" for k, n in sorted(used_tests.items())) + ".")
    if has_pred:
        notes.append(T["pred"])
    if smd:
        notes.append(T["smdn"])
    if missing:
        notes.append(T["miss"].format(lst="; ".join(missing)))
    return notes


# ---------------------------------------------------------------- Word
def write_docx(R: dict, path: str | Path, lang: str = "en", smd: bool = False, font: str = "Times New Roman", size: int = 9):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    df, meta = build_table(R, lang, smd)
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = font, Pt(size)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), font)

    t = doc.add_paragraph()
    r = t.add_run(meta["title"])
    r.bold, r.font.size = True, Pt(size + 1)

    cols = [TXT[lang]["var"], meta["col_a"], meta["col_b"], "p"] + (["SMD"] if smd else [])
    tbl = doc.add_table(rows=1, cols=len(cols))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = ([Cm(7.4), Cm(3.6), Cm(3.6), Cm(1.6), Cm(1.3)] if smd else [Cm(8.6), Cm(3.6), Cm(3.6), Cm(1.6)])

    def border(cell, **kw):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tcPr.append(b)
        for edge, val in kw.items():
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(val))
            e.set(qn("w:color"), "000000")
            b.append(e)

    def setcell(cell, text, bold=False, italic=False, align=None, indent=False, sup=None):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(1)
        if indent:
            p.paragraph_format.left_indent = Cm(0.4)
        if align:
            p.alignment = align
        run = p.add_run(text)
        run.bold, run.italic = bold, italic
        if sup:
            s = p.add_run(sup)
            s.font.superscript = True

    for i, c in enumerate(cols):
        cell = tbl.rows[0].cells[i]
        setcell(cell, c, bold=True, italic=(c == "p"), align=None if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
        border(cell, top=12, bottom=6)
    trPr = tbl.rows[0]._tr.get_or_add_trPr()   # repetir cabeçalho em cada página
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trPr.append(hdr)
    for _, row in df.iterrows():
        cells = tbl.add_row().cells
        if row["tipo"] == "secao":
            m = cells[0].merge(cells[-1])
            setcell(m, row["Variável"], bold=True)
            continue
        setcell(cells[0], row["Variável"], indent=row["tipo"] == "nivel" or True)
        if row["tipo"] == "nivel":
            cells[0].paragraphs[0].paragraph_format.left_indent = Cm(0.9)
        setcell(cells[1], row["casos"], align=WD_ALIGN_PARAGRAPH.CENTER)
        setcell(cells[2], row["controlos"], align=WD_ALIGN_PARAGRAPH.CENTER)
        k = row["teste"]
        setcell(cells[3], row["p"], align=WD_ALIGN_PARAGRAPH.CENTER, sup=k if isinstance(k, str) and k else None)
        if smd:
            v = row.get("SMD")
            setcell(cells[4], "" if v is None or pd.isna(v) else _num(v, 2, lang), align=WD_ALIGN_PARAGRAPH.CENTER)
    for c in tbl.rows[-1].cells:
        border(c, bottom=12)
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    for i, w in enumerate(widths):
        tbl.columns[i].width = w
        gc = tbl._tbl.tblGrid.findall(qn("w:gridCol"))
        if i < len(gc):
            gc[i].set(qn("w:w"), str(int(w.twips)))
    for row in tbl.rows:
        for i, w in enumerate(widths):
            if i < len(row.cells):
                row.cells[i].width = w
    for n in meta["notes"]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        rr = p.add_run(n)
        rr.font.size = Pt(size - 1)
    doc.save(path)
    return path


# ---------------------------------------------------------------- SPSS (.sav) + sintaxe
def _safe(name: str, used: set) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_") or "v"
    if not s[0].isalpha():
        s = "v_" + s
    s = re.sub(r"_+", "_", s)[:60]
    base, i = s, 2
    while s.lower() in used:
        s = f"{base[:56]}_{i}"
        i += 1
    used.add(s.lower())
    return s


def sav_frame(R: dict, lang: str = "en"):
    """Dados das amostras selecionadas (casos + controlos principais) prontos para .sav."""
    cfg, idc = R["cfg"], R["cfg"]["dados"]["id"]
    c = cfg["colunas"]
    vars_ = [v for v in cfg["equilibrio"]["variaveis"] if v["col"] in R["sel_cases"].columns]
    extra = [x for x in cfg.get("saida", {}).get("colunas", []) if x in R["sel_cases"].columns and x not in {v["col"] for v in vars_}]
    ns = c.get("n_amostra") if c.get("n_amostra") in R["sel_cases"].columns else None
    dc = c.get("data_colheita") if c.get("data_colheita") in R["sel_cases"].columns else None
    keep = list(dict.fromkeys([idc] + [x for x in (ns, dc) if x] + [v["col"] for v in vars_] + extra))
    a = R["sel_cases"][keep].copy()
    a["__grupo"], a["__conjunto"] = 1, a[idc]
    b = R["sel_ctrl"][keep + ["_caso"]].copy().rename(columns={"_caso": "__conjunto"})
    b["__grupo"] = 0
    d = pd.concat([a, b], ignore_index=True)
    used, ren, vlab, vval = set(), {}, {}, {}
    an, bn = group_names(R, lang)
    for col in ["__grupo", "__conjunto"] + keep:   # nomes reservados primeiro: nunca colidem com colunas da base
        ren[col] = _safe({idc: "id", ns: "n_amostra", dc: "data_colheita", "__grupo": "grupo", "__conjunto": "conjunto"}.get(col, col), used)
    vlab[ren["__grupo"]] = "Group (1 = case, 0 = control)" if lang == "en" else "Grupo (1 = caso, 0 = controlo)"
    vval[ren["__grupo"]] = {1: an, 0: bn}
    vlab[ren["__conjunto"]] = "Matched set (case ID)" if lang == "en" else "Conjunto emparelhado (ID do caso)"
    vlab[ren[idc]] = "Participant ID" if lang == "en" else "ID do participante"
    if ns:
        vlab[ren[ns]] = "Sample number" if lang == "en" else "Nº da amostra"
    if dc:
        vlab[ren[dc]] = "Sampling date" if lang == "en" else "Data da colheita"
    for v in vars_:
        name, _, labs = _vinfo(v, lang)
        vlab[ren[v["col"]]] = name
        if labs:
            vv = {}
            for k, t in labs.items():
                kk = int(k) if isinstance(k, (bool, np.bool_)) else k
                if isinstance(kk, (int, float)):
                    vv[kk] = str(t)
            if vv:
                vval[ren[v["col"]]] = vv
    for col in extra:
        vlab[ren[col]] = DEFAULTS.get(col, (col,))[0] if lang == "en" else col
    d = d[["__grupo", "__conjunto"] + keep].rename(columns=ren)
    for col in d.columns:
        if d[col].dtype == bool or str(d[col].dtype) == "boolean":
            d[col] = d[col].astype(float)
        elif d[col].dtype == object:
            num = pd.to_numeric(d[col], errors="coerce")
            if num.notna().sum() == d[col].notna().sum():
                d[col] = num
            else:
                d[col] = d[col].astype(str).where(d[col].notna(), "")
    if dc:
        d[ren[dc]] = pd.to_datetime(d[ren[dc]])
    tests = {}
    for v in vars_:
        if not v.get("testar", True):
            continue
        tests[ren[v["col"]]] = v.get("tipo", "continua")
    return d, vlab, vval, tests


def write_sav(R: dict, path: str | Path, lang: str = "en"):
    import pyreadstat
    d, vlab, vval, _ = sav_frame(R, lang)
    pyreadstat.write_sav(d, str(path), column_labels=[vlab.get(c, c) for c in d.columns],
                         variable_value_labels=vval, variable_measure={c: ("nominal" if c in vval or c == "grupo" or d[c].dtype == object else "scale") for c in d.columns})
    return path


def write_sps(R: dict, path: str | Path, sav_name: str, lang: str = "en"):
    """Sintaxe SPSS/PSPP para reproduzir os testes da tabela de comparação (compara com as linhas 'Pearson' e com variâncias iguais)."""
    d, vlab, vval, tests = sav_frame(R, lang)
    cont = [c for c, k in tests.items() if k == "continua"]
    cat = [c for c, k in tests.items() if k != "continua"]
    L = [f"* Validação da tabela de comparação gerada pelo SelPop ({R['meta']['estudo']}).",
         "* Comparar: contínuas → Mann-Whitney ou t (linha 'Equal variances assumed'; t de Welch → 'Equal variances not assumed');",
         "* categóricas → 'Pearson Chi-Square'",
         "* (sem correção de Yates) ou 'Fisher's Exact Test'. Fisher r×c: o SelPop usa Monte Carlo; pequenas diferenças são esperadas.",
         f"GET FILE='{sav_name}'.", ""]
    if cont:
        L += ["EXAMINE VARIABLES=" + " ".join(cont) + " BY grupo", "  /PLOT=NPPLOT", "  /STATISTICS=DESCRIPTIVES", "  /MISSING=PAIRWISE.", ""]
        L += ["NPAR TESTS", "  /M-W=" + " ".join(cont) + " BY grupo(0 1).", ""]
        L += ["T-TEST GROUPS=grupo(0 1)", "  /VARIABLES=" + " ".join(cont) + ".", ""]
    if cat:
        L += ["CROSSTABS", "  /TABLES=" + " ".join(cat) + " BY grupo", "  /STATISTICS=CHISQ", "  /CELLS=COUNT COLUMN.", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return path


def write_pub_sheet(wb, sheet_name: str, R: dict, lang: str = "en", smd: bool = True):
    """Acrescenta a tabela de publicação a um workbook openpyxl aberto."""
    from openpyxl.styles import Alignment, Border, Font, Side
    df, meta = build_table(R, lang, smd)
    ws = wb.create_sheet(sheet_name)
    def fnt(**k):
        k.setdefault("size", 10)
        return Font(name="Times New Roman", **k)
    ws["A1"] = meta["title"]
    ws["A1"].font = fnt(bold=True, size=11)
    cols = [TXT[lang]["var"], meta["col_a"], meta["col_b"], "p", "Test" if lang == "en" else "Teste"] + (["SMD"] if smd else [])
    thick, thin = Side(style="medium"), Side(style="thin")
    for j, c in enumerate(cols, 1):
        cell = ws.cell(3, j, c)
        cell.font = fnt(bold=True, italic=(c == "p"))
        cell.border = Border(top=thick, bottom=thin)
        cell.alignment = Alignment(horizontal="left" if j == 1 else "center", wrap_text=True)
    r = 4
    for _, row in df.iterrows():
        if row["tipo"] == "secao":
            ws.cell(r, 1, row["Variável"]).font = fnt(bold=True)
        else:
            ws.cell(r, 1, row["Variável"]).font = fnt()
            ws.cell(r, 1).alignment = Alignment(indent=2 if row["tipo"] == "nivel" else 1)
            vals = [row["casos"], row["controlos"], row["p"], row["teste"] if isinstance(row["teste"], str) else ""]
            if smd:
                v = row.get("SMD")
                vals.append("" if v is None or pd.isna(v) else round(float(v), 2))
            for j, v in enumerate(vals, 2):
                c = ws.cell(r, j, v)
                c.font = fnt()
                c.alignment = Alignment(horizontal="center")
        r += 1
    for j in range(1, len(cols) + 1):
        ws.cell(r - 1, j).border = Border(bottom=thick)
    r += 1
    for n in meta["notes"]:
        ws.cell(r, 1, n).font = fnt(size=9)
        ws.cell(r, 1).alignment = Alignment(wrap_text=False)
        r += 1
    for col, w in zip("ABCDEF", [62, 22, 22, 9, 6, 8]):
        ws.column_dimensions[col].width = w
    ws.sheet_view.showGridLines = False
    return ws
