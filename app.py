"""SelPop — interface guiada para selecionar populações, emparelhar controlos e comparar grupos.

Correr localmente (recomendado para dados reais):  streamlit run app.py
Os ficheiros carregados ficam apenas na memória desta sessão; nada é gravado em disco
exceto o que o utilizador descarregar.
"""
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy import stats as sps

from selpop import __version__
from selpop.cohort import q
from selpop.io import load_sources
from selpop.pipeline import prep_key_of, prepare, select
from selpop.publish import TXT as TXT_PUB, build_table, write_docx, write_sav, write_sps
from selpop.report import write
from selpop.stats import TESTES_CATEGORICA, TESTES_CONTINUA, explorar, inferir_tipo

ROOT = Path(__file__).parent
TEMPLATES = {p.stem: p for p in sorted((ROOT / "estudos").glob("*.yaml"))}
EXAMPLE = ROOT / "exemplos" / "exemplo_ficticio.csv"
INDEX_RULES = ["primeiro", "ultimo", "minimo", "maximo", "alvo", "aleatorio"]
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"
BLANK = "__novo__"
ON_CLOUD = Path("/mount/src").exists() or bool(os.environ.get("STREAMLIT_SHARING_MODE")) or os.environ.get("SELPOP_CLOUD") == "1"

# Mapeamento de colunas: chave interna → (rótulo, ajuda). Só o ID é obrigatório.
MAP_BASIC = {
    "data_colheita": ("Data da colheita/registo", "Necessária para dados longitudinais (várias linhas por pessoa)."),
    "n_amostra": ("Nº da amostra", "Identificador da amostra no biobanco. Se não existir, usa-se o ID."),
    "idade": ("Idade", "Usada na TFGe (CKD-EPI 2021) se também houver sexo e creatinina."),
    "sexo": ("Sexo", ""),
    "creatinina": ("Creatinina (mg/dL)", "Uma medição por linha. Permite calcular a TFGe."),
}
MAP_ICU = {
    "data_admissao": "Data de admissão", "data_alta": "Data de alta", "creatinina_max": "Creatinina — máximo do dia",
    "creatinina_min": "Creatinina — mínimo do dia", "ureia_max": "Ureia — máximo do dia", "ureia_min": "Ureia — mínimo do dia",
    "vmi": "VMI (0/1)", "vmi_inicio": "Início da VMI", "vmi_fim": "Fim da VMI", "ecmo": "ECMO (0/1)",
    "ecmo_inicio": "Início da ECMO", "ecmo_fim": "Fim da ECMO", "sofa_total": "SOFA total", "sofa_renal": "SOFA renal",
    "sofa_cardiovascular": "SOFA cardiovascular",
}

st.set_page_config(page_title="SelPop — seleção e comparação de grupos", page_icon="🧪", layout="wide")


# ================================================================ conversões configuração <-> tabelas editáveis
def crit_to_df(spec) -> pd.DataFrame:
    if isinstance(spec, str):
        spec = [spec] if spec.strip() else []
    rows = [{"Ativo": True, "Nome": i.get("nome", i["expr"]), "Expressão": i["expr"]} if isinstance(i, dict)
            else {"Ativo": True, "Nome": i, "Expressão": i} for i in spec or []]
    return pd.DataFrame(rows, columns=["Ativo", "Nome", "Expressão"])


def df_to_crit(d: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in d.iterrows():
        e = str(r.get("Expressão") or "").strip()
        if bool(r.get("Ativo", True)) and e and e != "nan":
            out.append({"nome": str(r.get("Nome") or e), "expr": e})
    return out


def stages_to_df(stages) -> pd.DataFrame:
    return pd.DataFrame([{"Exato": ", ".join(s.get("exato") or []),
                          "Calibres": "; ".join(f"{k}={v:g}" for k, v in (s.get("calibre") or {}).items())} for s in stages or []],
                        columns=["Exato", "Calibres"])


def df_to_stages(d: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in d.iterrows():
        ex = [x.strip() for x in str(r.get("Exato") or "").split(",") if x.strip() and x.strip() != "nan"]
        cal = {}
        for part in str(r.get("Calibres") or "").split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                cal[k.strip()] = float(v.replace(",", ".").strip())
        if ex or cal:
            out.append({"exato": ex, "calibre": cal})
    return out


def labels_to_txt(lab) -> str:
    return "; ".join(f"{k}={v}" for k, v in (lab or {}).items())


def txt_to_labels(t: str) -> dict | None:
    out = {}
    for part in str(t or "").split(";"):
        if "=" not in part:
            continue
        k, v = (x.strip() for x in part.split("=", 1))
        if k.lower() in ("true", "false"):
            key = k.lower() == "true"
        else:
            try:
                key = int(k)
            except ValueError:
                try:
                    key = float(k)
                except ValueError:
                    key = k
        out[key] = v
    return out or None


VAR_COLS = ["Coluna", "Nome", "Nome (EN)", "Tipo", "Papel", "Testar", "Log (SMD)", "Rótulos"]


def vars_to_df(vs) -> pd.DataFrame:
    return pd.DataFrame([{"Coluna": v["col"], "Nome": v.get("nome", v["col"]), "Nome (EN)": v.get("nome_en", ""),
                          "Tipo": v.get("tipo", "continua"), "Papel": v.get("papel", "confundidor"),
                          "Testar": bool(v.get("testar", True)), "Log (SMD)": bool(v.get("log", False)),
                          "Rótulos": labels_to_txt(v.get("rotulos"))} for v in vs], columns=VAR_COLS)


def df_to_vars(d: pd.DataFrame, base: list[dict]) -> list[dict]:
    """Junta o que está na tabela com os campos que a tabela não mostra (ex.: rotulos_en, secao, teste)."""
    keep = {v["col"]: v for v in base}
    out = []
    for _, r in d.iterrows():
        col = str(r.get("Coluna") or "").strip()
        if not col or col == "nan":
            continue
        v = {k: x for k, x in keep.get(col, {}).items() if k in ("rotulos_en", "secao", "decimais", "teste")}
        v.update({"col": col, "nome": str(r.get("Nome") or col), "tipo": r.get("Tipo") or "continua"})
        if str(r.get("Nome (EN)") or "").strip() not in ("", "nan", "None"):
            v["nome_en"] = str(r["Nome (EN)"]).strip()
        if not bool(r.get("Testar", True)):
            v["testar"] = False
        if r.get("Papel") and r.get("Papel") != "confundidor":
            v["papel"] = r.get("Papel")
        if bool(r.get("Log (SMD)", False)):
            v["log"] = True
        lab = txt_to_labels(r.get("Rótulos"))
        if lab:
            v["rotulos"] = lab
        out.append(v)
    return out


# ================================================================ dados e estado
@st.cache_data(show_spinner=False, max_entries=6)
def read_raw(name: str, data: bytes, sheet) -> pd.DataFrame:
    from selpop.io import _read
    return _read((name, data), sheet)[0]


@st.cache_data(show_spinner=False, max_entries=6)
def excel_sheets(data: bytes) -> list[str]:
    import io
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner="A preparar a base (variáveis derivadas)…", max_entries=4)
def cached_prepare(main_name: str, main_bytes: bytes, aux: tuple, key_yaml: str):
    cfg = yaml.safe_load(key_yaml)
    raw, prov = load_sources(cfg, (main_name, main_bytes), [None if a is None else (a[0], a[1]) for a in aux])
    df, qc = prepare(cfg, raw)
    return df, qc, prov


def blank_cfg(raw: pd.DataFrame, fname: str) -> dict:
    cols = [str(c) for c in raw.columns]
    guess = next((c for c in cols if c.lower() in ("id", "id_isel", "participante", "doente", "patient_id", "subject")), None) \
        or next((c for c in cols if "id" in c.lower()), cols[0])
    return {"estudo": {"nome": "Novo estudo", "descricao": "", "semente": 12345},
            "dados": {"ficheiro": fname, "folha": 0, "id": guess}, "colunas": {}, "lra": {"calcular": False},
            "grupos": {"casos": {"nome": "Casos", "nome_en": "cases", "doente": []},
                       "controlos": {"nome": "Controlos", "nome_en": "controls", "doente": []}},
            "emparelhamento": None, "equilibrio": {"alfa": 0.05, "variaveis": []}, "saida": {"colunas": []}}


def load_template(cfg: dict):
    ss.base_cfg = cfg
    ss.ver = ss.get("ver", 0) + 1
    for g in ("casos", "controlos"):
        for lvl in ("doente", "dia"):
            ss[f"init_{g}_{lvl}"] = crit_to_df(cfg["grupos"][g].get(lvl))
    ss.init_stages = stages_to_df((cfg.get("emparelhamento") or {}).get("etapas"))
    ss.init_vars = vars_to_df(cfg["equilibrio"]["variaveis"])
    ss.test_choice = {v["col"]: v.get("teste", "auto") for v in cfg["equilibrio"]["variaveis"]}
    for k in ("R", "R_cfg_yaml"):
        ss.pop(k, None)


def _tmp_bytes(fn, suffix: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"out{suffix}"
        fn(p)
        return p.read_bytes()


# ================================================================ gráficos
def love_plot(tb: pd.DataFrame, ta: pd.DataFrame, lim: float):
    def tidy(t, when):
        t = t[t["SMD"].notna()].copy()
        t["Variável"] = t["Variável"].replace("", np.nan).ffill()
        t = t[t["Teste"].ne("descritiva")]
        return pd.DataFrame({"Variável": t["Variável"], "|SMD|": t["SMD"].abs(), "SMD": t["SMD"], "p": t["p"], "Momento": when})
    d = pd.concat([tidy(tb, "Antes do emparelhamento"), tidy(ta, "Depois do emparelhamento")])
    order = d[d.Momento.str.startswith("Antes")].sort_values("|SMD|", ascending=False)["Variável"].tolist()
    base = alt.Chart(d).encode(
        y=alt.Y("Variável:N", sort=order, title=None, axis=alt.Axis(labelLimit=280, grid=False, labelOverlap=False, labelFontSize=12)),
        x=alt.X("|SMD|:Q", title="|Diferença média padronizada|", axis=alt.Axis(gridColor="#e6e5e1", tickCount=6)),
        color=alt.Color("Momento:N", scale=alt.Scale(domain=["Antes do emparelhamento", "Depois do emparelhamento"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=320, labelFontSize=12)),
        shape=alt.Shape("Momento:N", scale=alt.Scale(range=["circle", "diamond"]), legend=None),
        tooltip=["Variável", "Momento", alt.Tooltip("SMD:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"x": [lim]})).mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q")
    return (rule + base.mark_point(size=90, filled=True, opacity=0.95, stroke="white", strokeWidth=1.5)) \
        .properties(height=max(240, 30 * len(order))).configure_view(stroke=None)


def dist_plots(a: pd.Series, b: pd.Series, names: tuple[str, str], label: str):
    """Histograma e gráfico Q–Q normal, um painel por grupo (sem sobreposição)."""
    rows, qq = [], []
    for g, x in zip(names, (a, b)):
        x = pd.to_numeric(x, errors="coerce").dropna()
        rows.append(pd.DataFrame({"valor": x.values, "Grupo": g}))
        if len(x) >= 3 and x.std() > 0:
            z = (np.sort(x.values) - x.mean()) / x.std(ddof=1)
            th = sps.norm.ppf((np.arange(1, len(z) + 1) - 0.375) / (len(z) + 0.25))
            qq.append(pd.DataFrame({"Quantil teórico (normal)": th, "Quantil observado (padronizado)": z, "Grupo": g}))
    d = pd.concat(rows)
    col = alt.Color("Grupo:N", scale=alt.Scale(domain=list(names), range=[BLUE, ORANGE]), legend=None)
    hist = alt.Chart(d).mark_bar(opacity=0.9, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("valor:Q", bin=alt.Bin(maxbins=20), title=label), y=alt.Y("count():Q", title="N.º"), color=col,
        tooltip=[alt.Tooltip("count():Q", title="N.º")]).properties(width=280, height=170) \
        .facet(column=alt.Column("Grupo:N", title=None, sort=list(names)))
    charts = [hist]
    if qq:
        dq = pd.concat(qq)
        lim = float(np.nanmax(np.abs(dq[["Quantil teórico (normal)", "Quantil observado (padronizado)"]].values))) + 0.2
        line = alt.Chart(pd.DataFrame({"x": [-lim, lim], "y": [-lim, lim]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", title="Quantil teórico (normal)"), y=alt.Y("y:Q", title="Quantil observado (padronizado)"))
        pts = alt.Chart().mark_point(size=40, filled=True, opacity=0.8).encode(
            x=alt.X("Quantil teórico (normal):Q", scale=alt.Scale(domain=[-lim, lim]), title="Quantil teórico (normal)"),
            y=alt.Y("Quantil observado (padronizado):Q", scale=alt.Scale(domain=[-lim, lim]), title="Quantil observado (padronizado)"), color=col)
        charts.append(alt.layer(line, pts, data=dq).properties(width=280, height=200)
                      .facet(column=alt.Column("Grupo:N", title=None, sort=list(names))))
    return charts


# ================================================================ CABEÇALHO: privacidade e uso local
ss = st.session_state
st.title("SelPop — seleção de populações e comparação de grupos")
if ON_CLOUD:
    st.error("**Está a usar a versão na nuvem.** Os ficheiros que carregar são processados num servidor externo. "
             "Use apenas os **dados de exemplo** ou dados totalmente anonimizados. Para dados de doentes, use a "
             "aplicação **no seu computador** (instruções abaixo).", icon="☁️")
else:
    st.success("**A correr no seu computador.** Os dados carregados não saem desta máquina.", icon="🔒")
with st.expander("Como usar no seu computador (recomendado para dados reais)", expanded=ON_CLOUD):
    st.markdown(
        "1. Instalar o **Python 3.10 ou mais recente** (python.org; no Windows, marcar *Add Python to PATH*).\n"
        "2. Descarregar o código: `git clone https://github.com/PJZoio/SelPop.git` (ou *Code → Download ZIP* no GitHub).\n"
        "3. Na pasta do SelPop, instalar as dependências: `pip install -r requirements.txt`\n"
        "4. Abrir a aplicação: `streamlit run app.py` — abre no browser em `http://localhost:8501`.\n\n"
        "O endereço *localhost* significa que a aplicação corre neste computador: nada é enviado para a internet. "
        "Os dados não ficam guardados: ao fechar, só ficam os ficheiros que descarregar. "
        "Nunca coloque bases com dados de doentes no GitHub nem em serviços na nuvem sem autorização do encarregado de proteção de dados.")

# ================================================================ BARRA LATERAL: dados e modelo de estudo
with st.sidebar:
    st.header("Dados e estudo")
    st.caption(f"SelPop v{__version__}")
    src = st.radio("Dados", ["Exemplo (fictícios)", "Os meus dados"], index=0 if ON_CLOUD else 1,
                   help="O exemplo tem 400 participantes fictícios (casos e controlos) para experimentar sem riscos.")
    if src == "Os meus dados":
        up = st.file_uploader("Base (Excel ou CSV)", type=["xlsx", "xls", "csv", "tsv", "txt"])
        if up is None:
            main_name, main_bytes = None, None
        else:
            main_name, main_bytes = up.name, up.getvalue()
    else:
        main_name, main_bytes = EXAMPLE.name, EXAMPLE.read_bytes()
    sheet = 0
    if main_bytes is not None and main_name.lower().endswith((".xlsx", ".xls")):
        sheets = excel_sheets(main_bytes)
        sheet = st.selectbox("Folha", sheets, index=0) if len(sheets) > 1 else sheets[0]

    opts = [BLANK] + list(TEMPLATES)
    default_t = "exemplo_generico" if src.startswith("Exemplo") else BLANK
    tname = st.selectbox("Modelo de estudo", opts, index=opts.index(default_t) if default_t in opts else 0,
                         format_func=lambda s: "Novo estudo (em branco)" if s == BLANK else s.replace("_", " "),
                         help="Um modelo traz critérios, emparelhamento e variáveis já preenchidos, que pode alterar.")
    up_cfg = st.file_uploader("…ou carregar uma configuração guardada (YAML)", type=["yaml", "yml"])

if main_bytes is None:
    st.info("**Para começar:** escolha *Exemplo (fictícios)* para experimentar, ou carregue a sua base (Excel ou CSV) na barra lateral.")
    st.stop()

try:
    raw = read_raw(main_name, main_bytes, sheet)
except Exception as e:
    st.error(f"Não foi possível ler o ficheiro: {e}")
    st.stop()
raw_cols = [str(c) for c in raw.columns]

src_id = f"{main_name}:{len(main_bytes)}:{sheet}|" + (f"yaml:{up_cfg.name}:{up_cfg.size}" if up_cfg else f"tpl:{tname}")
if ss.get("cfg_src") != src_id:
    if up_cfg:
        cfg0 = yaml.safe_load(up_cfg.getvalue())
    elif tname == BLANK:
        cfg0 = blank_cfg(raw, main_name)
    else:
        cfg0 = yaml.safe_load(TEMPLATES[tname].read_text(encoding="utf-8"))
    cfg0.setdefault("dados", {})["folha"] = sheet
    load_template(cfg0)
    ss.cfg_src = src_id
base_cfg = ss.base_cfg
V = ss.ver

with st.sidebar:
    aux_ups = []
    if src == "Os meus dados":
        for i, a in enumerate(base_cfg["dados"].get("auxiliares") or []):
            aux_ups.append(st.file_uploader(f"Ficheiro auxiliar: {Path(a['ficheiro']).name} (opcional)",
                                            type=["xlsx", "xls", "csv"], key=f"aux{i}_{V}"))
    else:
        aux_ups = [None] * len(base_cfg["dados"].get("auxiliares") or [])

st.subheader(base_cfg["estudo"].get("nome", "Estudo"))
if base_cfg["estudo"].get("descricao"):
    st.caption(base_cfg["estudo"]["descricao"])

tabs = st.tabs(["1 · Colunas", "2 · Grupos", "3 · Emparelhamento", "4 · Variáveis", "5 · Escolha dos testes", "6 · Resultados", "Configuração"])

# ================================================================ 1. COLUNAS
with tabs[0]:
    st.markdown("Indique que coluna identifica cada participante e, se existirem, as colunas com significado especial. "
                "**Só o ID é obrigatório.** As restantes permitem calcular variáveis derivadas (ex.: TFGe, dias desde a admissão).")
    c0 = base_cfg.get("colunas") or {}
    none = "—"
    sel = lambda lab, cur, key, help="": st.selectbox(lab, [none] + raw_cols, index=([none] + raw_cols).index(cur) if cur in raw_cols else 0,
                                                       key=f"{key}_{V}", help=help or None)
    m1, m2, m3 = st.columns(3)
    id_cur = base_cfg["dados"].get("id")
    id_col = m1.selectbox("ID do participante (obrigatório)", raw_cols, index=raw_cols.index(id_cur) if id_cur in raw_cols else 0, key=f"id_{V}")
    mapping = {}
    grid = [m2, m3, m1, m2, m3]
    for i, (k, (lab, hp)) in enumerate(MAP_BASIC.items()):
        with grid[i % len(grid)]:
            v = sel(lab, c0.get(k), f"map_{k}", hp)
        if v != none:
            mapping[k] = v
    if mapping.get("sexo"):
        vals = sorted(map(str, raw[mapping["sexo"]].dropna().unique()))[:20]
        curf = str(c0.get("sexo_feminino", ""))
        fv = m1.selectbox("Valor que indica 'feminino'", vals, index=vals.index(curf) if curf in vals else 0, key=f"fem_{V}")
        orig = raw[mapping["sexo"]].dropna()
        mapping["sexo_feminino"] = next((x for x in orig.unique() if str(x) == fv), fv)
        if isinstance(mapping["sexo_feminino"], np.generic):
            mapping["sexo_feminino"] = mapping["sexo_feminino"].item()
    with st.expander("Variáveis clínicas de internamento (opcional — UCI)", expanded=any(c0.get(k) for k in MAP_ICU)):
        st.caption("Para bases longitudinais de UCI: permitem calcular LRA (KDIGO), dias de VMI/ECMO, SOFA não renal, etc.")
        cols3 = st.columns(3)
        for i, (k, lab) in enumerate(MAP_ICU.items()):
            with cols3[i % 3]:
                v = sel(lab, c0.get(k), f"map_{k}")
                if v != none:
                    mapping[k] = v
        calc_lra = st.checkbox("Calcular LRA (critério de creatinina KDIGO)", value=(base_cfg.get("lra") or {}).get("calcular", True)
                               and bool(mapping.get("creatinina_max") or mapping.get("creatinina")) and bool(mapping.get("data_colheita")),
                               key=f"lra_{V}", help="Requer data e creatinina diária (várias linhas por pessoa).")
    n_rows, n_ids = len(raw), raw[id_col].nunique()
    st.info(f"**{n_rows} linhas · {n_ids} participantes** — " + ("uma linha por participante (dados transversais)." if n_rows == n_ids
            else f"em média {n_rows / max(n_ids, 1):.1f} linhas por participante (dados longitudinais: o dia-índice escolhe uma linha por pessoa)."))
    with st.expander("Ver as primeiras linhas e os tipos das colunas"):
        st.dataframe(raw.head(8), width="stretch")
        tinfo = pd.DataFrame([{"Coluna": c, "Tipo sugerido": inferir_tipo(raw[c])[0], "Explicação": inferir_tipo(raw[c])[1],
                               "Valores em falta": int(raw[c].isna().sum())} for c in raw.columns])
        st.dataframe(tinfo, hide_index=True, width="stretch", height=300)

# configuração de preparação (depende do mapeamento)
prep_cfg = copy.deepcopy(base_cfg)
prep_cfg["dados"]["id"] = id_col
prep_cfg["dados"]["folha"] = sheet
prep_cfg["colunas"] = mapping
prep_cfg.setdefault("lra", {})
prep_cfg["lra"] = {**(prep_cfg.get("lra") or {}), "calcular": bool(calc_lra)}
aux_key = tuple(None if u is None else (u.name, u.getvalue()) for u in aux_ups)
try:
    df, qc, prov = cached_prepare(main_name, main_bytes, aux_key, prep_key_of(prep_cfg))
except Exception as e:
    st.error(f"Não foi possível preparar a base com este mapeamento de colunas: {e}")
    st.stop()
idc = id_col
cols_all = sorted(map(str, df.columns), key=str.lower)

# ================================================================ 2. GRUPOS
with tabs[1]:
    st.markdown("Defina quem são os **casos** (população-alvo) e os **controlos**. A forma mais simples: escolher a coluna que separa "
                "os grupos e os respetivos valores. Para critérios mais finos (ex.: idade ≥ 18 e sem transplante), use as tabelas de critérios.")
    with st.container(border=True):
        st.markdown("**Assistente rápido**")
        q1, q2, q3, q4 = st.columns([2, 2, 2, 1])
        gcol = q1.selectbox("Coluna que define os grupos", [none] + cols_all, key=f"gcol_{V}")
        if gcol != none:
            gvals = sorted(map(str, df[gcol].dropna().unique()))[:50]
            vcase = q2.selectbox("Valor dos casos", gvals, key=f"gcase_{V}")
            vctrl = q3.selectbox("Valor dos controlos", ["(todos os outros)"] + [g for g in gvals if g != vcase], key=f"gctrl_{V}")
            if q4.button("Aplicar", key=f"gapply_{V}", help="Substitui os critérios de doente dos dois grupos."):
                is_num = pd.api.types.is_numeric_dtype(df[gcol])
                lit = lambda x: x if is_num else repr(x)
                colref = gcol if gcol.isidentifier() else f"`{gcol}`"
                ss[f"init_casos_doente"] = crit_to_df([{"nome": f"{gcol} = {vcase}", "expr": f"{colref} == {lit(vcase)}"}])
                ss[f"init_controlos_doente"] = crit_to_df([{"nome": f"{gcol} ≠ {vcase}" if vctrl.startswith("(") else f"{gcol} = {vctrl}",
                                                             "expr": f"{colref} != {lit(vcase)}" if vctrl.startswith("(") else f"{colref} == {lit(vctrl)}"}])
                ss.crit_ver = ss.get("crit_ver", 0) + 1   # só as tabelas de critérios recomeçam
                st.rerun()
    st.markdown("Cada linha é um critério, aplicado por ordem (expressões pandas: `and`, `or`, `not`, `==`, `>=`; nomes com espaços ou "
                "acentos entre acentos graves). Desmarque **Ativo** para ver o efeito de retirar um critério.")
    edited = {}
    gcols = st.columns(2)
    for gi, g in enumerate(("casos", "controlos")):
        with gcols[gi]:
            edited[(g, "nome")] = st.text_input("Nome do grupo", base_cfg["grupos"][g].get("nome", g), key=f"name_{g}_{V}")
            edited[(g, "nome_en")] = st.text_input("Nome em inglês (para a tabela do artigo)", base_cfg["grupos"][g].get("nome_en", ""),
                                                   key=f"nameen_{g}_{V}")
            for lvl, lab in (("doente", "Critérios por participante"), ("dia", "Critérios por linha/amostra (dados longitudinais)")):
                st.markdown(f"**{lab}**")
                edited[(g, lvl)] = st.data_editor(ss[f"init_{g}_{lvl}"], key=f"ed_{g}_{lvl}_{V}_{ss.get('crit_ver', 0)}", num_rows="dynamic", width="stretch",
                                                  column_config={"Ativo": st.column_config.CheckboxColumn(width="small"),
                                                                 "Nome": st.column_config.TextColumn(width="medium"),
                                                                 "Expressão": st.column_config.TextColumn(width="large")})
            if g == "casos":
                ind = base_cfg["grupos"]["casos"].get("indice") or {"regra": "primeiro"}
                st.markdown("**Linha-índice dos casos** (uma por participante; só importa com várias linhas por pessoa)")
                i1, i2, i3 = st.columns(3)
                rule = i1.selectbox("Regra", INDEX_RULES, index=INDEX_RULES.index(ind.get("regra", "primeiro")), key=f"rule_{V}")
                rcol = i2.selectbox("Coluna", [""] + cols_all, index=([""] + cols_all).index(ind["coluna"]) if ind.get("coluna") in cols_all else 0,
                                    key=f"rcol_{V}", disabled=rule not in ("minimo", "maximo", "alvo"))
                ralvo = i3.number_input("Alvo", value=float(ind.get("alvo", 0) or 0), key=f"ralvo_{V}", disabled=rule != "alvo")
    st.divider()
    st.markdown("**Funil em tempo real** (antes do emparelhamento)")
    fcols = st.columns(2)
    crit_errors = []
    for gi, g in enumerate(("casos", "controlos")):
        rows, m = [{"Critério": "Base completa", "Participantes": df[idc].nunique(), "Linhas": len(df)}], pd.Series(True, index=df.index)
        for lvl in ("doente", "dia"):
            for c in df_to_crit(edited[(g, lvl)]):
                try:
                    m = m & q(df, c["expr"])
                    rows.append({"Critério": f"[{lvl}] {c['nome']}", "Participantes": df.loc[m, idc].nunique(), "Linhas": int(m.sum())})
                except Exception as e:
                    crit_errors.append(f"{edited[(g, 'nome')]} — «{c['expr']}»: {e}")
                    rows.append({"Critério": f"[{lvl}] {c['nome']} — ERRO", "Participantes": None, "Linhas": None})
        with fcols[gi]:
            st.caption(edited[(g, "nome")])
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for e in crit_errors:
        st.error(e)
    if not df_to_crit(edited[("casos", "doente")]) + df_to_crit(edited[("casos", "dia")]):
        st.warning("Os casos ainda não têm critérios: neste momento seriam todos os participantes. Use o assistente rápido acima.")

# ================================================================ 3. EMPARElHAMENTO
emp0 = base_cfg.get("emparelhamento") or {}
reeq0 = emp0.get("reequilibrar") or {}
with tabs[2]:
    st.markdown("O **emparelhamento** escolhe, para cada caso, controlos semelhantes nas variáveis indicadas (ex.: mesmo sexo e idade ±5 anos). "
                "Torna os grupos comparáveis, para que as diferenças encontradas não se devam a essas variáveis.")
    use_match = st.toggle("Emparelhar controlos aos casos", value=bool(emp0), key=f"usem_{V}")
    e1, e2, e3, e4 = st.columns(4)
    razao = e1.number_input("Controlos por caso", 1, 10, int(emp0.get("razao", 2)), key=f"razao_{V}")
    nsup = e2.number_input("Suplentes por caso", 0, 10, int(emp0.get("suplentes_por_caso", 1)), key=f"nsup_{V}")
    pol = e3.selectbox("Casos sem controlo", ["excluir", "manter"], index=0 if emp0.get("casos_sem_controlo", "excluir") == "excluir" else 1,
                       key=f"pol_{V}", help="'excluir' retira-os da comparação; ficam listados à parte.")
    seed = e4.number_input("Semente aleatória", value=int(base_cfg["estudo"].get("semente", 12345)), step=1, key=f"seed_{V}",
                           help="Garante que a mesma configuração dá sempre a mesma seleção.")
    st.markdown("**Etapas** — aplicadas por ordem aos casos ainda sem controlos suficientes. "
                "*Exato*: colunas separadas por vírgula (valor igual). *Calibres*: `coluna=largura` separados por `;` (diferença máxima).")
    init_st = ss.init_stages if len(ss.init_stages) else pd.DataFrame([{"Exato": mapping.get("sexo", ""),
                                                                       "Calibres": f"{mapping['idade']}=5" if mapping.get("idade") else ""}])
    stages_df = st.data_editor(init_st, key=f"stages_{V}", num_rows="dynamic", width="stretch",
                               column_config={"Exato": st.column_config.TextColumn(width="medium"), "Calibres": st.column_config.TextColumn(width="large")})
    st.markdown("**Reequilíbrio automático**")
    r1, r2, r3 = st.columns([1, 1, 3])
    reeq_on = r1.toggle("Ativo", value=bool(reeq0.get("ativo")), key=f"reeq_{V}")
    maxit = r2.number_input("Máx. iterações", 0, 10, int(reeq0.get("max_iter", 3)), key=f"maxit_{V}")
    fator = r2.number_input("Calibre = fator × DP", 0.05, 2.0, float(reeq0.get("fator_calibre", 0.5)), 0.05, key=f"fator_{V}")
    cands = r3.multiselect("Variáveis que pode acrescentar ao emparelhamento", cols_all,
                           default=[c for c in reeq0.get("candidatas", []) if c in cols_all], key=f"cands_{V}")
    st.caption("Se algum confundidor destas variáveis tiver p < α, é acrescentado ao emparelhamento e a seleção repete-se; fica a iteração "
               "com menos diferenças. É uma ajuda prática — a escolha das variáveis de emparelhamento deve ser científica.")

# ================================================================ 4. VARIÁVEIS
with tabs[3]:
    st.markdown("Escolha as variáveis a comparar entre grupos e o **papel** de cada uma:\n"
                "- **confundidor** — deve estar equilibrado entre grupos (conta para o reequilíbrio);\n"
                "- **preditor** — candidato a explicar o desfecho; testa-se mas não se emparelha (emparelhar apagaria o sinal);\n"
                "- **interferente** — pode afetar a medição (ex.: hemólise); monitorizar.\n\n"
                "Desmarque **Testar** para variáveis posteriores à amostra (ex.: óbito, duração do internamento).")
    a1, a2 = st.columns([3, 1])
    add = a1.multiselect("Acrescentar variáveis (o tipo é sugerido automaticamente)",
                         [c for c in cols_all if c not in set(ss.init_vars["Coluna"]) and c != idc], key=f"add_{V}")
    if a2.button("Acrescentar", key=f"addb_{V}", disabled=not add):
        new = pd.DataFrame([{"Coluna": c, "Nome": c, "Nome (EN)": "", "Tipo": inferir_tipo(df[c])[0], "Papel": "confundidor",
                             "Testar": True, "Log (SMD)": False, "Rótulos": ""} for c in add], columns=VAR_COLS)
        ss.init_vars = pd.concat([ss.get("vars_snapshot", ss.init_vars), new], ignore_index=True)
        ss.vars_ver = ss.get("vars_ver", 0) + 1   # só a tabela de variáveis recomeça
        st.rerun()
    alpha = st.selectbox("α (nível de significância)", [0.05, 0.01, 0.10],
                         index=[0.05, 0.01, 0.10].index(float(base_cfg["equilibrio"].get("alfa", 0.05))), key=f"alpha_{V}")
    vars_df = st.data_editor(ss.init_vars, key=f"vars_{V}_{ss.get('vars_ver', 0)}", num_rows="dynamic", width="stretch", height=460,
                             column_config={"Coluna": st.column_config.SelectboxColumn(options=cols_all, width="medium"),
                                            "Tipo": st.column_config.SelectboxColumn(options=["continua", "categorica"], width="small",
                                                                                     help="Ver o passo 5 para a explicação."),
                                            "Papel": st.column_config.SelectboxColumn(options=["confundidor", "preditor", "interferente"], width="small"),
                                            "Testar": st.column_config.CheckboxColumn(width="small"),
                                            "Log (SMD)": st.column_config.CheckboxColumn(width="small", help="SMD em escala log (variáveis muito assimétricas)"),
                                            "Nome": st.column_config.TextColumn(width="medium"),
                                            "Nome (EN)": st.column_config.TextColumn(width="medium", help="Nome na tabela em inglês"),
                                            "Rótulos": st.column_config.TextColumn(width="medium", help="Ex.: 0=Não; 1=Sim")})
    ss["vars_snapshot"] = vars_df


# ================================================================ configuração atual
def current_cfg() -> dict:
    cfg = copy.deepcopy(prep_cfg)
    cfg["estudo"]["semente"] = int(seed)
    for g in ("casos", "controlos"):
        cfg["grupos"][g]["nome"] = edited[(g, "nome")]
        if edited[(g, "nome_en")].strip():
            cfg["grupos"][g]["nome_en"] = edited[(g, "nome_en")].strip()
        cfg["grupos"][g]["doente"] = df_to_crit(edited[(g, "doente")])
        cfg["grupos"][g]["dia"] = df_to_crit(edited[(g, "dia")])
    ind = {"regra": rule}
    if rule in ("minimo", "maximo", "alvo") and rcol:
        ind["coluna"] = rcol
    if rule == "alvo":
        ind["alvo"] = float(ralvo)
    cfg["grupos"]["casos"]["indice"] = ind
    cfg["emparelhamento"] = {"razao": int(razao), "suplentes_por_caso": int(nsup), "casos_sem_controlo": pol,
                             "etapas": df_to_stages(stages_df),
                             "reequilibrar": {"ativo": bool(reeq_on), "max_iter": int(maxit), "fator_calibre": float(fator),
                                              "candidatas": list(cands)}} if use_match else None
    vs = df_to_vars(vars_df, base_cfg["equilibrio"]["variaveis"])
    for v in vs:
        t = ss.test_choice.get(v["col"], "auto")
        if t and t != "auto":
            v["teste"] = t
        else:
            v.pop("teste", None)
    cfg["equilibrio"] = {"alfa": float(alpha), "variaveis": vs}
    return cfg


def validate(cfg) -> tuple[list[str], list[str]]:
    errs, warns = list(crit_errors), []
    if not cfg["grupos"]["casos"]["doente"] + cfg["grupos"]["casos"].get("dia", []):
        errs.append("Defina pelo menos um critério para os casos (passo 2).")
    if cfg["emparelhamento"]:
        if not cfg["emparelhamento"]["etapas"]:
            errs.append("Emparelhamento ativo sem etapas (passo 3).")
        for s in cfg["emparelhamento"]["etapas"]:
            for c in list(s["exato"]) + list(s["calibre"]):
                if c not in df.columns:
                    errs.append(f"Coluna de emparelhamento inexistente: {c}")
            for c, w in s["calibre"].items():
                if w <= 0:
                    errs.append(f"O calibre de {c} tem de ser > 0")
                elif c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                    errs.append(f"{c} não é numérica: não pode ter calibre (use-a em 'Exato').")
    if not cfg["equilibrio"]["variaveis"]:
        errs.append("Escolha pelo menos uma variável a comparar (passo 4).")
    miss = [v["col"] for v in cfg["equilibrio"]["variaveis"] if v["col"] not in df.columns]
    if miss:
        warns.append(f"Variáveis sem coluna nos dados (aparecem como 'coluna em falta'): {', '.join(miss)}")
    return errs, warns


try:
    cfg_now = current_cfg()
except ValueError as e:
    cfg_now = None
    st.error(f"Valor inválido nas etapas de emparelhamento (os calibres têm de ser números): {e}")


def run_selection(cfg):
    with st.spinner("A selecionar, emparelhar e testar…"):
        ss.R = select(cfg, df, qc, prov, log=lambda *_: None)
        ss.R_cfg_yaml = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


# ================================================================ 5. ESCOLHA DOS TESTES (pedagógico)
with tabs[4]:
    st.markdown("### Como se escolhe o teste estatístico")
    st.markdown(
        "A escolha segue três perguntas, por esta ordem:\n\n"
        "1. **Que tipo de variável é?** *Categórica* (categorias: sexo, diabetes sim/não, centro) ou *contínua* (medida numa escala: idade, "
        "creatinina). Uma variável discreta com poucos valores pode ser categórica (códigos) ou ordinal (escala, ex.: SOFA): nesse caso "
        "trata-se como contínua com teste não paramétrico.\n"
        "2. **Se é contínua: a distribuição é normal em cada grupo?** Avalia-se com o teste de Shapiro–Wilk (p > 0,05 → compatível com a normal) "
        "e, sobretudo, olhando para o histograma e para o gráfico Q–Q (pontos sobre a diagonal = normal).\n"
        "   - Normal nos dois grupos → **t de Student** (ou **t de Welch** se as variâncias forem diferentes — teste de Levene p < 0,05).\n"
        "   - Não normal → **Mann–Whitney U** (não paramétrico; compara distribuições/medianas).\n"
        "3. **Se é categórica: há poucos casos por célula?** Calcula-se a frequência esperada de cada célula da tabela grupo × categoria.\n"
        "   - Todas ≥ 5 → **qui-quadrado de Pearson**.\n"
        "   - Alguma < 5 → **teste exato de Fisher**.\n\n"
        "Os casos e os controlos são tratados como amostras independentes, como é habitual numa tabela de características. "
        "A tabela abaixo aplica estas regras às suas variáveis, explica a recomendação e permite escolher outro teste.")
    if cfg_now is None:
        st.stop()
    errs, warns = validate(cfg_now)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    if st.button("Correr a seleção", type="primary", disabled=bool(errs), key="run5"):
        run_selection(cfg_now)
    R = ss.get("R")
    if R is None:
        st.info("Carregue em **Correr a seleção** para ver o diagnóstico de cada variável nos grupos selecionados.")
    else:
        names = R["names"]
        rows = []
        for v in R["cfg"]["equilibrio"]["variaveis"]:
            if v["col"] not in R["sel_cases"].columns or not v.get("testar", True):
                continue
            e = explorar(R["sel_cases"][v["col"]], R["sel_ctrl"][v["col"]], v)
            e["Tipo (sugestão automática)"] = inferir_tipo(pd.concat([R["sel_cases"][v["col"]], R["sel_ctrl"][v["col"]]]))[0]
            rows.append(e)
        X = pd.DataFrame(rows)
        if X.empty:
            st.info("Não há variáveis testáveis.")
        else:
            st.markdown("#### Diagnóstico e recomendação por variável")
            show = X[[c for c in ["Variável", "Tipo", "Teste recomendado", "Justificação", "n (casos/controlos)", "Shapiro–Wilk p (casos)",
                                  "Shapiro–Wilk p (controlos)", "Levene p", "Frequência esperada mínima", "Tipo (sugestão automática)"]
                      if c in X.columns]].copy()
            for c in ["Shapiro–Wilk p (casos)", "Shapiro–Wilk p (controlos)", "Levene p", "Frequência esperada mínima"]:
                if c in show:
                    show[c] = [("" if pd.isna(x) else (f"{x:.1f}" if c.startswith("Frequência") else ("<0,001" if x < 0.001 else f"{x:.3f}")))
                               for x in show[c]]
            st.dataframe(show, hide_index=True, width="stretch", height=min(40 * (len(show) + 1), 520),
                         column_config={"Justificação": st.column_config.TextColumn(width="large"),
                                        "Teste recomendado": st.column_config.TextColumn(width="medium"),
                                        "Variável": st.column_config.TextColumn(width="medium")})
            with st.expander("Ler a justificação completa de cada variável"):
                for _, r in X.iterrows():
                    st.markdown(f"**{r['Variável']}** — *{r['Teste recomendado']}*. {r['Justificação']}")
            mism = X[X["Tipo"] != X["Tipo (sugestão automática)"]]
            if len(mism):
                st.caption("Tipo diferente da sugestão automática em: " + ", ".join(mism["Variável"]) +
                           ". Pode estar certo (ex.: códigos numéricos são categóricos) — confirme no passo 4.")

            st.markdown("#### Teste a usar em cada variável")
            st.caption("*Automático* aplica as regras acima (Shapiro–Wilk → Student ou Mann–Whitney; Fisher se esperada < 5). "
                       "Escolher outro teste fica registado na configuração e nas notas da tabela.")
            choice_rows = []
            for _, r in X.iterrows():
                opts_map = TESTES_CONTINUA if r["Tipo"] == "continua" else TESTES_CATEGORICA
                choice_rows.append({"Variável": r["Variável"], "Coluna": r["Coluna"], "Recomendado": r["Teste recomendado"],
                                    "Teste escolhido": opts_map.get(ss.test_choice.get(r["Coluna"], "auto"), opts_map["auto"])})
            all_opts = list(dict.fromkeys(list(TESTES_CONTINUA.values()) + list(TESTES_CATEGORICA.values())))
            ch = st.data_editor(pd.DataFrame(choice_rows), key=f"choice_{V}_{len(choice_rows)}", hide_index=True, width="stretch",
                                disabled=["Variável", "Coluna", "Recomendado"],
                                column_config={"Teste escolhido": st.column_config.SelectboxColumn(options=all_opts, width="large")})
            inv = {**{t: k for k, t in TESTES_CONTINUA.items()}, **{t: k for k, t in TESTES_CATEGORICA.items()}}
            new_choice = {}
            bad = []
            for (_, r), (_, o) in zip(ch.iterrows(), X.iterrows()):
                k = inv.get(r["Teste escolhido"], "auto")
                valid = TESTES_CONTINUA if o["Tipo"] == "continua" else TESTES_CATEGORICA
                if k not in valid:
                    bad.append(r["Variável"])
                    k = "auto"
                new_choice[r["Coluna"]] = k
            for b in bad:
                st.error(f"{b}: esse teste não se aplica a este tipo de variável — mantém-se o automático.")
            if new_choice != {c: ss.test_choice.get(c, "auto") for c in new_choice}:
                ss.test_choice.update(new_choice)
                st.rerun()
            if ss.get("R_cfg_yaml") and yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False) != ss.R_cfg_yaml:
                st.warning("Mudou a configuração ou um teste desde a última execução.")
                if st.button("Aplicar e correr novamente", type="primary", key="rerun5", disabled=bool(errs)):
                    run_selection(current_cfg())
                    st.rerun()

            st.markdown("#### Ver a distribuição de uma variável contínua")
            conts = X[X["Tipo"] == "continua"]
            if len(conts):
                pick = st.selectbox("Variável", conts["Variável"].tolist(), key=f"pick_{V}")
                rr = conts[conts["Variável"] == pick].iloc[0]
                for ch_ in dist_plots(R["sel_cases"][rr["Coluna"]], R["sel_ctrl"][rr["Coluna"]], names, pick):
                    st.altair_chart(ch_)
                st.caption("Histograma: forma de sino e simétrica sugere normalidade. Gráfico Q–Q: se os pontos seguem a diagonal, a distribuição "
                           "é aproximadamente normal; curvas nas pontas indicam caudas pesadas ou assimetria. " + rr["Justificação"])

# ================================================================ 6. RESULTADOS
with tabs[5]:
    if cfg_now is None:
        st.stop()
    errs, _ = validate(cfg_now)
    if st.button("Correr a seleção", type="primary", disabled=bool(errs), key="run6"):
        run_selection(cfg_now)
    R = ss.get("R")
    if R is None:
        st.info("Reveja os passos 1–4 e carregue em **Correr a seleção** (aqui ou no passo 5).")
    else:
        if yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False) != ss.R_cfg_yaml:
            st.warning("A configuração mudou desde a última execução. Os resultados abaixo são da execução anterior.")
        names = R["names"]
        t = R["t_after"]
        nsig = int(((t["Diferença significativa"] == "Sim") & t["Papel"].astype(str).str.startswith("Confundidor")).sum())
        m = R["match"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(names[0], len(R["sel_cases"]))
        k2.metric(f"{names[1]}" + (" (principais)" if cfg_now["emparelhamento"] else ""), len(R["sel_ctrl"]))
        k3.metric("Suplentes", int((m.papel == "suplente").sum()) if len(m) else 0)
        k4.metric("Casos sem controlo", len(R["cases"]) - R["sel_cases"][idc].nunique() if cfg_now["emparelhamento"] else 0)
        k5.metric(f"Confundidores com p < {R['meta']['alfa']}", nsig, help="Preditores e interferentes são testados mas não contam.")

        st.markdown("#### Descarregar")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Excel completo", _tmp_bytes(lambda p: write(R, p), ".xlsx"), file_name="selecao.xlsx", type="primary",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           help="Seleção, comparação entre grupos, funil, suplentes, controlo de qualidade e configuração.")
        d2.download_button("Configuração (YAML)", ss.R_cfg_yaml.encode("utf-8"), file_name="configuracao_selpop.yaml",
                           help="Guarde com os resultados: permite reproduzir exatamente esta seleção.")
        lang = d3.radio("Língua da tabela do artigo", ["en", "pt"], horizontal=True, format_func=lambda x: {"en": "Inglês", "pt": "Português"}[x],
                        key="pub_lang")
        with_smd = d3.checkbox("Incluir coluna SMD", value=False, key="pub_smd")
        b1, b2, b3 = st.columns(3)
        b1.download_button("Tabela de características (Word)", _tmp_bytes(lambda p: write_docx(R, p, lang=lang, smd=with_smd), ".docx"),
                           file_name=f"tabela_caracteristicas_{lang}.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        b2.download_button("Dados para SPSS/PSPP/jamovi (.sav)", _tmp_bytes(lambda p: write_sav(R, p, lang=lang), ".sav"),
                           file_name="dados_selecao.sav", mime="application/octet-stream")
        b3.download_button("Sintaxe de validação (.sps)", _tmp_bytes(lambda p: write_sps(R, p, "dados_selecao.sav", lang=lang), ".sps"),
                           file_name="validacao.sps", mime="text/plain",
                           help="Reproduz os testes no SPSS/PSPP. Comparar com 'Pearson Chi-Square' e 'Equal variances assumed'.")

        st.markdown("#### Tabela para o artigo (pré-visualização)")
        pub, pmeta = build_table(R, lang, with_smd)
        st.markdown(f"**{pmeta['title']}**")
        prev = pub.rename(columns={"Variável": TXT_PUB[lang]["var"], "casos": pmeta["col_a"], "controlos": pmeta["col_b"]})
        prev[TXT_PUB[lang]["var"]] = [("    " if tt == "nivel" else "") + v for tt, v in zip(pub["tipo"], pub["Variável"])]
        prev["p"] = [f"{p}{('' if not isinstance(k, str) else ' ' + k)}" for p, k in zip(pub["p"], pub["teste"])]
        st.dataframe(prev.drop(columns=["tipo", "teste"]), hide_index=True, width="stretch", height=440)
        for n in pmeta["notes"]:
            st.caption(n)

        st.markdown("#### Comparação detalhada (após emparelhamento)")
        pnum = pd.to_numeric(t["p"], errors="coerce")
        show = t.copy().astype(object).where(t.notna(), "")
        show["p"] = [("" if pd.isna(v) else ("<0,001" if v < 0.001 else f"{v:.3f}")) for v in pnum]
        show["SMD"] = [("" if pd.isna(v) else f"{v:.2f}") for v in pd.to_numeric(t["SMD"], errors="coerce")]
        flag = (pnum < R["meta"]["alfa"]).fillna(False).values
        st.dataframe(show.style.apply(lambda r: ["background-color: #fbe3d6" if flag[r.name] else "" for _ in r], axis=1),
                     hide_index=True, width="stretch", height=min(36 * (len(show) + 1), 620),
                     column_config={"Variável": st.column_config.TextColumn(width="medium"), "Nível": st.column_config.TextColumn(width="medium")})
        if cfg_now["emparelhamento"]:
            st.markdown("#### Equilíbrio antes e depois do emparelhamento")
            st.caption("Cada ponto é uma variável testada. A linha tracejada marca |SMD| = 0,1 (equilíbrio adequado). "
                       "Com grupos pequenos, valores até ~0,2 são frequentes e devem ser discutidos, mesmo com p ≥ α.")
            st.altair_chart(love_plot(R["t_before"], t, 0.1))
        f1, f2 = st.columns([3, 2])
        with f1:
            st.markdown("#### Funil")
            st.dataframe(R["funil"], hide_index=True, width="stretch")
        with f2:
            if len(R["iters"]):
                st.markdown("#### Reequilíbrio")
                st.dataframe(R["iters"][["Iteração", "Casos emparelhados", "Confundidores com p < α", "Escolhida"]], hide_index=True, width="stretch")
        st.markdown("#### Amostras selecionadas")
        cmap = cfg_now["colunas"]
        sel = pd.concat([R["sel_cases"].assign(Grupo=names[0]), R["sel_ctrl"].assign(Grupo=names[1])])
        showcols = ["Grupo", idc] + [cmap[k] for k in ("n_amostra", "data_colheita") if cmap.get(k) in sel.columns] + ["_caso"] + \
                   [v["col"] for v in cfg_now["equilibrio"]["variaveis"] if v["col"] in sel.columns][:10]
        sel = sel[[c for c in dict.fromkeys(showcols) if c in sel.columns]].rename(columns={"_caso": "Emparelhado com"})
        if cmap.get("data_colheita") in sel.columns:
            sel[cmap["data_colheita"]] = pd.to_datetime(sel[cmap["data_colheita"]], errors="coerce").dt.date
        if "Emparelhado com" in sel:
            sel["Emparelhado com"] = ["" if pd.isna(v) else str(v) for v in sel["Emparelhado com"]]
        st.dataframe(sel, hide_index=True, width="stretch", height=360)
        with st.expander("Controlo de qualidade dos dados"):
            st.caption("Incoerências detetadas (não são corrigidas). Confirme na fonte os registos listados.")
            st.dataframe(qc, hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(prov), hide_index=True, width="stretch")

# ================================================================ CONFIGURAÇÃO
with tabs[6]:
    st.caption("Configuração equivalente ao que está nos passos. Guarde-a para repetir a análise (aqui, em *carregar uma configuração*, "
               "ou na linha de comandos: `python -m selpop config.yaml`).")
    if cfg_now is not None:
        txt = yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False)
        st.download_button("Descarregar YAML", txt.encode("utf-8"), file_name="configuracao_selpop.yaml")
        st.code(txt, language="yaml")
