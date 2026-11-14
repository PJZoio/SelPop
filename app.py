"""SelPop — interface Streamlit para a seleção de populações-alvo e controlos.

Correr localmente:  streamlit run app.py
Os ficheiros carregados ficam apenas na memória da sessão local; nada é gravado em disco
exceto o Excel que o utilizador descarregar.
"""
from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from selpop import __version__
from selpop.cohort import q
from selpop.io import load_sources
from selpop.pipeline import prep_key_of, prepare, select
from selpop.report import write

ROOT = Path(__file__).parent
TEMPLATES = {p.stem: p for p in sorted((ROOT / "estudos").glob("*.yaml"))}
INDEX_RULES = ["primeiro", "ultimo", "minimo", "maximo", "alvo", "aleatorio"]
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"

st.set_page_config(page_title="SelPop — seleção de populações", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------- conversões config <-> tabelas editáveis
def crit_to_df(spec) -> pd.DataFrame:
    rows = []
    if isinstance(spec, str):
        spec = [spec] if spec.strip() else []
    for item in spec or []:
        if isinstance(item, dict):
            rows.append({"Ativo": True, "Nome": item.get("nome", item["expr"]), "Expressão": item["expr"]})
        else:
            rows.append({"Ativo": True, "Nome": item, "Expressão": item})
    return pd.DataFrame(rows, columns=["Ativo", "Nome", "Expressão"])


def df_to_crit(d: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in d.iterrows():
        e = str(r.get("Expressão") or "").strip()
        if bool(r.get("Ativo", True)) and e and e != "nan":
            out.append({"nome": str(r.get("Nome") or e), "expr": e})
    return out


def stages_to_df(stages) -> pd.DataFrame:
    rows = [{"Exato": ", ".join(s.get("exato") or []),
             "Calibres": "; ".join(f"{k}={v:g}" for k, v in (s.get("calibre") or {}).items())} for s in stages or []]
    return pd.DataFrame(rows, columns=["Exato", "Calibres"])


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


def vars_to_df(vs) -> pd.DataFrame:
    return pd.DataFrame([{"Coluna": v["col"], "Nome": v.get("nome", v["col"]), "Tipo": v.get("tipo", "continua"),
                          "Papel": v.get("papel", "confundidor"), "Testar": bool(v.get("testar", True)),
                          "Log (SMD)": bool(v.get("log", False)), "Rótulos": labels_to_txt(v.get("rotulos"))} for v in vs],
                        columns=["Coluna", "Nome", "Tipo", "Papel", "Testar", "Log (SMD)", "Rótulos"])


def df_to_vars(d: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in d.iterrows():
        col = str(r.get("Coluna") or "").strip()
        if not col or col == "nan":
            continue
        v = {"col": col, "nome": str(r.get("Nome") or col), "tipo": r.get("Tipo") or "continua"}
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


# ---------------------------------------------------------------- estado
def load_template(cfg: dict):
    ss = st.session_state
    ss.base_cfg = cfg
    ss.ver = ss.get("ver", 0) + 1
    G = cfg["grupos"]
    for g in ("casos", "controlos"):
        for lvl in ("doente", "dia"):
            ss[f"init_{g}_{lvl}"] = crit_to_df(G[g].get(lvl))
    ss.init_stages = stages_to_df((cfg.get("emparelhamento") or {}).get("etapas"))
    ss.init_vars = vars_to_df(cfg["equilibrio"]["variaveis"])
    ss.pop("R", None)


def prep_key(cfg: dict) -> str:
    """Só estas partes da configuração afetam a preparação da base (o resto pode mudar sem recalcular)."""
    return prep_key_of(cfg)


@st.cache_data(show_spinner="A preparar a base (variáveis derivadas, LRA, TFGe)…", max_entries=4)
def cached_prepare(main_name: str, main_bytes: bytes, aux: tuple, key_yaml: str):
    cfg = yaml.safe_load(key_yaml)
    aux_srcs = [None if a is None else (a[0], a[1]) for a in aux]
    raw, prov = load_sources(cfg, (main_name, main_bytes), aux_srcs)
    df, qc = prepare(cfg, raw)
    return df, qc, prov


def excel_bytes(R) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "selecao.xlsx"
        write(R, p)
        return p.read_bytes()


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
        color=alt.Color("Momento:N", scale=alt.Scale(domain=["Antes do emparelhamento", "Depois do emparelhamento"],
                                                      range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None, labelLimit=320, labelFontSize=12)),
        shape=alt.Shape("Momento:N", scale=alt.Scale(range=["circle", "diamond"]), legend=None),
        tooltip=["Variável", "Momento", alt.Tooltip("SMD:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f")],
    )
    pts = base.mark_point(size=90, filled=True, opacity=0.95, stroke="white", strokeWidth=1.5)
    rule = alt.Chart(pd.DataFrame({"x": [lim]})).mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q")
    return (rule + pts).properties(height=max(240, 30 * len(order))).configure_view(stroke=None)


# ---------------------------------------------------------------- barra lateral
ss = st.session_state
with st.sidebar:
    st.title("SelPop")
    st.caption(f"Seleção de casos e controlos · v{__version__}")
    st.subheader("1. Estudo")
    tname = st.selectbox("Modelo de estudo", list(TEMPLATES), format_func=lambda s: s.replace("_", " "))
    up_cfg = st.file_uploader("…ou carregar uma configuração YAML", type=["yaml", "yml"])
    src_id = f"up:{up_cfg.name}:{up_cfg.size}" if up_cfg else f"tpl:{tname}"
    if ss.get("cfg_src") != src_id:
        cfg0 = yaml.safe_load(up_cfg.getvalue()) if up_cfg else yaml.safe_load(TEMPLATES[tname].read_text(encoding="utf-8"))
        load_template(cfg0)
        ss.cfg_src = src_id
    base_cfg = ss.base_cfg

    st.subheader("2. Dados")
    main_up = st.file_uploader("Base principal (Excel)", type=["xlsx", "xls"])
    aux_ups = []
    for i, a in enumerate(base_cfg["dados"].get("auxiliares") or []):
        aux_ups.append(st.file_uploader(f"Auxiliar: {Path(a['ficheiro']).name} (opcional)", type=["xlsx", "xls"], key=f"aux{i}_{ss.ver}"))
    st.info("Os ficheiros ficam só na memória desta sessão. Correr a aplicação localmente; "
            "não a publicar num servidor público com dados de doentes.", icon="🔒")

st.title(base_cfg["estudo"].get("nome", "Estudo"))
st.caption(base_cfg["estudo"].get("descricao", ""))

if not main_up:
    st.markdown("**Para começar:** escolher o modelo de estudo e carregar a base principal na barra lateral.\n\n"
                "Depois: rever critérios e emparelhamento, correr a seleção e descarregar o Excel com a lista de amostras, "
                "a Tabela 1 e o registo das decisões.")
    st.stop()

aux_key = tuple(None if u is None else (u.name, u.getvalue()) for u in aux_ups)
try:
    df, qc, prov = cached_prepare(main_up.name, main_up.getvalue(), aux_key, prep_key(base_cfg))
except Exception as e:  # erros de mapeamento de colunas, folha errada, etc.
    st.error(f"Não foi possível preparar a base com esta configuração: {e}")
    st.stop()
idc = base_cfg["dados"]["id"]
missing_aux = [Path(a["ficheiro"]).name for a, u in zip(base_cfg["dados"].get("auxiliares") or [], aux_ups) if u is None]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Doentes", f"{df[idc].nunique():,}".replace(",", " "))
c2.metric("Dias com dados", f"{len(df):,}".replace(",", " "))
c3.metric("Dias com amostra", f"{int(df['amostra'].sum()):,}".replace(",", " "))
c4.metric("Alertas de qualidade", int((qc["N.º doentes"] > 0).sum()), help="Ver separador Dados")
if missing_aux:
    st.warning(f"Sem ficheiro auxiliar ({', '.join(missing_aux)}): as variáveis que dele dependem ficam vazias.")

tabs = st.tabs(["Critérios", "Emparelhamento", "Variáveis da Tabela 1", "Resultados", "Dados", "Configuração"])
cols_all = sorted(df.columns, key=str)

# ---------------------------------------------------------------- 1. critérios
with tabs[0]:
    st.markdown("Cada linha é um critério, aplicado por ordem. Expressões em sintaxe pandas "
                "(`and`, `or`, `not`, `==`, `>=`); colunas com espaços ou acentos entre acentos graves. "
                "Desmarcar **Ativo** para testar o efeito de retirar um critério.")
    edited = {}
    gcols = st.columns(2)
    for gi, g in enumerate(("casos", "controlos")):
        with gcols[gi]:
            gname = st.text_input("Nome do grupo", base_cfg["grupos"][g].get("nome", g), key=f"name_{g}_{ss.ver}")
            for lvl, lab in (("doente", "Critérios de doente"), ("dia", "Critérios de dia/amostra")):
                st.markdown(f"**{lab}**")
                edited[(g, lvl)] = st.data_editor(
                    ss[f"init_{g}_{lvl}"], key=f"ed_{g}_{lvl}_{ss.ver}", num_rows="dynamic", width="stretch",
                    column_config={"Ativo": st.column_config.CheckboxColumn(width="small"),
                                   "Nome": st.column_config.TextColumn(width="medium"),
                                   "Expressão": st.column_config.TextColumn(width="large")})
            edited[(g, "nome")] = gname
            if g == "casos":
                ind = base_cfg["grupos"]["casos"].get("indice") or {"regra": "primeiro"}
                st.markdown("**Dia-índice dos casos** (um por doente)")
                i1, i2, i3 = st.columns(3)
                rule = i1.selectbox("Regra", INDEX_RULES, index=INDEX_RULES.index(ind.get("regra", "primeiro")), key=f"rule_{ss.ver}")
                rcol = i2.selectbox("Coluna", [""] + cols_all, index=([""] + cols_all).index(ind["coluna"]) if ind.get("coluna") in cols_all else 0,
                                    key=f"rcol_{ss.ver}", disabled=rule not in ("minimo", "maximo", "alvo"))
                ralvo = i3.number_input("Alvo", value=float(ind.get("alvo", 0) or 0), key=f"ralvo_{ss.ver}", disabled=rule != "alvo")
            else:
                st.caption("O dia dos controlos é escolhido no emparelhamento (o mais próximo do caso nas variáveis de calibre).")

    st.divider()
    st.markdown("**Funil em tempo real** (antes do emparelhamento)")
    fcols = st.columns(2)
    crit_errors = []
    for gi, g in enumerate(("casos", "controlos")):
        rows, m = [], pd.Series(True, index=df.index)
        rows.append({"Critério": "Base completa", "Doentes": df[idc].nunique(), "Dias": len(df)})
        for lvl in ("doente", "dia"):
            for c in df_to_crit(edited[(g, lvl)]):
                try:
                    m = m & q(df, c["expr"])
                    rows.append({"Critério": f"[{lvl}] {c['nome']}", "Doentes": df.loc[m, idc].nunique(), "Dias": int(m.sum())})
                except Exception as e:
                    crit_errors.append(f"{edited[(g, 'nome')]} — «{c['expr']}»: {e}")
                    rows.append({"Critério": f"[{lvl}] {c['nome']} — ERRO", "Doentes": None, "Dias": None})
        with fcols[gi]:
            st.caption(edited[(g, "nome")])
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for e in crit_errors:
        st.error(e)
    with st.expander("Colunas disponíveis para as expressões"):
        info = pd.DataFrame({"Coluna": df.columns.astype(str), "Tipo": df.dtypes.astype(str).values,
                             "Exemplos": [", ".join(map(str, df[c].dropna().unique()[:4])) for c in df.columns]})
        st.dataframe(info, hide_index=True, width="stretch", height=320)

# ---------------------------------------------------------------- 2. emparelhamento
emp0 = base_cfg.get("emparelhamento") or {}
reeq0 = emp0.get("reequilibrar") or {}
with tabs[1]:
    use_match = st.toggle("Emparelhar controlos aos casos", value=bool(emp0), key=f"usem_{ss.ver}")
    e1, e2, e3, e4 = st.columns(4)
    razao = e1.number_input("Controlos por caso", 1, 10, int(emp0.get("razao", 2)), key=f"razao_{ss.ver}")
    nsup = e2.number_input("Suplentes por caso", 0, 10, int(emp0.get("suplentes_por_caso", 1)), key=f"nsup_{ss.ver}")
    pol = e3.selectbox("Casos sem controlo", ["excluir", "manter"], index=0 if emp0.get("casos_sem_controlo", "excluir") == "excluir" else 1,
                       key=f"pol_{ss.ver}", help="'excluir' retira-os da Tabela 1; ficam listados à parte.")
    seed = e4.number_input("Semente aleatória", value=int(base_cfg["estudo"].get("semente", 12345)), step=1, key=f"seed_{ss.ver}")
    st.markdown("**Etapas** — aplicadas por ordem aos casos que ainda não completaram a razão. "
                "Exato: colunas separadas por vírgula. Calibres: `coluna=largura`, separados por ponto e vírgula.")
    stages_df = st.data_editor(ss.init_stages, key=f"stages_{ss.ver}", num_rows="dynamic", width="stretch",
                               column_config={"Exato": st.column_config.TextColumn(width="medium"),
                                              "Calibres": st.column_config.TextColumn(width="large")})
    st.markdown("**Reequilíbrio automático**")
    r1, r2, r3 = st.columns([1, 1, 3])
    reeq_on = r1.toggle("Ativo", value=bool(reeq0.get("ativo")), key=f"reeq_{ss.ver}")
    maxit = r2.number_input("Máx. iterações", 0, 10, int(reeq0.get("max_iter", 3)), key=f"maxit_{ss.ver}")
    fator = r2.number_input("Calibre = fator × DP", 0.05, 2.0, float(reeq0.get("fator_calibre", 0.5)), 0.05, key=f"fator_{ss.ver}")
    cand_opts = [v["Coluna"] for _, v in ss.init_vars.iterrows()]
    cands = r3.multiselect("Variáveis que pode acrescentar ao emparelhamento",
                           sorted(set(cand_opts) | set(reeq0.get("candidatas", []))),
                           default=[c for c in reeq0.get("candidatas", [])], key=f"cands_{ss.ver}")
    st.caption("Se alguma destas variáveis tiver p < α, é acrescentada ao emparelhamento (contínua → calibre; categórica → exato) "
               "e a seleção repete-se. Fica a iteração com menos variáveis significativas. É uma ajuda prática: "
               "a escolha das variáveis de emparelhamento deve ser científica.")

# ---------------------------------------------------------------- 3. variáveis
with tabs[2]:
    a1, a2 = st.columns([1, 3])
    alpha = a1.selectbox("α (nível de significância)", [0.05, 0.01, 0.10],
                         index=[0.05, 0.01, 0.10].index(float(base_cfg["equilibrio"].get("alfa", 0.05))), key=f"alpha_{ss.ver}")
    a2.caption("Contínuas: mediana (P25–P75); Shapiro–Wilk → t de Student ou Mann–Whitney U. "
               "Categóricas: n (%); qui-quadrado ou Fisher exato (frequência esperada < 5). "
               "Desmarcar **Testar** para variáveis posteriores à amostra (ex.: óbito, duração do internamento). "
               "Rótulos: `0=Não; 1=Sim`.")
    vars_df = st.data_editor(ss.init_vars, key=f"vars_{ss.ver}", num_rows="dynamic", width="stretch", height=520,
                             column_config={"Coluna": st.column_config.SelectboxColumn(options=cols_all, width="medium"),
                                            "Tipo": st.column_config.SelectboxColumn(options=["continua", "categorica"], width="small"),
                                            "Testar": st.column_config.CheckboxColumn(width="small"),
                                            "Papel": st.column_config.SelectboxColumn(
                                                options=["confundidor", "preditor", "interferente"], width="small",
                                                help="Só os confundidores contam para o reequilíbrio; preditores e interferentes são testados mas não emparelhados."),
                                            "Log (SMD)": st.column_config.CheckboxColumn(width="small", help="SMD em escala log (variáveis muito assimétricas)"),
                                            "Nome": st.column_config.TextColumn(width="medium"),
                                            "Rótulos": st.column_config.TextColumn(width="medium")})


# ---------------------------------------------------------------- montar a configuração atual
def current_cfg() -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["estudo"]["semente"] = int(seed)
    for g in ("casos", "controlos"):
        cfg["grupos"][g]["nome"] = edited[(g, "nome")]
        cfg["grupos"][g]["doente"] = df_to_crit(edited[(g, "doente")])
        cfg["grupos"][g]["dia"] = df_to_crit(edited[(g, "dia")])
    ind = {"regra": rule}
    if rule in ("minimo", "maximo", "alvo") and rcol:
        ind["coluna"] = rcol
    if rule == "alvo":
        ind["alvo"] = float(ralvo)
    cfg["grupos"]["casos"]["indice"] = ind
    if use_match:
        cfg["emparelhamento"] = {"razao": int(razao), "suplentes_por_caso": int(nsup), "casos_sem_controlo": pol,
                                 "etapas": df_to_stages(stages_df),
                                 "reequilibrar": {"ativo": bool(reeq_on), "max_iter": int(maxit), "fator_calibre": float(fator),
                                                  "candidatas": list(cands)}}
    else:
        cfg["emparelhamento"] = None
    cfg["equilibrio"] = {"alfa": float(alpha), "variaveis": df_to_vars(vars_df)}
    return cfg


def validate(cfg) -> list[str]:
    errs = list(crit_errors)
    if cfg["emparelhamento"]:
        if not cfg["emparelhamento"]["etapas"]:
            errs.append("Emparelhamento ativo sem etapas definidas.")
        for s in cfg["emparelhamento"]["etapas"]:
            for c in list(s["exato"]) + list(s["calibre"]):
                if c not in df.columns:
                    errs.append(f"Coluna de emparelhamento inexistente: {c}")
            for c, w in s["calibre"].items():
                if w <= 0:
                    errs.append(f"Calibre de {c} tem de ser > 0")
    return errs


def warnings_for(cfg) -> list[str]:
    miss = [v["col"] for v in cfg["equilibrio"]["variaveis"] if v["col"] not in df.columns]
    return [f"Variáveis da Tabela 1 sem coluna nos dados (aparecem como 'coluna em falta'): {', '.join(miss)}"] if miss else []


try:
    cfg_now = current_cfg()
except ValueError as e:
    cfg_now = None
    st.error(f"Valor inválido nas etapas de emparelhamento (calibres devem ser números): {e}")

# ---------------------------------------------------------------- 4. resultados
with tabs[3]:
    if cfg_now is None:
        st.stop()
    errs = validate(cfg_now)
    for e in errs:
        st.error(e)
    for w in warnings_for(cfg_now):
        st.warning(w)
    run = st.button("Correr seleção", type="primary", disabled=bool(errs))
    if run:
        with st.spinner("A selecionar, emparelhar e testar…"):
            ss.R = select(cfg_now, df, qc, prov, log=lambda *_: None)
            ss.R_cfg_yaml = yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False)
    R = ss.get("R")
    if R is None:
        st.info("Rever os separadores anteriores e carregar em **Correr seleção**.")
    else:
        if yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False) != ss.R_cfg_yaml:
            st.warning("A configuração mudou desde a última execução. Os resultados abaixo são da execução anterior.")
        names = R["names"]
        t = R["t_after"]
        nsig = int(((t["Diferença significativa"] == "Sim") & t["Papel"].astype(str).str.startswith("Confundidor")).sum())
        m = R["match"]
        nsup_n = int((m.papel == "suplente").sum()) if len(m) else 0
        unmatched = len(R["cases"]) - R["sel_cases"][idc].nunique() if cfg_now["emparelhamento"] else 0
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(names[0], len(R["sel_cases"]))
        k2.metric(f"{names[1]} (principais)", len(R["sel_ctrl"]))
        k3.metric("Suplentes", nsup_n)
        k4.metric("Casos sem controlo", unmatched)
        k5.metric(f"Confundidores com p < {R['meta']['alfa']}", nsig,
                  help="Preditores candidatos e interferentes são testados mas não contam: não se emparelham.")

        d1, d2 = st.columns(2)
        d1.download_button("Descarregar Excel completo", excel_bytes(R), file_name=f"selecao_{tname}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        d2.download_button("Descarregar configuração (YAML)", ss.R_cfg_yaml.encode("utf-8"), file_name=f"{tname}_config.yaml",
                           help="Guardar com o Excel: permite reproduzir exatamente esta seleção.")

        st.subheader("Tabela 1 (após emparelhamento)")
        pnum = pd.to_numeric(t["p"], errors="coerce")
        show = t.copy().astype(object).where(t.notna(), "")
        show["p"] = [("" if pd.isna(v) else ("<0,001" if v < 0.001 else f"{v:.3f}")) for v in pnum]
        show["SMD"] = [("" if pd.isna(v) else f"{v:.2f}") for v in pd.to_numeric(t["SMD"], errors="coerce")]
        flag = (pnum < R["meta"]["alfa"]).fillna(False).values
        sty = show.style.apply(lambda r: ["background-color: #fbe3d6" if flag[r.name] else "" for _ in r], axis=1)
        st.dataframe(sty, hide_index=True, width="stretch", height=min(36 * (len(show) + 1), 760),
                     column_config={"Variável": st.column_config.TextColumn(width="medium"),
                                    "Nível": st.column_config.TextColumn(width="medium")})

        st.subheader("Equilíbrio antes e depois do emparelhamento")
        st.caption("Cada ponto é uma variável testada. A linha tracejada marca |SMD| = 0,1, limiar habitual de equilíbrio adequado. "
                   "Com grupos pequenos, valores até ~0,2 são frequentes e devem ser discutidos, mesmo com p ≥ α.")
        st.altair_chart(love_plot(R["t_before"], t, 0.1), width="stretch")

        f1, f2 = st.columns([3, 2])
        with f1:
            st.subheader("Funil")
            st.dataframe(R["funil"], hide_index=True, width="stretch")
        with f2:
            if len(R["iters"]):
                st.subheader("Reequilíbrio")
                st.dataframe(R["iters"][["Iteração", "Casos emparelhados", "Confundidores com p < α", "Escolhida"]],
                             hide_index=True, width="stretch")
        st.subheader("Amostras selecionadas")
        cmap = cfg_now["colunas"]
        sel = pd.concat([R["sel_cases"].assign(Grupo=names[0]), R["sel_ctrl"].assign(Grupo=names[1])])
        showcols = ["Grupo", idc, cmap["n_amostra"], cmap["data_colheita"], "_caso"] + \
                   [c for c in cfg_now.get("saida", {}).get("colunas", []) if c in sel.columns]
        sel = sel[[c for c in showcols if c in sel.columns]].rename(columns={"_caso": "Emparelhado com"})
        sel[cmap["data_colheita"]] = pd.to_datetime(sel[cmap["data_colheita"]]).dt.date
        sel["Emparelhado com"] = [("" if pd.isna(v) else str(int(v))) for v in pd.to_numeric(sel["Emparelhado com"], errors="coerce")]
        st.dataframe(sel, hide_index=True, width="stretch", height=360)

# ---------------------------------------------------------------- 5. dados
with tabs[4]:
    st.subheader("Controlo de qualidade")
    st.caption("Incoerências detetadas nos dados (não são corrigidas). Confirmar na fonte os doentes listados.")
    st.dataframe(qc, hide_index=True, width="stretch")
    st.subheader("Ficheiros carregados")
    st.dataframe(pd.DataFrame(prov), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 6. configuração
with tabs[5]:
    st.caption("Configuração equivalente ao que está nos separadores. Pode ser descarregada e usada na linha de comandos: "
               "`python -m selpop config.yaml`.")
    txt = yaml.safe_dump(cfg_now, allow_unicode=True, sort_keys=False)
    st.download_button("Descarregar YAML", txt.encode("utf-8"), file_name=f"{tname}_config.yaml")
    st.code(txt, language="yaml")
