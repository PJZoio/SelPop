"""Tabela de comparação e testes de equilíbrio entre dois grupos.

Segue a prática dos artigos do grupo (Fonseca et al. 2025; Araújo et al. 2025):
- contínuas: mediana (P25–P75); Shapiro–Wilk em cada grupo → t de Student se ambos normais, senão Mann–Whitney U;
- categóricas: n (%); qui-quadrado, ou exato de Fisher quando alguma frequência esperada < 5.
Acrescenta a diferença média padronizada (SMD), recomendada para avaliar equilíbrio após emparelhamento
porque não depende do tamanho da amostra (|SMD| < 0,1 costuma considerar-se equilíbrio adequado).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    return float(p)


def smd_cont(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if s == 0 else (a.mean() - b.mean()) / s


def smd_prop(p1, p2):
    v = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    return 0.0 if v == 0 else (p1 - p2) / np.sqrt(v)


TESTES_CONTINUA = {"auto": "Automático (Shapiro–Wilk → t de Student ou Mann–Whitney)", "t": "t de Student",
                   "welch": "t de Welch (variâncias diferentes)", "mw": "Mann–Whitney U"}
TESTES_CATEGORICA = {"auto": "Automático (qui-quadrado; Fisher se esperada < 5)", "chi2": "Qui-quadrado de Pearson",
                     "fisher": "Fisher exato"}


def comparar_continua(a, b, alpha_normal=0.05, forced: str | None = None):
    """Compara duas amostras independentes. `forced`: None/'auto' (regra Shapiro–Wilk), 't', 'welch' ou 'mw'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None, "—"
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0, "—"
    if forced in (None, "auto"):
        normal = all(len(x) >= 3 and np.ptp(x) > 0 and st.shapiro(x).pvalue > alpha_normal for x in (a, b))
        forced = "t" if normal else "mw"
    if forced == "t":
        return st.ttest_ind(a, b, equal_var=True).pvalue, "t de Student"
    if forced == "welch":
        return st.ttest_ind(a, b, equal_var=False).pvalue, "t de Welch"
    return st.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=False).pvalue, "Mann–Whitney U"  # como SPSS/PSPP


def comparar_categorica(tab: np.ndarray, rng: np.random.Generator, forced: str | None = None):
    """Tabela de contingência grupos × níveis. `forced`: None/'auto', 'chi2' ou 'fisher'."""
    tab = tab[(tab.sum(1) > 0)]
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None, "—"
    chi2, p, dof, exp = st.chi2_contingency(tab, correction=False)
    use_fisher = (exp < 5).any() if forced in (None, "auto") else forced == "fisher"
    if use_fisher:
        if tab.shape == (2, 2):
            return st.fisher_exact(tab).pvalue, "Fisher exato"
        # r×c: o SciPy estima o p de Fisher por reamostragem; semente fixa para o resultado ser reprodutível
        try:
            meth = st.MonteCarloMethod(n_resamples=FISHER_RXC_N, rng=np.random.default_rng(FISHER_RXC_SEED))
            return st.fisher_exact(tab, method=meth).pvalue, "Fisher (Monte Carlo)"
        except (TypeError, AttributeError, ValueError):  # SciPy antigo: sem Fisher r×c
            return _perm_chi2(tab, chi2, np.random.default_rng(FISHER_RXC_SEED)), "Qui-quadrado (permutação)"
    return p, "Qui-quadrado"


FISHER_RXC_N = 9999
FISHER_RXC_SEED = 20250101


def _perm_chi2(tab, chi2_obs, rng, n=10000):
    rows = np.repeat(np.arange(tab.shape[0]), tab.sum(1))
    cols = np.repeat(np.arange(tab.shape[1]), tab.sum(0))
    hits = 0
    for _ in range(n):
        t = np.zeros_like(tab)
        np.add.at(t, (rows, rng.permutation(cols)), 1)
        hits += st.chi2_contingency(t, correction=False)[0] >= chi2_obs - 1e-12
    return (hits + 1) / (n + 1)


PAPEIS = {"confundidor": "Confundidor (equilibrar)", "preditor": "Preditor candidato (não emparelhar)",
          "interferente": "Interferente analítico (monitorizar)", "descritiva": "Descritiva"}


def papel(v: dict) -> str:
    """Papel da variável na tabela de comparação. Só os confundidores contam para o reequilíbrio."""
    if not v.get("testar", True):
        return PAPEIS["descritiva"]
    return PAPEIS.get(v.get("papel", "confundidor"), v.get("papel", "confundidor"))


def table1(a: pd.DataFrame, b: pd.DataFrame, variables: list[dict], alpha: float, rng, names=("Casos", "Controlos"),
           smd_lim: float = 0.1) -> pd.DataFrame:
    out = []
    for v in variables:
        col, label, kind = v["col"], v.get("nome", v["col"]), v.get("tipo", "continua")
        test = v.get("testar", True)
        if col not in a.columns:
            out.append({"Variável": label, "Nível": "", names[0]: "coluna em falta", names[1]: "", "Teste": "", "p": None, "SMD": None,
                        "Papel": papel(v)})
            continue
        xa, xb = a[col], b[col]
        miss = f"{int(xa.isna().sum())}/{int(xb.isna().sum())}"
        if kind == "continua":
            xa, xb = pd.to_numeric(xa, errors="coerce"), pd.to_numeric(xb, errors="coerce")
            f = lambda s: "—" if s.notna().sum() == 0 else f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"
            p, tn = comparar_continua(xa, xb, forced=v.get("teste")) if test else (None, "descritiva")
            if v.get("log"):  # SMD em escala log para variáveis assimétricas (ex.: troponina, D-dímero)
                la, lb = np.log(xa.where(xa > 0)), np.log(xb.where(xb > 0))
                smd = smd_cont(la, lb)
            else:
                smd = smd_cont(xa, xb)
            out.append({"Variável": label, "Nível": "mediana (P25–P75)", names[0]: f(xa), names[1]: f(xb), "Teste": tn,
                        "p": _fmt_p(p), "SMD": round(float(smd), 3), "Em falta (casos/controlos)": miss, "Papel": papel(v)})
        else:
            labs = v.get("rotulos") or {}
            xa_, xb_ = xa.dropna(), xb.dropna()
            levels = sorted(set(xa_.unique()) | set(xb_.unique()), key=lambda z: (str(type(z)), z))
            tab = np.array([[int((xa_ == l).sum()) for l in levels], [int((xb_ == l).sum()) for l in levels]])
            p, tn = comparar_categorica(tab, rng, forced=v.get("teste")) if test else (None, "descritiva")
            smds = [smd_prop(tab[0, i] / max(tab[0].sum(), 1), tab[1, i] / max(tab[1].sum(), 1)) for i in range(len(levels))]
            smd = max(smds, key=abs) if len(levels) > 2 else (smds[-1] if smds else np.nan)
            show = levels if len(levels) > 2 or v.get("mostrar_todos") else levels[-1:]
            for j, l in enumerate(show):
                i = levels.index(l)
                pa = 100 * tab[0, i] / max(tab[0].sum(), 1)
                pb = 100 * tab[1, i] / max(tab[1].sum(), 1)
                out.append({"Variável": label if j == 0 else "", "Nível": str(labs.get(l, labs.get(str(l), l))),
                            names[0]: f"{tab[0, i]} ({pa:.0f}%)", names[1]: f"{tab[1, i]} ({pb:.0f}%)",
                            "Teste": tn if j == 0 else "", "p": _fmt_p(p) if j == 0 else None,
                            "SMD": round(float(smd), 3) if j == 0 else None, "Em falta (casos/controlos)": miss if j == 0 else "",
                            "Papel": papel(v) if j == 0 else ""})
    t = pd.DataFrame(out)
    t["Diferença significativa"] = np.where(t.p.notna() & (t.p < alpha), "Sim", np.where(t.p.notna(), "Não", ""))
    tested = t["Teste"].ne("descritiva")
    t["Desequilíbrio (|SMD|>%.2g)" % smd_lim] = np.where(tested & t.SMD.notna() & (t.SMD.abs() > smd_lim), "Sim",
                                                       np.where(tested & t.SMD.notna(), "Não", ""))
    return t



# ---------------------------------------------------------------- exploração (passo pedagógico)
def inferir_tipo(s: pd.Series) -> tuple[str, str]:
    """Classifica uma coluna: devolve (tipo para a tabela, descrição pedagógica).

    tipo ∈ {'continua', 'categorica'}; a descrição explica a classificação e alerta para casos ambíguos
    (ex.: códigos numéricos que podem ser nominais, escalas ordinais com poucos níveis).
    """
    x = s.dropna()
    if x.empty:
        return "categorica", "Sem dados."
    if x.dtype == bool or str(x.dtype) == "boolean":
        return "categorica", "Binária (verdadeiro/falso)."
    num = pd.to_numeric(x, errors="coerce")
    if num.notna().mean() < 0.95:
        k = x.nunique()
        return "categorica", f"Categórica nominal (texto, {k} categorias)."
    k = num.nunique()
    integer = np.allclose(num, np.round(num))
    if k <= 2:
        return "categorica", "Binária (2 valores)."
    if integer and k <= 10:
        return "categorica", (f"Discreta com {k} valores inteiros. Se forem códigos (ex.: 1 = …, 2 = …) é nominal → categórica. "
                              "Se for uma escala ordenada ou contagem (ex.: SOFA, n.º de episódios), pode tratar-se como "
                              "contínua com teste não paramétrico (Mann–Whitney).")
    if integer:
        return "continua", f"Discreta com muitos valores ({k}): pode tratar-se como contínua."
    return "continua", "Contínua."


def explorar(a: pd.Series, b: pd.Series, v: dict, alpha_normal: float = 0.05) -> dict:
    """Diagnóstico de uma variável para a escolha do teste, com justificação em linguagem simples."""
    kind = v.get("tipo", "continua")
    out = {"Coluna": v["col"], "Variável": v.get("nome", v["col"]), "Tipo": kind,
           "n (casos/controlos)": f"{int(a.notna().sum())}/{int(b.notna().sum())}",
           "Em falta": f"{int(a.isna().sum())}/{int(b.isna().sum())}"}
    if kind == "continua":
        xa, xb = (pd.to_numeric(x, errors="coerce").dropna() for x in (a, b))
        sw = []
        for x in (xa, xb):
            sw.append(st.shapiro(x).pvalue if len(x) >= 3 and np.ptp(x) > 0 else np.nan)
        sk = [float(st.skew(x)) if len(x) >= 3 else np.nan for x in (xa, xb)]
        lev = st.levene(xa, xb, center="median").pvalue if len(xa) >= 2 and len(xb) >= 2 and (np.ptp(xa) > 0 or np.ptp(xb) > 0) else np.nan
        normal = all(np.isfinite(p) and p > alpha_normal for p in sw)
        out.update({"Shapiro–Wilk p (casos)": sw[0], "Shapiro–Wilk p (controlos)": sw[1],
                    "Assimetria (casos)": sk[0], "Assimetria (controlos)": sk[1], "Levene p": lev,
                    "Normal nos dois grupos?": "Sim" if normal else "Não"})
        n_min = min(len(xa), len(xb))
        if normal:
            rec = "t" if not (np.isfinite(lev) and lev < 0.05) else "welch"
            why = ("Distribuição compatível com a normal nos dois grupos (Shapiro–Wilk p > 0,05) → teste paramétrico. "
                   + ("Variâncias semelhantes (Levene p ≥ 0,05) → t de Student." if rec == "t" else
                      "Variâncias diferentes (Levene p < 0,05) → t de Welch é mais seguro (a regra automática usa Student)."))
        else:
            rec = "mw"
            which = " e ".join(g for g, p in zip(("casos", "controlos"), sw) if not (np.isfinite(p) and p > alpha_normal))
            why = f"Afasta-se da normal em {which} (Shapiro–Wilk p ≤ 0,05) → teste não paramétrico: Mann–Whitney U (compara distribuições/medianas)."
        if n_min < 15:
            why += " Atenção: com n < 15 por grupo o Shapiro–Wilk tem pouca potência — ver também o histograma e o gráfico Q–Q."
        elif n_min > 200:
            why += " Com n > 200 por grupo, o Shapiro–Wilk deteta desvios irrelevantes; o t é robusto a desvios moderados."
        out.update({"Teste recomendado": TESTES_CONTINUA[rec], "_rec": rec, "Justificação": why})
    else:
        xa, xb = a.dropna(), b.dropna()
        levels = sorted(set(xa.unique()) | set(xb.unique()), key=lambda z: (str(type(z)), z))
        tab = np.array([[int((xa == l).sum()) for l in levels], [int((xb == l).sum()) for l in levels]])
        tab = tab[:, tab.sum(0) > 0]
        if tab.shape[1] < 2:
            out.update({"Frequência esperada mínima": np.nan, "Teste recomendado": "— (só um nível)", "_rec": "auto",
                        "Justificação": "Todos os indivíduos têm o mesmo valor: não há nada a comparar."})
            return out
        exp = st.chi2_contingency(tab, correction=False)[3]
        emin = float(exp.min())
        n_low = int((exp < 5).sum())
        out.update({"Níveis": int(tab.shape[1]), "Frequência esperada mínima": emin,
                    "Células com esperada < 5": f"{n_low}/{exp.size}"})
        if n_low:
            rec = "fisher"
            why = (f"{n_low} de {exp.size} células com frequência esperada < 5 → a aproximação do qui-quadrado não é fiável; "
                   "usa-se o teste exato de Fisher" + (" (estimado por Monte Carlo, por ter mais de 2 níveis)." if tab.shape[1] > 2 else "."))
        else:
            rec = "chi2"
            why = "Todas as frequências esperadas ≥ 5 → qui-quadrado de Pearson (sem correção de Yates)."
        out.update({"Teste recomendado": TESTES_CATEGORICA[rec], "_rec": rec, "Justificação": why})
    return out
