"""Tabela 1 e testes de equilíbrio entre dois grupos.

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


def comparar_continua(a, b, alpha_normal=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None, "—"
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0, "—"
    normal = all(len(x) >= 3 and np.ptp(x) > 0 and st.shapiro(x).pvalue > alpha_normal for x in (a, b))
    if normal:
        return st.ttest_ind(a, b, equal_var=True).pvalue, "t de Student"
    return st.mannwhitneyu(a, b, alternative="two-sided").pvalue, "Mann–Whitney U"


def comparar_categorica(tab: np.ndarray, rng: np.random.Generator):
    tab = tab[(tab.sum(1) > 0)]
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None, "—"
    chi2, p, dof, exp = st.chi2_contingency(tab, correction=False)
    if (exp < 5).any():
        try:
            return st.fisher_exact(tab).pvalue, "Fisher exato"
        except Exception:  # SciPy antigo: Fisher só 2x2 → p por permutação do qui-quadrado
            return _perm_chi2(tab, chi2, rng), "Qui-quadrado (permutação)"
    return p, "Qui-quadrado"


def _perm_chi2(tab, chi2_obs, rng, n=10000):
    rows = np.repeat(np.arange(tab.shape[0]), tab.sum(1))
    cols = np.repeat(np.arange(tab.shape[1]), tab.sum(0))
    hits = 0
    for _ in range(n):
        t = np.zeros_like(tab)
        np.add.at(t, (rows, rng.permutation(cols)), 1)
        hits += st.chi2_contingency(t, correction=False)[0] >= chi2_obs - 1e-12
    return (hits + 1) / (n + 1)


def table1(a: pd.DataFrame, b: pd.DataFrame, variables: list[dict], alpha: float, rng, names=("Casos", "Controlos"),
           smd_lim: float = 0.1) -> pd.DataFrame:
    out = []
    for v in variables:
        col, label, kind = v["col"], v.get("nome", v["col"]), v.get("tipo", "continua")
        test = v.get("testar", True)
        if col not in a.columns:
            out.append({"Variável": label, "Nível": "", names[0]: "coluna em falta", names[1]: "", "Teste": "", "p": None, "SMD": None})
            continue
        xa, xb = a[col], b[col]
        miss = f"{int(xa.isna().sum())}/{int(xb.isna().sum())}"
        if kind == "continua":
            xa, xb = pd.to_numeric(xa, errors="coerce"), pd.to_numeric(xb, errors="coerce")
            f = lambda s: "—" if s.notna().sum() == 0 else f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"
            p, tn = comparar_continua(xa, xb) if test else (None, "descritiva")
            out.append({"Variável": label, "Nível": "mediana (P25–P75)", names[0]: f(xa), names[1]: f(xb), "Teste": tn,
                        "p": _fmt_p(p), "SMD": round(float(smd_cont(xa, xb)), 3), "Em falta (casos/controlos)": miss})
        else:
            labs = v.get("rotulos") or {}
            xa_, xb_ = xa.dropna(), xb.dropna()
            levels = sorted(set(xa_.unique()) | set(xb_.unique()), key=lambda z: (str(type(z)), z))
            tab = np.array([[int((xa_ == l).sum()) for l in levels], [int((xb_ == l).sum()) for l in levels]])
            p, tn = comparar_categorica(tab, rng) if test else (None, "descritiva")
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
                            "SMD": round(float(smd), 3) if j == 0 else None, "Em falta (casos/controlos)": miss if j == 0 else ""})
    t = pd.DataFrame(out)
    t["Diferença significativa"] = np.where(t.p.notna() & (t.p < alpha), "Sim", np.where(t.p.notna(), "Não", ""))
    tested = t["Teste"].ne("descritiva")
    t["Desequilíbrio (|SMD|>%.2g)" % smd_lim] = np.where(tested & t.SMD.notna() & (t.SMD.abs() > smd_lim), "Sim",
                                                       np.where(tested & t.SMD.notna(), "Não", ""))
    return t
