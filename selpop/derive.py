"""Variáveis derivadas, ao nível do dia e do doente.

Tudo o que aqui se calcula fica como coluna nova (prefixo implícito em português, minúsculas) e pode
ser usado nas expressões de elegibilidade e de definição de grupos do ficheiro de configuração.

Definições (documentadas também no relatório):
- TFGe: CKD-EPI 2021 sem coeficiente de raça (Inker et al., NEJM 2021).
- LRA: critério de creatinina KDIGO 2012 aplicado a valores diários:
    subida  — creatinina máx. do dia ≥ 1,5 × mínimo dos 7 dias anteriores (incluindo o próprio dia)
              ou ≥ 0,3 mg/dL acima do mínimo das 48 h anteriores (dia, dia-1, dia-2);
    presente na admissão — nos primeiros `dias_admissao` dias, creatinina máx. ≥ 1,5 × mínimo dos 7 dias
              seguintes (nadir posterior como basal retrospetivo).
  Estádio no dia de critério (rácio face ao basal de 7 dias): 1 = 1,5–1,9×; 2 = 2,0–2,9×; 3 = ≥3,0× ou ≥4,0 mg/dL.
  Sem débito urinário nem creatinina pré-admissão: é uma aproximação, não um diagnóstico KDIGO completo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import to_num

D = lambda n: np.timedelta64(int(n), "D")


def ckdepi2021(scr, age, female):
    scr, age, female = np.asarray(scr, float), np.asarray(age, float), np.asarray(female, bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    return 142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)


def in_period(day, start, end, flag=None):
    """Verdadeiro se `day` está entre início e fim (fim em falta = período aberto)."""
    end = end.fillna(pd.Timestamp("2100-01-01"))
    r = start.notna() & (day >= start) & (day <= end)
    if flag is not None:
        r &= flag == 1
    return r


def _kdigo_patient(t, hi, lo, dias_admissao, detail=None):
    """Critério de creatinina KDIGO por dia. Se `detail` for um dicionário, recebe os valores intermédios
    (para rastreabilidade): mínimo dos 7 dias anteriores, razão, mínimo das 48 h, subida em 48 h, e o teste de
    LRA na admissão (máximo do dia vs. nadir dos 7 dias seguintes)."""
    n = len(t)
    crit = np.zeros(n, bool)
    stage = np.zeros(n, int)
    B7, R7, B2, D2 = (np.full(n, np.nan) for _ in range(4))
    AN, AR = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if np.isnan(hi[i]):
            continue
        m7 = (t >= t[i] - D(7)) & (t <= t[i])
        m2 = (t >= t[i] - D(2)) & (t <= t[i])
        b7 = np.nanmin(lo[m7]) if np.isfinite(lo[m7]).any() else np.nan
        b2 = np.nanmin(lo[m2]) if np.isfinite(lo[m2]).any() else np.nan
        r = hi[i] / b7 if b7 and np.isfinite(b7) else np.nan
        B7[i], R7[i], B2[i], D2[i] = b7, r, b2, hi[i] - b2 if np.isfinite(b2) else np.nan
        c = (np.isfinite(r) and r >= 1.5) or (np.isfinite(b2) and hi[i] - b2 >= 0.3)
        if c:
            crit[i] = True
            stage[i] = 3 if (np.isfinite(r) and r >= 3) or hi[i] >= 4.0 else 2 if (np.isfinite(r) and r >= 2) else 1
    # presente na admissão (nadir posterior)
    adm = False
    if n:
        t0 = t[0]
        for i in range(n):
            if t[i] > t0 + D(dias_admissao - 1) or np.isnan(hi[i]):
                continue
            f7 = lo[(t >= t[i]) & (t <= t[i] + D(7))]
            f7 = f7[np.isfinite(f7)]
            if len(f7):
                AN[i], AR[i] = f7.min(), hi[i] / f7.min() if f7.min() else np.nan
            if len(f7) and hi[i] >= 1.5 * f7.min():
                adm = True
    if detail is not None:
        detail.update(cr_min_7d_antes=B7, razao_7d=R7, cr_min_48h_antes=B2, subida_48h=D2,
                      adm_nadir_7d_seguintes=AN, adm_razao=AR)
    return crit, stage, adm


def _has(df, c, *keys) -> bool:
    """Todas as chaves estão mapeadas na configuração e existem na base."""
    return all(c.get(k) and c.get(k) in df.columns for k in keys)


def derive(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula as variáveis derivadas disponíveis com as colunas mapeadas em `colunas`.

    Só `dados.id` é obrigatório. Cada bloco (amostra, idade/sexo, internamento, VMI/ECMO, SOFA, função renal
    e LRA) só é calculado se as colunas de que precisa estiverem mapeadas; os restantes são ignorados.
    """
    c = dict(cfg.get("colunas") or {})
    # uma só coluna de creatinina/ureia serve como máximo e mínimo do dia
    for base in ("creatinina", "ureia"):
        if c.get(base) and not c.get(f"{base}_max"):
            c[f"{base}_max"] = c[f"{base}_min"] = c[base]
    for alias, col in (cfg.get("aliases") or {}).items():
        if col in df.columns:
            df[alias] = to_num(df[col])
    idc = cfg["dados"]["id"]
    dt = c.get("data_colheita") if _has(df, c, "data_colheita") else None
    for k in ("data_admissao", "data_alta", "vmi_inicio", "vmi_fim", "ecmo_inicio", "ecmo_fim", "data_colheita"):
        if _has(df, c, k):
            df[c[k]] = pd.to_datetime(df[c[k]], errors="coerce")
    df["_ordem_original"] = np.arange(len(df))
    df = df.sort_values([idc, dt, "_ordem_original"] if dt else [idc, "_ordem_original"]).reset_index(drop=True).copy()

    # --- amostra e quantidade de dados
    df["amostra"] = df[c["n_amostra"]].notna() if _has(df, c, "n_amostra") else True
    g = df.groupby(idc)
    df["dias_com_dados"] = g[idc].transform("size")
    df["n_amostras_doente"] = g["amostra"].transform("sum")
    fem = (df[c["sexo"]] == c.get("sexo_feminino", 1)) if _has(df, c, "sexo") else None
    age = to_num(df[c["idade"]]) if _has(df, c, "idade") else None

    # --- internamento
    if dt and _has(df, c, "data_admissao"):
        df["dias_desde_admissao"] = (df[dt] - df[c["data_admissao"]]).dt.days
    if _has(df, c, "data_alta", "data_admissao"):
        df["duracao_uci_dias"] = (df[c["data_alta"]] - df[c["data_admissao"]]).dt.days
    if dt and _has(df, c, "vmi_inicio", "vmi_fim"):
        df["em_vmi_dia"] = in_period(df[dt], df[c["vmi_inicio"]], df[c["vmi_fim"]], df[c["vmi"]] if _has(df, c, "vmi") else None)
        df["dias_vmi_ate_amostra"] = np.where(df.em_vmi_dia, (df[dt] - df[c["vmi_inicio"]]).dt.days, 0)
    if dt and _has(df, c, "ecmo_inicio", "ecmo_fim"):
        df["em_ecmo_dia"] = in_period(df[dt], df[c["ecmo_inicio"]], df[c["ecmo_fim"]], df[c["ecmo"]] if _has(df, c, "ecmo") else None)

    # --- SOFA
    if _has(df, c, "sofa_total"):
        sofa_t = to_num(df[c["sofa_total"]])
        df["sofa_total_dia"] = sofa_t
        if _has(df, c, "sofa_renal"):
            sofa_r = to_num(df[c["sofa_renal"]])
            df["sofa_nao_renal_dia"] = sofa_t - sofa_r
            if _has(df, c, "sofa_cardiovascular"):
                sofa_cv = to_num(df[c["sofa_cardiovascular"]])
                df["sofa_cv_dia"] = sofa_cv
                df["sofa_nao_renal_nao_cv_dia"] = sofa_t - sofa_r - sofa_cv

    # --- função renal
    if _has(df, c, "creatinina_max", "creatinina_min"):
        crmax, crmin = to_num(df[c["creatinina_max"]]), to_num(df[c["creatinina_min"]])
        df["cr_unico"] = crmax.notna() & (crmax == crmin)
        df["cr_dia"] = crmax.where(df.cr_unico)
        df["cr_max_dia"], df["cr_min_dia"] = crmax, crmin
        df["cr_min_internamento"] = df.groupby(idc)["cr_min_dia"].transform("min")
        if age is not None and fem is not None:
            df["tfge_dia"] = ckdepi2021(crmin, age, fem)
            df["tfge_basal_proxy"] = ckdepi2021(df.cr_min_internamento, age, fem)
        if dt and (cfg.get("lra") or {}).get("calcular", True):
            df = _lra_block(df, cfg, idc, dt, crmax, crmin)
    if _has(df, c, "ureia_max", "ureia_min"):
        urmax, urmin = to_num(df[c["ureia_max"]]), to_num(df[c["ureia_min"]])
        df["ur_unico"] = urmax.notna() & (urmax == urmin)
        df["ur_dia"] = urmax.where(df.ur_unico)
    return df.drop(columns="_ordem_original")


def _lra_block(df, cfg, idc, dt, crmax, crmin):
    """LRA (critério de creatinina KDIGO) e janelas temporais — requer datas e creatinina diária."""
    crmax, crmin = crmax.values, crmin.values
    df["lra_dia"] = False
    df["lra_estadio_dia"] = 0
    lra_cfg = cfg.get("lra", {}) or {}
    k = lra_cfg.get("janela_estavel_dias", 2)
    pat = {}
    for pid, idx in df.groupby(idc).indices.items():
        g = df.iloc[idx]
        t = g[dt].values
        det = {}
        crit, stage, adm = _kdigo_patient(t, crmax[idx], crmin[idx], lra_cfg.get("dias_admissao", 2), detail=det)
        for kk, vv in det.items():
            df.loc[g.index, f"kdigo_{kk}"] = vv
        df.loc[g.index, "lra_dia"] = crit
        df.loc[g.index, "lra_estadio_dia"] = stage
        onset = t[crit.argmax()] if crit.any() else np.datetime64("NaT")
        has_cr = np.isfinite(crmax[idx])
        last_cr = t[has_cr][-1] if has_cr.any() else np.datetime64("NaT")
        quiet = np.array([not crit[(t >= t[i] - D(k)) & (t <= t[i] + D(k))].any() for i in range(len(t))])
        df.loc[g.index, "sem_lra_janela"] = quiet
        df.loc[g.index, "dias_obs_apos"] = (last_cr - t) / D(1) if not pd.isna(last_cr) else np.nan
        df.loc[g.index, "dias_ate_lra"] = (onset - t) / D(1) if not pd.isna(onset) else np.nan
        first = g.index[crit.argmax()] if crit.any() else None
        pat[pid] = {"lra_subida": bool(crit.any()), "lra_admissao": bool(adm), "lra_inicio": onset,
                    "lra_estadio_max": int(stage.max()) if len(stage) else 0,
                    "lra_inicio_dias_admissao": float(df.loc[first, "dias_desde_admissao"]) if first is not None and "dias_desde_admissao" in df else np.nan}
    df["sem_lra_janela"] = df["sem_lra_janela"].astype(bool)
    P = pd.DataFrame.from_dict(pat, orient="index")
    P["lra_qualquer"] = P.lra_subida | P.lra_admissao
    return df.join(P, on=idc).copy()


def quality_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Verificações de coerência, ao nível do indivíduo, com as colunas disponíveis. Não altera os dados: só reporta."""
    c, idc = dict(cfg.get("colunas") or {}), cfg["dados"]["id"]
    g = df.groupby(idc)
    checks = []
    first = lambda k: g[c[k]].first()
    if _has(df, c, "data_colheita", "data_admissao"):
        checks.append(("Amostra/registo com data anterior à admissão", (g[c["data_colheita"]].min() - first("data_admissao")).dt.days < 0))
    if _has(df, c, "vmi_inicio", "data_admissao"):
        d = (first("vmi_inicio") - first("data_admissao")).dt.days
        checks += [("Início de VMI >1 dia antes da admissão", d < -1), ("Início de VMI >60 dias após a admissão", d > 60)]
    if _has(df, c, "vmi_inicio", "vmi_fim"):
        checks.append(("Fim de VMI anterior ao início", (first("vmi_fim") - first("vmi_inicio")).dt.days < 0))
    if _has(df, c, "ecmo_inicio", "ecmo_fim"):
        checks.append(("Fim de ECMO anterior ao início", (first("ecmo_fim") - first("ecmo_inicio")).dt.days < 0))
    if _has(df, c, "idade"):
        a = to_num(first("idade"))
        checks.append(("Idade fora de 18–105", (a < 18) | (a > 105)))
    if "cr_max_dia" in df:
        checks.append(("Creatinina fora de 0,1–20 mg/dL", (g["cr_min_dia"].min() < 0.1) | (g["cr_max_dia"].max() > 20)))
    if _has(df, c, "data_colheita") and _has(df, c, "n_amostra"):
        dup = df[df[c["n_amostra"]].notna()].duplicated(c["n_amostra"], keep=False)
        ids = df[df[c["n_amostra"]].notna()][dup][idc].unique()
        checks.append(("Nº de amostra repetido", pd.Series(True, index=ids) if len(ids) else pd.Series(dtype=bool)))
    checks.append(("Linhas sem ID", pd.Series(True, index=["(linhas)"]) if df[idc].isna().any() else pd.Series(dtype=bool)))
    rows = []
    for name, m in checks:
        m = m.fillna(False).astype(bool)
        ids = m.index[m].tolist()
        rows.append({"Verificação": name, "N.º doentes": len(ids), "IDs (até 40)": ", ".join(map(str, ids[:40]))})
    return pd.DataFrame(rows)
