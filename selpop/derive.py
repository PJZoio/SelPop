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


def _kdigo_patient(t, hi, lo, dias_admissao):
    n = len(t)
    crit = np.zeros(n, bool)
    stage = np.zeros(n, int)
    for i in range(n):
        if np.isnan(hi[i]):
            continue
        m7 = (t >= t[i] - D(7)) & (t <= t[i])
        m2 = (t >= t[i] - D(2)) & (t <= t[i])
        b7 = np.nanmin(lo[m7]) if np.isfinite(lo[m7]).any() else np.nan
        b2 = np.nanmin(lo[m2]) if np.isfinite(lo[m2]).any() else np.nan
        r = hi[i] / b7 if b7 and np.isfinite(b7) else np.nan
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
            if len(f7) and hi[i] >= 1.5 * f7.min():
                adm = True
    return crit, stage, adm


def derive(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["colunas"]
    idc, dt = cfg["dados"]["id"], c["data_colheita"]
    df = df.sort_values([idc, dt]).reset_index(drop=True).copy()
    for k in ("data_admissao", "vmi_inicio", "vmi_fim", "ecmo_inicio", "ecmo_fim", "data_colheita"):
        if c.get(k) in df.columns:
            df[c[k]] = pd.to_datetime(df[c[k]], errors="coerce")

    crmax, crmin = to_num(df[c["creatinina_max"]]), to_num(df[c["creatinina_min"]])
    urmax, urmin = to_num(df[c["ureia_max"]]), to_num(df[c["ureia_min"]])
    fem = df[c["sexo"]] == c["sexo_feminino"]

    # --- nível do dia
    df["amostra"] = df[c["n_amostra"]].notna()
    df["cr_unico"] = crmax.notna() & (crmax == crmin)
    df["ur_unico"] = urmax.notna() & (urmax == urmin)
    df["cr_dia"] = crmax.where(df.cr_unico)
    df["ur_dia"] = urmax.where(df.ur_unico)
    df["cr_max_dia"], df["cr_min_dia"] = crmax, crmin
    df["dias_desde_admissao"] = (df[dt] - df[c["data_admissao"]]).dt.days
    df["em_vmi_dia"] = in_period(df[dt], df[c["vmi_inicio"]], df[c["vmi_fim"]], df[c["vmi"]])
    df["em_ecmo_dia"] = in_period(df[dt], df[c["ecmo_inicio"]], df[c["ecmo_fim"]], df[c["ecmo"]])
    df["dias_vmi_ate_amostra"] = np.where(df.em_vmi_dia, (df[dt] - df[c["vmi_inicio"]]).dt.days, 0)
    sofa_t, sofa_r = to_num(df[c["sofa_total"]]), to_num(df[c["sofa_renal"]])
    df["sofa_total_dia"] = sofa_t
    df["sofa_nao_renal_dia"] = sofa_t - sofa_r
    df["tfge_dia"] = ckdepi2021(crmin, df[c["idade"]], fem)

    # --- LRA (KDIGO, critério de creatinina) e janelas temporais
    df["lra_dia"] = False
    df["lra_estadio_dia"] = 0
    pat = {}
    for pid, idx in df.groupby(idc).indices.items():
        g = df.iloc[idx]
        t = g[dt].values
        crit, stage, adm = _kdigo_patient(t, crmax.values[idx], crmin.values[idx], cfg.get("lra", {}).get("dias_admissao", 2))
        df.loc[g.index, "lra_dia"] = crit
        df.loc[g.index, "lra_estadio_dia"] = stage
        onset = t[crit.argmax()] if crit.any() else np.datetime64("NaT")
        has_cr = np.isfinite(crmax.values[idx])
        last_cr = t[has_cr][-1] if has_cr.any() else np.datetime64("NaT")
        # janela ±k dias sem critério de LRA
        k = cfg.get("lra", {}).get("janela_estavel_dias", 2)
        quiet = np.array([not crit[(t >= t[i] - D(k)) & (t <= t[i] + D(k))].any() for i in range(len(t))])
        df.loc[g.index, "sem_lra_janela"] = quiet
        df.loc[g.index, "dias_obs_apos"] = (last_cr - t) / D(1) if not pd.isna(last_cr) else np.nan
        df.loc[g.index, "dias_ate_lra"] = (onset - t) / D(1) if not pd.isna(onset) else np.nan
        pat[pid] = {"lra_subida": bool(crit.any()), "lra_admissao": bool(adm), "lra_inicio": onset,
                    "lra_estadio_max": int(stage.max()) if len(stage) else 0,
                    "lra_inicio_dias_admissao": float(df.loc[g.index[crit.argmax()], "dias_desde_admissao"]) if crit.any() else np.nan}
    df["sem_lra_janela"] = df["sem_lra_janela"].astype(bool)
    P = pd.DataFrame.from_dict(pat, orient="index")
    P["lra_qualquer"] = P.lra_subida | P.lra_admissao
    df = df.join(P, on=idc).copy()

    # --- nível do doente
    g = df.groupby(idc)
    df["cr_min_internamento"] = g["cr_min_dia"].transform("min")
    df["tfge_basal_proxy"] = ckdepi2021(df.cr_min_internamento, df[c["idade"]], fem)
    df["dias_com_dados"] = g[dt].transform("count")
    df["n_amostras_doente"] = g["amostra"].transform("sum")
    if c.get("data_alta") in df.columns:
        df["duracao_uci_dias"] = (pd.to_datetime(df[c["data_alta"]], errors="coerce") - df[c["data_admissao"]]).dt.days
    return df


def quality_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Verificações de coerência das datas e valores, ao nível do doente. Não altera os dados: só reporta."""
    c, idc = cfg["colunas"], cfg["dados"]["id"]
    p = df.groupby(idc).agg(primeira=(c["data_colheita"], "min"), adm=(c["data_admissao"], "first"),
                            vi=(c["vmi_inicio"], "first"), vf=(c["vmi_fim"], "first"),
                            ei=(c["ecmo_inicio"], "first"), ef=(c["ecmo_fim"], "first"),
                            idade=(c["idade"], "first"), crmax=("cr_max_dia", "max"), crmin=("cr_min_dia", "min"))
    checks = [
        ("Amostra/registo com data anterior à admissão", (p.primeira - p.adm).dt.days < 0),
        ("Início de VMI >1 dia antes da admissão na UCI", (p.vi - p.adm).dt.days < -1),
        ("Início de VMI >60 dias após a admissão", (p.vi - p.adm).dt.days > 60),
        ("Fim de VMI anterior ao início", (p.vf - p.vi).dt.days < 0),
        ("Fim de ECMO anterior ao início", (p.ef - p.ei).dt.days < 0),
        ("Idade fora de 18–105", (p.idade < 18) | (p.idade > 105)),
        ("Creatinina fora de 0,1–20 mg/dL", (p.crmin < 0.1) | (p.crmax > 20)),
    ]
    rows = []
    for name, m in checks:
        m = m.fillna(False)
        ids = p.index[m].tolist()
        rows.append({"Verificação": name, "N.º doentes": len(ids), "IDs (até 40)": ", ".join(map(str, ids[:40]))})
    return pd.DataFrame(rows)
