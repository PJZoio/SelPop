"""Gera uma base de exemplo com dados FICTÍCIOS (uma linha por participante), para demonstração e ensino.

    python scripts/gerar_exemplo.py            → exemplos/exemplo_ficticio.csv

Os casos são, de propósito, mais velhos e com mais diabetes/hipertensão do que os controlos, para mostrar o
efeito do emparelhamento. Nenhum valor corresponde a pessoas reais.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 400
caso = rng.random(N) < 0.25
idade = np.where(caso, rng.normal(66, 9, N), rng.normal(56, 12, N)).clip(18, 95).round()
sexo = np.where(rng.random(N) < np.where(caso, 0.38, 0.52), "F", "M")
imc = np.exp(rng.normal(np.log(np.where(caso, 28.5, 26.5)), 0.15, N)).round(1)
diabetes = (rng.random(N) < np.where(caso, 0.45, 0.18)).astype(int)
hta = (rng.random(N) < np.where(caso, 0.70, 0.40)).astype(int)
fumador = rng.choice(["Nunca", "Ex-fumador", "Atual"], N, p=[0.5, 0.3, 0.2])
centro = rng.choice(["A", "B", "C"], N, p=[0.5, 0.35, 0.15])
creat = np.exp(rng.normal(np.log(np.where(caso, 1.6, 0.9)), np.where(caso, 0.45, 0.2), N)).round(2)
ureia = (creat * 22 + rng.normal(0, 6, N)).clip(8).round()
albumina = rng.normal(np.where(caso, 3.7, 4.1), 0.4, N).round(1)
pcr = np.exp(rng.normal(np.log(np.where(caso, 0.9, 0.4)), 0.9, N)).round(2)
datas = pd.Timestamp("2024-01-08") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
df = pd.DataFrame({
    "id": [f"P{i + 1:04d}" for i in range(N)], "amostra": [f"S{i + 1:05d}" for i in range(N)],
    "data_colheita": datas.strftime("%Y-%m-%d"), "grupo": np.where(caso, "Caso", "Controlo"),
    "idade": idade.astype(int), "sexo": sexo, "imc": imc, "fumador": fumador, "diabetes": diabetes, "hta": hta,
    "centro": centro, "creatinina": creat, "ureia": ureia, "albumina": albumina, "pcr": pcr,
})
for col, frac in (("imc", 0.04), ("albumina", 0.03), ("pcr", 0.06)):   # alguns valores em falta, como na vida real
    df.loc[rng.random(N) < frac, col] = np.nan
out = Path(__file__).resolve().parents[1] / "exemplos" / "exemplo_ficticio.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"{out} — {len(df)} participantes ({int(caso.sum())} casos)")
