"""Testes com dados sintéticos (não usam a base real). Correr: python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest

from selpop.derive import _kdigo_patient, ckdepi2021
from selpop.match import match
from selpop.stats import smd_prop, table1, comparar_categorica, comparar_continua

T = lambda days: np.array([np.datetime64("2021-01-01") + np.timedelta64(d, "D") for d in days])


def test_ckdepi2021_reference_values():
    # valores de referência calculados à mão com a equação CKD-EPI 2021
    assert ckdepi2021([1.0], [50], [False])[0] == pytest.approx(91.7, abs=0.2)
    assert ckdepi2021([1.0], [50], [True])[0] == pytest.approx(68.8, abs=0.2)


def test_kdigo_rise_by_ratio_and_stage():
    t = T(range(5))
    hi = np.array([1.0, 1.0, 1.2, 2.1, 3.2]); lo = hi.copy()
    crit, stage, adm = _kdigo_patient(t, hi, lo, 2)
    assert crit.tolist() == [False, False, False, True, True]
    assert stage[3] == 2 and stage[4] == 3 and not adm


def test_kdigo_rise_by_03_in_48h():
    t = T(range(3)); hi = np.array([0.8, 0.9, 1.1]); lo = hi.copy()
    crit, _, _ = _kdigo_patient(t, hi, lo, 2)
    assert crit[2] and not crit[1]


def test_kdigo_present_at_admission():
    t = T(range(5)); hi = np.array([3.0, 2.5, 1.8, 1.2, 1.0]); lo = hi.copy()
    crit, _, adm = _kdigo_patient(t, hi, lo, 2)
    assert adm and not crit.any()


def test_kdigo_7day_window_excludes_old_nadir():
    t = T([0, 10]); hi = np.array([0.8, 1.3]); lo = hi.copy()   # subida de 0,5 mas 10 dias depois e <1,5×
    crit, _, _ = _kdigo_patient(t, hi, lo, 2)
    assert not crit.any()


def test_tests_choose_expected_methods():
    rng = np.random.default_rng(0)
    p, name = comparar_continua(rng.normal(0, 1, 40), rng.normal(0, 1, 40))
    assert name == "t de Student" and p > 0.01
    p, name = comparar_continua(rng.exponential(1, 40), rng.exponential(3, 40))
    assert name == "Mann–Whitney U" and p < 0.05
    p, name = comparar_categorica(np.array([[2, 8], [9, 1]]), rng)
    assert name == "Fisher exato"
    p, name = comparar_categorica(np.array([[30, 30], [32, 28]]), rng)
    assert name == "Qui-quadrado"


def test_smd_prop_sign():
    assert smd_prop(0.5, 0.2) > 0 and smd_prop(0.2, 0.5) < 0 and smd_prop(0.3, 0.3) == 0


def _toy():
    cases = pd.DataFrame({"id": [1, 2], "SEXO": [0, 1], "IDADE": [60, 70], "dia": [3, 5]})
    ctrl = pd.DataFrame({"id": [10, 10, 11, 12, 13, 14], "SEXO": [0, 0, 0, 1, 1, 1],
                         "IDADE": [61, 61, 75, 69, 90, 71], "dia": [3, 8, 3, 5, 5, 6]})
    return cases, ctrl


def test_match_respects_exact_caliper_and_uniqueness():
    cases, ctrl = _toy()
    spec = {"razao": 2, "etapas": [{"exato": ["SEXO"], "calibre": {"IDADE": 5, "dia": 2}}]}
    m = match(cases, ctrl, spec, "id", np.random.default_rng(1))
    assert m.controlo_id.is_unique                       # cada doente-controlo usado uma vez
    c1 = m[m.caso_id == 1]
    assert set(c1.controlo_id) == {10}                   # 11 falha o calibre de idade
    assert ctrl.loc[c1._row.iloc[0], "dia"] == 3         # escolhe o dia do controlo mais próximo
    assert set(m[m.caso_id == 2].controlo_id) == {12, 14}  # 13 falha o calibre de idade


def test_match_relaxed_stage_only_for_pending():
    cases, ctrl = _toy()
    spec = {"razao": 1, "etapas": [{"exato": ["SEXO"], "calibre": {"IDADE": 0.5}},
                                   {"exato": ["SEXO"], "calibre": {"IDADE": 2}}]}
    m = match(cases, ctrl, spec, "id", np.random.default_rng(1))
    assert m.set_index("caso_id").etapa.to_dict() == {1: 2, 2: 2}


def test_table1_flags_difference():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(10, 1, 50), "s": [1] * 40 + [0] * 10})
    b = pd.DataFrame({"x": rng.normal(13, 1, 50), "s": [1] * 25 + [0] * 25})
    t = table1(a, b, [{"col": "x", "tipo": "continua"}, {"col": "s", "tipo": "categorica"}], 0.05, rng)
    assert (t["Diferença significativa"] == "Sim").sum() == 2


def test_fisher_rxc_is_reproducible():
    t = np.array([[10, 1, 0, 4, 1, 1], [22, 6, 1, 0, 0, 4]])
    ps = {comparar_categorica(t, np.random.default_rng(i))[0] for i in range(3)}
    assert len(ps) == 1
