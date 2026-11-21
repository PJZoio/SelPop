"""A app corre uma seleção estratificada (dados fictícios) e mostra um estrato de cada vez."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _sel(at, label):
    return next(w for w in at.selectbox if w.label == label)


def test_app_stratified_run():
    at = st_testing.AppTest.from_file(APP, default_timeout=300)
    at.run()
    next(r for r in at.radio if r.label == "Dados").set_value("Exemplo (fictícios)").run()
    _sel(at, "Estratificar por").set_value("hta").run()
    next(b for b in at.button if b.key == "run6").click().run()
    assert not at.exception, [e.message for e in at.exception]
    est = next(r for r in at.radio if r.label.startswith("Estrato a mostrar"))
    assert len(est.options) == 2
    ss = at.session_state
    assert [e["valor"] for e, _ in ss["RS"]] == [0, 1]
    assert (ss["RS"][1][1]["sel_cases"].hta == 1).all()
    est.set_value(1).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Hipertensão arterial: Sim" in i.value for i in at.info)   # passo 5 indica o estrato mostrado
