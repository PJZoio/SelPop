"""A app recupera quando o Streamlit mantém em memória módulos selpop antigos (situação após um merge no Streamlit Cloud)."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_survives_stale_selpop_modules():
    import selpop.stats as old
    saved = old.explorar
    try:
        del old.explorar                       # simula a versão antiga em memória
        at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
        at.run()
        assert not at.exception, [e.message for e in at.exception]
    finally:
        import sys
        if "selpop.stats" in sys.modules and not hasattr(sys.modules["selpop.stats"], "explorar"):
            sys.modules["selpop.stats"].explorar = saved
