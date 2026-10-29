"""Runs the real Streamlit app headless and uses all three tabs.

Slow (loads the models): run with  python -m pytest -q tests/test_app_smoke.py
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py")


@pytest.mark.slow
def test_app_find_ask_insights():
    at = AppTest.from_file(APP, default_timeout=600).run()
    assert not at.exception, at.exception

    # Find schemes: Hindi need, no profile
    at.text_area[0].input("मैं उत्तर प्रदेश की 62 साल की विधवा हूँ, पेंशन चाहिए")
    at.button[0].click().run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Pension" in page or "pension" in page
    assert any(m.label == "Eligible" for m in at.metric)

    # Ask tab
    at.chat_input[0].set_value("What documents are needed for PM Kisan?").run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Aadhaar" in page and "Source" in page

    # Insights tab rendered its metrics
    assert any(m.label == "Schemes" for m in at.metric)
