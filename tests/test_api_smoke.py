"""Calls every API endpoint through FastAPI's test client (loads the models; ~1 minute)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.slow
def test_api_endpoints():
    with TestClient(app) as c:
        assert c.get("/health").json()["schemes"] == 4693
        r = c.post("/recommend", json={"profile": {"state": "Gujarat", "age": 30},
                                       "need": "loan to start a small business", "k": 3}).json()
        assert len(r["results"]) == 3 and r["results"][0]["status"] in ("eligible", "possibly eligible")
        a = c.post("/ask", json={"question": "Who can apply for PM Vishwakarma?"}).json()
        assert a["intent"] == "eligibility" and "artisan" in a["answer"].lower()
        sid = r["results"][0]["id"]
        assert c.get(f"/scheme/{sid}").json()["id"] == sid
        assert c.get("/scheme/does-not-exist").status_code == 404
        assert c.get("/stats").json()["n_schemes"] == 4693
