"""Checks the OpenAI-compatible LLM call against a tiny fake server (no network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.recommend.rag import call_llm, detect_intent, detect_language


class FakeLLM(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.seen = {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        reply = {"choices": [{"message": {"content": " You need an Aadhaar card [1]. "}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_call_llm_roundtrip():
    server = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = {"base": f"http://127.0.0.1:{server.server_port}/v1", "model": "test-model", "key": "k123"}
        passages = [{"scheme": "PM Kisan", "section": "documents_required", "text": "Aadhaar card"}]
        out = call_llm(cfg, "Which documents?", passages, "hi")
    finally:
        server.shutdown()
    assert out == "You need an Aadhaar card [1]."
    assert FakeLLM.seen["path"] == "/v1/chat/completions"
    assert FakeLLM.seen["auth"] == "Bearer k123"
    msgs = FakeLLM.seen["body"]["messages"]
    assert "Reply in Hindi" in msgs[0]["content"] and "[1] PM Kisan" in msgs[1]["content"]


def test_language_and_intent():
    assert detect_language("विधवा पेंशन के लिए कौन से दस्तावेज चाहिए") == "hi"
    assert detect_language("widow pension documents") == "en"
    assert detect_intent("What documents are needed?") == "documents"
    assert detect_intent("How do I apply?") == "apply"
    assert detect_intent("Who can apply for PM Vishwakarma?") == "eligibility"


def test_closest_match_flag():
    from src.recommend.rag import names_match
    assert names_match("How much money does PM Kisan give?", "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)")
    assert not names_match("What is the benefit of Sukanya Samriddhi Yojana?", "Dr. Syama Prasad Mookerjee Durghtana Sahayta Yojana")
    assert names_match("What documents are needed?", "Anything")  # no scheme named -> no flag
