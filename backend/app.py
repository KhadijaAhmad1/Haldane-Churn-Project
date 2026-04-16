"""
=============================================================================
  OPTIONAL BACKEND PROXY — keeps your API key server-side
  KTP Project · Haldane Group × Queen's University Belfast
=============================================================================

  WHY USE THIS:
  -------------
  The frontend/index.html works by taking your API key in the browser.
  That is fine for demos and presentations but not for production.
  This Flask server acts as a middleman — the frontend calls THIS server,
  and THIS server calls the AI provider using a key stored in .env.
  The key never leaves your server.

  HOW TO RUN:
  -----------
  1. pip install flask flask-cors python-dotenv
  2. Create a .env file in the project root:
       ANTHROPIC_API_KEY=sk-ant-...
  3. python backend/app.py
  4. Server starts at http://localhost:5000
  5. Open frontend/index.html — it will auto-detect the local proxy

=============================================================================
"""

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()                          # reads .env from project root

app = Flask(__name__)
CORS(app)                              # allow requests from the HTML frontend

API_KEY = os.getenv("API_KEY", "")


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "key_loaded": bool(API_KEY)})


# ── Proxy: Claude (Anthropic) ─────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({"error": "API_KEY not set in .env"}), 500

    body = request.get_json()

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":                          "application/json",
            "x-api-key":                             API_KEY,
            "anthropic-version":                     "2023-06-01",
        },
        json={
            "model":      body.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": body.get("max_tokens", 1000),
            "system":     body.get("system", ""),
            "messages":   body.get("messages", []),
        },
        timeout=60,
    )

    return jsonify(response.json()), response.status_code


if __name__ == "__main__":
    print("=" * 55)
    print("  Haldane Group AI Retention Hub — Backend Proxy")
    print(f"  API key loaded: {'YES' if API_KEY else 'NO — check your .env file'}")
    print("  Running at: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
