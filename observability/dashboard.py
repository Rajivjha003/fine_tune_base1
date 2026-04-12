"""
Streamlit dashboard for observing MerchFine LLMOps metrics.

Run with: `streamlit run d:/Fine_tuning/merchfine/observability/dashboard.py`
"""

import streamlit as st
import httpx
import time

def fetch_admin_metrics():
    # Make a request to the FastAPI admin metrics endpoint
    # that we built in Layer 4
    try:
        resp = httpx.get("http://127.0.0.1:8000/admin/metrics", timeout=5.0)
        return resp.json()
    except Exception:
        return None

def fetch_admin_health():
    try:
        resp = httpx.get("http://127.0.0.1:8000/admin/health", timeout=5.0)
        return resp.json()
    except Exception:
        return None

st.set_page_config(page_title="MerchFine LLMOps", layout="wide")
st.title("🛍️ MerchMix LLMOps Dashboard")

# Health
st.header("⚡ System Health")
health = fetch_admin_health()

if health:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "🟢 Healthy" if health.get("is_healthy") else "🔴 Degraded")
    c2.metric("GPU", health.get("gpu_name", "N/A"))
    c3.metric("Ollama Reachable", "✅ Yes" if health.get("ollama_reachable") else "❌ No")
    c4.metric("MLflow Reachable", "✅ Yes" if health.get("mlflow_reachable") else "❌ No")
else:
    st.warning("Cannot connect to API. Is FastAPI running? (`uvicorn api.app:app --reload`)")

# Metrics
st.header("📊 Live Metrics")
metrics = fetch_admin_metrics()

if metrics:
    # Model
    st.subheader("Models")
    m1, m2 = st.columns(2)
    cfg = metrics.get("config", {})
    m1.metric("Primary Model", cfg.get("primary_model", "N/A"))
    
    fallbacks = cfg.get("fallback_models", [])
    m2.metric("Fallback Models", len(fallbacks) if fallbacks else 0)

    # Cache
    st.subheader("Semantic Cache")
    c1, c2 = st.columns(2)
    cache = metrics.get("cache", {})
    c1.metric("Backend", cache.get("backend", "None"))
    c2.metric("Entries in Memory", cache.get("memory_entries", 0))

    # Circuit Breaker
    st.subheader("Circuit Breaker Status")
    chain = metrics.get("fallback_chain", [])
    if chain:
        for idx, item in enumerate(chain):
            cb_status = "🟢" if item["state"] == "closed" else ("🔴" if item["state"] == "open" else "🟡")
            st.write(f"{idx+1}. {item['name']} - Status: {cb_status} {item['state']} (Failures: {item['consecutive_failures']})")
    else:
        st.write("No fallback backends configured.")

st.divider()
st.caption("Refresh browser to update data. Data is fetched live from the FastAPI admin routes.")
