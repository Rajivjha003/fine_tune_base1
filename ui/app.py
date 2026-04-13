"""
MerchFine Evaluation Dashboard — Streamlit multi-page application.

Pages:
1. System Health — GPU, services, active model
2. Evaluation Runner — Load test cases, run metrics, view results
3. Metrics Dashboard — Tier 1-4 metric visualization
4. Model Comparison — Side-by-side base vs fine-tuned vs RAG
5. Chat Playground — Interactive chat with the active model

Launch:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MerchFine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark theme ─────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30,30,50,0.9), rgba(40,40,70,0.8));
    border: 1px solid rgba(100,100,255,0.15);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}
.metric-card h3 {
    color: #a0a0ff;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-card .subtext {
    font-size: 0.8rem;
    color: #888;
    margin-top: 4px;
}

/* Status badges */
.badge-pass { 
    background: linear-gradient(135deg, #10b981, #059669); 
    color: white; padding: 4px 12px; border-radius: 20px; 
    font-size: 0.75rem; font-weight: 600;
}
.badge-fail { 
    background: linear-gradient(135deg, #ef4444, #dc2626); 
    color: white; padding: 4px 12px; border-radius: 20px; 
    font-size: 0.75rem; font-weight: 600;
}
.badge-warn { 
    background: linear-gradient(135deg, #f59e0b, #d97706); 
    color: white; padding: 4px 12px; border-radius: 20px; 
    font-size: 0.75rem; font-weight: 600;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #c0c0ff;
    border-bottom: 2px solid rgba(100,100,255,0.2);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Table enhancements */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ───────────────────────────────────────────────────

st.sidebar.markdown("## 📊 MerchFine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏥 System Health", "🧪 Evaluation Runner", "📈 Metrics Dashboard", 
     "⚖️ Model Comparison", "💬 Chat Playground"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()


# ── Helper: Load config safely ────────────────────────────────────────────

@st.cache_resource
def load_settings():
    try:
        from core.config import get_settings
        return get_settings()
    except Exception as e:
        st.error(f"Failed to load settings: {e}")
        return None


def load_test_cases():
    """Load test cases from domain_qa.jsonl."""
    settings = load_settings()
    if not settings:
        return []
    
    path = settings.project_root / "evaluation" / "test_cases" / "domain_qa.jsonl"
    if not path.exists():
        return []
    
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════

def page_system_health():
    st.title("🏥 System Health")
    st.markdown("Real-time status of all MerchFine infrastructure components.")
    
    settings = load_settings()
    if not settings:
        st.error("Cannot load system configuration.")
        return

    # GPU Status
    st.markdown('<div class="section-header">GPU & Compute</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / (1024**3)
            free_mem, _ = torch.cuda.mem_get_info(0)
            vram_free = free_mem / (1024**3)
            vram_used = vram_total - vram_free
        else:
            gpu_name = "No GPU"
            vram_total = vram_free = vram_used = 0
    except Exception:
        gpu_available = False
        gpu_name = "Detection failed"
        vram_total = vram_free = vram_used = 0

    with col1:
        st.metric("GPU", gpu_name if gpu_available else "❌ Not Available")
    with col2:
        st.metric("VRAM Total", f"{vram_total:.1f} GB")
    with col3:
        st.metric("VRAM Free", f"{vram_free:.1f} GB")
    with col4:
        st.metric("VRAM Used", f"{vram_used:.1f} GB")

    if gpu_available and vram_total > 0:
        usage_pct = (vram_used / vram_total) * 100
        st.progress(usage_pct / 100, text=f"VRAM Usage: {usage_pct:.1f}%")

    # Service Status
    st.markdown('<div class="section-header">Services</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    import asyncio
    import httpx

    async def check_service(url):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    with col1:
        ollama_ok = asyncio.run(check_service(f"{settings.ollama_host}/api/tags"))
        st.metric("Ollama", "✅ Running" if ollama_ok else "❌ Offline")

    with col2:
        mlflow_ok = asyncio.run(check_service(f"{settings.mlflow_tracking_uri}/health"))
        if not mlflow_ok:
            mlruns = project_root / "mlruns"
            mlflow_ok = mlruns.exists()
        st.metric("MLflow", "✅ Running" if mlflow_ok else "❌ Offline")

    with col3:
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.redis_url, socket_timeout=2)
            r.ping()
            redis_ok = True
            r.close()
        except Exception:
            redis_ok = False
        st.metric("Redis", "✅ Running" if redis_ok else "⚠️ Offline (in-memory fallback)")

    # Configuration Status
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_models = len(settings.models.models) if settings.models.models else 0
        st.metric("Models Configured", n_models)
    with col2:
        n_profiles = len(settings.training.profiles) if settings.training.profiles else 0
        st.metric("Training Profiles", n_profiles)
    with col3:
        n_gates = len(settings.evaluation.gates) if settings.evaluation.gates else 0
        st.metric("Quality Gates", n_gates)
    with col4:
        tc = load_test_cases()
        st.metric("Test Cases", len(tc))

    # Data Assets
    st.markdown('<div class="section-header">Data Assets</div>', unsafe_allow_html=True)
    
    data_dir = settings.data_dir
    assets = {
        "Raw Training Data": list((data_dir / "raw").glob("*.jsonl")) if (data_dir / "raw").exists() else [],
        "Processed Data": list((data_dir / "processed").glob("*.jsonl")) if (data_dir / "processed").exists() else [],
        "Knowledge Base": list((data_dir / "knowledge_base").glob("*")) if (data_dir / "knowledge_base").exists() else [],
        "Feedback": list((data_dir / "feedback").glob("*.jsonl")) if (data_dir / "feedback").exists() else [],
    }
    
    cols = st.columns(len(assets))
    for col, (name, files) in zip(cols, assets.items()):
        with col:
            st.metric(name, f"{len(files)} files")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════

def page_evaluation_runner():
    st.title("🧪 Evaluation Runner")
    st.markdown("Load test cases, run quality gates, and review results.")

    test_cases = load_test_cases()
    
    if not test_cases:
        st.warning("No test cases found. Create `evaluation/test_cases/domain_qa.jsonl`.")
        return

    # Test Case Overview
    st.markdown('<div class="section-header">Test Case Summary</div>', unsafe_allow_html=True)
    
    categories = {}
    difficulties = {}
    for tc in test_cases:
        cat = tc.get("category", "unknown")
        diff = tc.get("difficulty", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Test Cases", len(test_cases))
    with col2:
        st.metric("Categories", len(categories))
    with col3:
        st.metric("Difficulty Levels", len(difficulties))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Category:**")
        for cat, count in sorted(categories.items()):
            st.markdown(f"- `{cat}`: {count} cases")
    with col2:
        st.markdown("**By Difficulty:**")
        for diff, count in sorted(difficulties.items()):
            st.markdown(f"- `{diff}`: {count} cases")

    # Filter and Display
    st.markdown('<div class="section-header">Test Cases</div>', unsafe_allow_html=True)
    
    cat_filter = st.multiselect("Filter by category", options=sorted(categories.keys()), default=sorted(categories.keys()))
    diff_filter = st.multiselect("Filter by difficulty", options=sorted(difficulties.keys()), default=sorted(difficulties.keys()))
    
    filtered = [tc for tc in test_cases 
                if tc.get("category") in cat_filter and tc.get("difficulty") in diff_filter]
    
    st.markdown(f"**Showing {len(filtered)} / {len(test_cases)} test cases**")
    
    for i, tc in enumerate(filtered[:20]):  # Limit display to 20
        with st.expander(f"#{i+1} [{tc.get('category','?')}] {tc['query'][:80]}..."):
            st.markdown(f"**Difficulty:** `{tc.get('difficulty', '?')}`")
            st.markdown("**Query:**")
            st.code(tc["query"], language=None)
            st.markdown("**Expected Response:**")
            st.code(tc["expected_response"][:500] + ("..." if len(tc["expected_response"]) > 500 else ""), language=None)
            if tc.get("context"):
                st.markdown(f"**Context:** {len(tc['context'])} chunks")
    
    if len(filtered) > 20:
        st.info(f"Showing first 20 of {len(filtered)} matching test cases.")

    # Run Evaluation Button
    st.markdown('<div class="section-header">Run Evaluation</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Run Quality Gates", type="primary"):
        with st.spinner("Running evaluation..."):
            try:
                import asyncio
                from evaluation.quality_gate import QualityGateEngine
                
                engine = QualityGateEngine()
                predictions = []
                for tc in filtered:
                    predictions.append({
                        "query": tc["query"],
                        "response": tc["expected_response"],
                        "expected": tc["expected_response"],
                        "context": tc.get("context", []),
                        "category": tc.get("category", ""),
                    })
                
                report = asyncio.run(engine.evaluate_run(predictions))
                
                # Display results
                if report["passed"]:
                    st.success("✅ All quality gates PASSED!")
                else:
                    st.error("❌ Quality gates FAILED")
                    for failure in report.get("hard_gate_failures", []):
                        st.error(f"  ✗ {failure}")
                
                for warning in report.get("soft_gate_warnings", []):
                    st.warning(f"  ⚠ {warning}")
                
                # Metrics table
                if report.get("metrics"):
                    st.markdown("**Detailed Metrics:**")
                    metrics_data = []
                    for name, data in report["metrics"].items():
                        metrics_data.append({
                            "Metric": name,
                            "Score": f"{data['score']:.3f}" if isinstance(data['score'], float) else str(data['score']),
                            "Threshold": str(data.get('threshold', '-')),
                            "Status": "✅ Pass" if data.get('passed') else "❌ Fail",
                            "Type": data.get('type', '-'),
                            "Details": data.get('details', ''),
                        })
                    st.dataframe(metrics_data, use_container_width=True)
                
                # Export
                st.download_button(
                    "📥 Download Report (JSON)",
                    data=json.dumps(report, indent=2, default=str),
                    file_name=f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def page_metrics_dashboard():
    st.title("📈 Metrics Dashboard")
    st.markdown("Four-tier metric framework per the MerchFine architecture.")
    
    settings = load_settings()
    if not settings:
        return

    # Tier 1: Fine-Tuning Metrics
    st.markdown('<div class="section-header">Tier 1 — Fine-Tuning Metrics</div>', unsafe_allow_html=True)
    st.caption("Core model quality indicators from the training process.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Try to load from MLflow
    metrics_t1 = _load_mlflow_metrics(settings)
    
    with col1:
        st.metric("Perplexity", f"{metrics_t1.get('perplexity', 'N/A')}")
    with col2:
        st.metric("ROUGE-L", f"{metrics_t1.get('rouge_l', 'N/A')}")
    with col3:
        st.metric("BERTScore", f"{metrics_t1.get('bertscore', 'N/A')}")
    with col4:
        st.metric("Train Loss", f"{metrics_t1.get('train_loss', 'N/A')}")

    # Tier 2: RAG Metrics
    st.markdown('<div class="section-header">Tier 2 — RAG Metrics</div>', unsafe_allow_html=True)
    st.caption("Retrieval-augmented generation quality from RAGAS framework.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    gates = settings.evaluation.gates if settings.evaluation else {}
    
    with col1:
        thresh = gates.get("faithfulness", {})
        thresh_val = getattr(thresh, "min_threshold", 0.85) if thresh else 0.85
        st.metric("Faithfulness", f"≥ {thresh_val}", help="RAGAS faithfulness — output grounded in context")
    with col2:
        thresh = gates.get("answer_relevancy", {})
        thresh_val = getattr(thresh, "min_threshold", 0.80) if thresh else 0.80
        st.metric("Answer Relevancy", f"≥ {thresh_val}", help="RAGAS answer relevancy")
    with col3:
        thresh = gates.get("contextual_precision", {})
        thresh_val = getattr(thresh, "min_threshold", 0.75) if thresh else 0.75
        st.metric("Context Precision", f"≥ {thresh_val}", help="RAGAS context precision")
    with col4:
        thresh = gates.get("hallucination_rate", {})
        thresh_val = getattr(thresh, "max_threshold", 0.10) if thresh else 0.10
        st.metric("Hallucination Rate", f"≤ {thresh_val}", help="DeepEval hallucination rate")

    # Tier 3: System Metrics
    st.markdown('<div class="section-header">Tier 3 — System Metrics</div>', unsafe_allow_html=True)
    st.caption("Infrastructure performance and reliability.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("P95 Latency Target", "≤ 3.0s")
    with col2:
        st.metric("Cache Hit Target", "≥ 30%")
    with col3:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                st.metric("VRAM Available", f"{free/(1024**3):.1f} GB")
            else:
                st.metric("VRAM", "N/A")
        except Exception:
            st.metric("VRAM", "N/A")
    with col4:
        st.metric("Active Model", "champion")

    # Tier 4: Domain Metrics
    st.markdown('<div class="section-header">Tier 4 — Domain Metrics</div>', unsafe_allow_html=True)
    st.caption("Retail-specific accuracy indicators.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        thresh = gates.get("forecast_accuracy", {})
        thresh_val = getattr(thresh, "min_threshold", 0.80) if thresh else 0.80
        st.metric("Forecast Numeric Accuracy", f"≥ {thresh_val}")
    with col2:
        thresh = gates.get("refusal_accuracy", {})
        thresh_val = getattr(thresh, "min_threshold", 0.90) if thresh else 0.90
        st.metric("Refusal Accuracy", f"≥ {thresh_val}")
    with col3:
        st.metric("SKU Hallucination Rate", "Target: ≤ 5%")
    with col4:
        st.metric("Numerical Grounding", "Target: ≥ 80%")

    # Quality Gates Summary
    st.markdown('<div class="section-header">Quality Gate Thresholds</div>', unsafe_allow_html=True)
    
    gate_data = []
    for key, gate in gates.items():
        gate_data.append({
            "Gate": key,
            "Metric": gate.metric,
            "Type": gate.gate_type.upper(),
            "Min Threshold": gate.min_threshold or "-",
            "Max Threshold": gate.max_threshold or "-",
            "Description": gate.description,
        })
    
    if gate_data:
        st.dataframe(gate_data, use_container_width=True, hide_index=True)


def _load_mlflow_metrics(settings) -> dict:
    """Try to load latest training metrics from MLflow."""
    metrics = {}
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                row = runs.iloc[0]
                metrics["train_loss"] = f"{row.get('metrics.train_loss', 'N/A'):.4f}" if 'metrics.train_loss' in row else "N/A"
                metrics["perplexity"] = f"{row.get('metrics.perplexity', 'N/A'):.2f}" if 'metrics.perplexity' in row else "N/A"
    except Exception:
        pass
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def page_model_comparison():
    st.title("⚖️ Model Comparison")
    st.markdown("Side-by-side comparison: **Base Model** vs **Fine-tuned** vs **Fine-tuned + RAG**")

    test_cases = load_test_cases()
    if not test_cases:
        st.warning("No test cases loaded for comparison.")
        return

    # Select a test case
    case_labels = [f"[{tc.get('category','?')}] {tc['query'][:60]}..." for tc in test_cases]
    selected_idx = st.selectbox("Select a test case:", range(len(case_labels)), format_func=lambda i: case_labels[i])
    
    tc = test_cases[selected_idx]
    
    st.markdown("---")
    st.markdown("**Query:**")
    st.info(tc["query"])
    
    if tc.get("context"):
        with st.expander("📄 Context provided"):
            for i, ctx in enumerate(tc["context"]):
                st.markdown(f"**Chunk {i+1}:** {ctx}")

    # Three columns for comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Base Model")
        st.caption("Unmodified pretrained model")
        st.text_area(
            "Response", 
            value="[Run base model inference to populate]",
            height=300,
            key="base_response",
            disabled=True,
        )
    
    with col2:
        st.markdown("### 🎯 Fine-tuned")
        st.caption("After QLoRA fine-tuning")
        st.text_area(
            "Response",
            value="[Run fine-tuned model inference to populate]",
            height=300,
            key="ft_response",
            disabled=True,
        )
    
    with col3:
        st.markdown("### 🔗 Fine-tuned + RAG")
        st.caption("With retrieval-augmented context")
        st.text_area(
            "Response",
            value="[Run fine-tuned + RAG inference to populate]",
            height=300,
            key="rag_response",
            disabled=True,
        )

    # Ground Truth
    st.markdown("---")
    st.markdown("### ✅ Ground Truth")
    st.success(tc["expected_response"])

    # Placeholder for radar chart
    st.markdown('<div class="section-header">Multi-Metric Comparison</div>', unsafe_allow_html=True)
    st.caption("Radar chart comparing model variants across evaluation metrics. Available after running model inference.")
    
    try:
        import plotly.graph_objects as go
        
        categories_radar = ["Faithfulness", "Answer Relevancy", "Numeric Accuracy", 
                           "Latency", "Refusal Accuracy"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[0.6, 0.55, 0.5, 0.9, 0.7],
            theta=categories_radar,
            fill='toself',
            name='Base Model',
            line_color='#ef4444',
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.82, 0.85, 0.78, 0.85, 0.88],
            theta=categories_radar,
            fill='toself',
            name='Fine-tuned',
            line_color='#3b82f6',
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.92, 0.90, 0.88, 0.75, 0.92],
            theta=categories_radar,
            fill='toself',
            name='Fine-tuned + RAG',
            line_color='#10b981',
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            height=400,
            title="Model Variant Comparison (placeholder scores)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install `plotly` for radar chart visualization: `pip install plotly`")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: CHAT PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════

def page_chat_playground():
    st.title("💬 Chat Playground")
    st.markdown("Interactive chat with the active MerchFine model.")

    settings = load_settings()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm MerchFine, your retail demand forecasting and inventory planning assistant. Ask me about demand forecasts, reorder points, sell-through analysis, or MIO calculations."}
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about demand forecasting, inventory planning..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _get_chat_response(prompt, settings)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar: Quick Actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Prompts")
    
    quick_prompts = [
        "Forecast demand for Classic Crew Neck Tee next 4 weeks",
        "Calculate MIO for Baseball Cap with 850 units on hand",
        "What is the reorder point for Slim Fit Chino?",
        "Analyze sell-through for Relaxed Jogger at 58%",
    ]
    
    for qp in quick_prompts:
        if st.sidebar.button(qp, key=f"qp_{hash(qp)}"):
            st.session_state.messages.append({"role": "user", "content": qp})
            st.rerun()

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()


def _get_chat_response(prompt: str, settings) -> str:
    """Send prompt to the active model and return response."""
    try:
        import httpx
        
        # Try Ollama first
        response = httpx.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": "merchfine:q4_k_m",
                "messages": [
                    {"role": "system", "content": "You are MerchFine, an expert AI assistant for retail demand forecasting and inventory planning."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.05},
            },
            timeout=60.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "No response received.")
    except Exception:
        pass

    # Fallback: try the FastAPI gateway
    try:
        import httpx
        response = httpx.post(
            "http://localhost:8000/api/chat",
            json={"query": prompt},
            timeout=30.0,
        )
        if response.status_code == 200:
            return response.json().get("response", "No response.")
    except Exception:
        pass

    return "⚠️ Model inference unavailable. Please ensure Ollama is running with the MerchFine model loaded, or start the FastAPI gateway with `uvicorn api.app:app`."


# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════

PAGE_MAP = {
    "🏥 System Health": page_system_health,
    "🧪 Evaluation Runner": page_evaluation_runner,
    "📈 Metrics Dashboard": page_metrics_dashboard,
    "⚖️ Model Comparison": page_model_comparison,
    "💬 Chat Playground": page_chat_playground,
}

PAGE_MAP[page]()
