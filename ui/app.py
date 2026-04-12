import streamlit as st
import requests
import json
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MerchFine Planning System", 
    page_icon="🔮", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODERN CSS AESTHETICS (PREMIUM DESIGN)
# ==========================================
st.markdown("""
<style>
    /* Sleek Application Background */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] h1 {
        color: #60a5fa;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Transparent Header */
    [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.85);
        backdrop-filter: blur(10px);
    }
    
    /* Premium Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* User Message Specific Styling */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #3b82f6 !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10b981 !important;
    }
    
    /* Style standard text to be gorgeous */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 600;
    }
    
    /* Custom Input Box */
    [data-testid="stChatInput"] {
        border-color: #3b82f6;
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an AI greeting
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I am the MerchFine engine (Gemma-3-4B). I can generate strict data models, calculate supply chain inventory targets, or analyze stock levels. How can I assist you today?"
    })

# ==========================================
# SIDEBAR CONFIGURATIONS
# ==========================================
with st.sidebar:
    st.title("🔮 MerchFine Control")
    st.markdown("___")
    
    st.subheader("Backend Connection")
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000/api/chat")
    
    st.subheader("Inference Settings")
    use_rag = st.toggle("Enable RAG Context 📚", value=False, help="Connects to the ChromaDB Knowledge Base (if booted). Keeps False to run raw weights.")
    
    st.markdown("___")
    st.markdown("**Powered by:**")
    st.markdown("`unsloth/gemma-3-4b-it`")
    st.markdown("**(Local LoRA Finetuned)**")

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("MerchFine Operational Agent")
st.markdown("Data-driven inventory forecasting & planning engine.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Trap
if prompt := st.chat_input("Ask about sku logic, replenishment plans, or forecasts..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Analyzing logical parameters & calculating...*")
        
        # Prepare Request Payload
        payload = {
            "message": prompt,
            "use_rag": use_rag
        }
        
        try:
            start_time = time.time()
            response = requests.post(api_url, json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                reply_text = result.get("message", "No content returned.")
                model_used = result.get("model_used", "Unknown Model")
                
                # Append a nice footer showing latency
                latency = round((time.time() - start_time) * 1000, 2)
                footer = f"\n\n---\n*⚡ {latency}ms via {model_used}*"
                full_reply = reply_text + footer
                
                message_placeholder.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
            else:
                error_msg = f"❌ API Error {response.status_code}: {response.text}"
                message_placeholder.markdown(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.markdown("❌ **Connection Failed!** Is the FastAPI server running natively in your terminal (`python run_inference.py --api`)?")
        except Exception as e:
            message_placeholder.markdown(f"❌ **System Error:** {str(e)}")
