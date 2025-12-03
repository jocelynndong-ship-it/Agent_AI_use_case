import streamlit as st
import time
import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import bigquery

# --- IMPORTATION DES AGENTS ---
try:
    from generate_kpi_query_g3 import SQLGeneratorAgent
    from pipeline_agent import OrchestratorAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Agent Development Kit",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement ENV
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID", "sales_data")
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Config Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# --- CSS STYLE "DARK IDE" & UI MODERNE ---
st.markdown("""
<style>
    /* Fond global et Sidebar */
    .stApp { background-color: #1e1e1e; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #252526; border-right: 1px solid #333; }
    
    /* Style des Boutons Suggestions (Chips) */
    div.stButton > button:first-child {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3e3e42;
        border-radius: 8px;
        text-align: left;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        border-color: #007fd4;
        color: white;
        background-color: #333;
    }
    
    /* Chips d'étapes (Tools) */
    .tool-step { 
        background-color: #2d2d2d; 
        border: 1px solid #3e3e42; 
        border-radius: 6px; 
        padding: 8px 15px; 
        margin: 5px 0; 
        font-family: monospace; 
        font-size: 13px; 
        display: flex; 
        align-items: center; 
    }
    
    /* Masquer menu hamburger et footer standard */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    
    /* Custom Scrollbar pour faire plus "App" */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1e1e1e; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
</style>
""", unsafe_allow_html=True)

# --- FONCTION : SUGGESTIONS INTELLIGENTES ---
@st.cache_data(show_spinner=False)
def get_smart_suggestions(full_table_id):
    defaults = [
        "Combien de lignes dans la table ?",
        "Montre-moi les 5 premières lignes",
        "Quelle est la période couverte ?"
    ]
    if not api_key or not PROJECT_ID: return defaults
    try:
        client = bigquery.Client(project=PROJECT_ID)
        try:
            table = client.get_table(full_table_id)
            schema_str = "\n".join([f"{f.name} ({f.field_type})" for f in table.schema])
        except: return defaults

        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Analyse ce schéma BigQuery :
        {schema_str}
        Propose 4 questions business pertinentes, courtes et précises en Français.
        Réponds UNIQUEMENT avec un JSON array de strings.
        Exemple: ["Question A", "Question B"]
        """
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return defaults

# --- INITIALISATION SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "invocations" not in st.session_state: st.session_state.invocations = []
if "prompt_trigger" not in st.session_state: st.session_state.prompt_trigger = None
if "last_data" not in st.session_state: st.session_state.last_data = None # Pour le téléchargement

# --- SIDEBAR ---
with st.sidebar:
    # 1. HEADER
    c1, c2 = st.columns([1, 4])
    with c1: st.write("🛠️")
    with c2: 
        st.markdown("**Agent Dev Kit**")
        st.caption(f"Target: `{TABLE_ID}`")
    
    st.divider()
    
    # 2. ACTIONS
    if st.button("🗑️ Nouvelle Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.invocations = []
        st.session_state.prompt_trigger = None
        st.session_state.last_data = None
        st.rerun()

    st.divider()
    
    # 3. DEBUGGER FONCTIONNEL (TABS)
    st.markdown("### 🔍 Debugger")
    tab_trace, tab_state, tab_config = st.tabs(["Trace", "State", "Config"])
    
    with tab_trace:
        # Recherche du dernier SQL généré dans l'historique
        last_sql = next((m["content"] for m in reversed(st.session_state.messages) if m.get("tool_name") == "generate_kpi_query"), None)
        if last_sql:
            st.caption("Dernier SQL généré :")
            st.code(last_sql, language="sql")
        else:
            st.info("Aucune trace SQL.")

    with tab_state:
        # Affiche l'historique des prompts
        if st.session_state.invocations:
            for i, text in enumerate(reversed(st.session_state.invocations)):
                if i > 5: break
                st.markdown(f"<div style='background:#333; padding:5px; border-radius:4px; margin-bottom:4px; font-size:11px; border-left: 2px solid #007fd4;'>{text[:30]}...</div>", unsafe_allow_html=True)
        else:
            st.caption("Historique vide.")

    with tab_config:
        st.caption(f"Project: {PROJECT_ID}")
        st.caption(f"Dataset: {DATASET_ID}")
        st.checkbox("Verbose Mode", value=True, disabled=True)

    st.divider()
    
    # 4. SUGGESTIONS PERSISTANTES
    with st.expander("💡 Idées de questions", expanded=False):
        sidebar_suggs = get_smart_suggestions(FULL_TABLE_ID)
        for q in sidebar_suggs:
            if st.button(f"👉 {q}", key=f"side_{q}"):
                st.session_state.prompt_trigger = q
                st.rerun()

    # 5. SIGNATURE DÉVELOPPEUR
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 12px; margin-top: 10px;'>
            Developed by<br>
            <strong style='color: #e0e0e0; font-size: 13px;'>Jocelyn Ndong</strong><br>
            <span style='font-size: 11px; color: #666;'>Devoteam G Cloud</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- ZONE PRINCIPALE ---

# Header Session
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:10px; border-bottom:1px solid #333; margin-bottom: 20px;">
    <div><span style="color:#888; font-size:12px;">SESSION ID</span> <span style="font-family:monospace; color: #cecece;">active-session-dev</span></div>
    <div><span style="background:#0e639c; color:white; padding:2px 8px; border-radius:10px; font-size:10px;">Live Mode</span></div>
</div>""", unsafe_allow_html=True)

# 1. BIENVENUE & SUGGESTIONS (Si vide)
if len(st.session_state.messages) == 0:
    st.markdown("### 👋 Bonjour Analyste")
    st.markdown(f"Je suis connecté à `{FULL_TABLE_ID}`. Voici des pistes d'analyse :")
    suggestions = get_smart_suggestions(FULL_TABLE_ID)
    cols = st.columns(2)
    for i, question in enumerate(suggestions):
        col_idx = i % 2
        if cols[col_idx].button(question, use_container_width=True, key=f"main_sugg_{i}"):
            st.session_state.prompt_trigger = question
            st.rerun()

# 2. AFFICHAGE DE L'HISTORIQUE
for i, msg in enumerate(st.session_state.messages):
    if msg["type"] == "user":
        with st.chat_message("user", avatar="👤"): st.write(msg["content"])
    elif msg["type"] == "tool":
        with st.chat_message("assistant", avatar="🛠️"): # Avatar différent pour les outils
            st.markdown(f"<div class='tool-step'><span style='color: #4caf50; margin-right: 10px;'>✔</span> <b>{msg['tool_name']}</b></div>", unsafe_allow_html=True)
            with st.expander("Voir détails technique"): 
                st.code(msg["content"], language="sql" if "SELECT" in str(msg["content"]) else "json")
    elif msg["type"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if "image" in msg: st.image(msg["image"])
            # Feedback Button (Nouveauté Streamlit 1.31+)
            try:
                st.feedback("thumbs", key=f"fb_{i}")
            except: pass # Fallback si vieille version streamlit

# 3. GESTION INPUT
user_input = st.chat_input("Posez une question sur vos données...")

# Détection de l'origine du prompt
final_prompt = None
if st.session_state.prompt_trigger:
    final_prompt = st.session_state.prompt_trigger
    st.session_state.prompt_trigger = None 
elif user_input:
    final_prompt = user_input

# 4. EXÉCUTION
if final_prompt:
    # A. Ajout User Message
    st.session_state.messages.append({"type": "user", "content": final_prompt})
    st.session_state.invocations.append(final_prompt)
    if user_input: 
        with st.chat_message("user", avatar="👤"): st.write(final_prompt)
    
    if not AGENTS_AVAILABLE:
        st.error("❌ Les fichiers agents (generate_kpi_query_g3.py...) sont introuvables.")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            
            # --- AGENT 1: SQL ---
            status = st.empty()
            status.markdown("<div class='tool-step'><span style='margin-right:10px'>⚡</span> generating_sql...</div>", unsafe_allow_html=True)
            try:
                sql_agent = SQLGeneratorAgent()
                generated_sql = sql_agent.generate(FULL_TABLE_ID, final_prompt)
                
                if generated_sql:
                    status.markdown("<div class='tool-step'><span style='color:#4caf50; margin-right:10px'>✔</span> generate_kpi_query</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"type": "tool", "tool_name": "generate_kpi_query", "content": generated_sql})
                    with st.expander("Voir SQL généré"): st.code(generated_sql, language="sql")
                else:
                    status.markdown("<div class='tool-step'><span style='color:#f44336; margin-right:10px'>✖</span> SQL generation failed</div>", unsafe_allow_html=True)
                    st.stop()
            except Exception as e:
                status.error(f"Erreur SQL: {e}")
                st.stop()

            # --- AGENT 2: PIPELINE ---
            status_2 = st.empty()
            status_2.markdown("<div class='tool-step'><span style='margin-right:10px'>⚡</span> running_pipeline...</div>", unsafe_allow_html=True)
            try:
                pipeline = OrchestratorAgent()
                result = pipeline.act({"query": generated_sql})
                
                if "error" in result:
                    status_2.markdown("<div class='tool-step'><span style='color:#f44336; margin-right:10px'>✖</span> Pipeline failed</div>", unsafe_allow_html=True)
                    st.error(result["error"])
                else:
                    status_2.markdown("<div class='tool-step'><span style='color:#4caf50; margin-right:10px'>✔</span> run_pipeline</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"type": "tool", "tool_name": "run_pipeline", "content": "Pipeline success"})
                    
                    # Réponse finale
                    final_text = result.get("final_report", "Voici les résultats.")
                    
                    # Streaming effect
                    msg_ph = st.empty()
                    full_resp = ""
                    for chunk in final_text.split():
                        full_resp += chunk + " "
                        time.sleep(0.01)
                        msg_ph.markdown(full_resp + "▌")
                    msg_ph.markdown(full_resp)
                    
                    st.session_state.messages.append({"type": "assistant", "content": final_text})

                    # Affichage des Images
                    viz_list = result.get("generated_visualizations", [])
                    if viz_list:
                        for viz in viz_list:
                            import base64
                            img_data = base64.b64decode(viz["img_base64"])
                            st.image(img_data, caption=viz["name"])
                            st.session_state.messages.append({"type": "assistant", "content": "", "image": img_data})

                    # Bouton de Téléchargement CSV (Nouvelle fonctionnalité)
                    if "data" in result and not result["data"].empty:
                        csv = result["data"].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger les données (CSV)",
                            data=csv,
                            file_name="export_data.csv",
                            mime="text/csv",
                            key=f"dl_{len(st.session_state.messages)}"
                        )

            except Exception as e:
                status_2.error(f"Erreur Pipeline: {e}")
    
    # Refresh si déclenché par bouton pour nettoyer UI
    if not user_input:
        st.rerun()