import streamlit as st
import time
import os
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import bigquery

# --- IMPORTATION AGENTS ---
try:
    from generate_kpi_query_g3 import SQLGeneratorAgent
    from pipeline_agent import OrchestratorAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# --- CONFIGURATION PAGE ---
st.set_page_config(
    page_title="Agent Analytics",
    page_icon="📊",
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

# --- CSS STYLE "MODERN DASHBOARD" ---
st.markdown("""
<style>
    /* Fond sombre et texte */
    .stApp { background-color: #121212; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #1e1e1e; border-right: 1px solid #333; }
    
    /* Boutons Suggestions */
    div.stButton > button:first-child {
        background-color: #252526;
        color: #cecece;
        border: 1px solid #3e3e42;
        border-radius: 6px;
        text-align: left;
        padding: 10px;
    }
    div.stButton > button:first-child:hover {
        border-color: #007fd4;
        color: white;
        background-color: #333;
    }
    
    /* Containers de Graphiques Uniformes */
    .viz-container {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Logs et Tools */
    .tool-step { background-color: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 8px 12px; margin: 5px 0; font-family: monospace; font-size: 13px; display: flex; align-items: center; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #121212; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FONCTION : INTELLIGENT PLOTTING (PLOTLY) ---
def generate_interactive_dashboard(df):
    """Génère des graphiques Plotly interactifs et propres à partir du DataFrame."""
    
    # 1. Nettoyage et Typage
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    date_cols = []
    
    # Détection des dates
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except: pass

    # --- KPI CARDS (En haut) ---
    if len(num_cols) > 0:
        st.markdown("### 📈 Indicateurs Clés")
        cols = st.columns(4)
        for i, col in enumerate(num_cols[:4]):
            val = df[col].sum()
            # Formatage compact (K, M)
            if val > 1_000_000: fmt_val = f"{val/1_000_000:.1f}M"
            elif val > 1_000: fmt_val = f"{val/1_000:.1f}K"
            else: fmt_val = f"{val:.0f}"
            
            cols[i].metric(label=col, value=fmt_val)
        st.divider()

    # --- GRAPHIQUES ---
    st.markdown("### 📊 Visualisations")
    
    # Layout en 2 colonnes pour uniformité
    c1, c2 = st.columns(2)
    
    # VIZ 1 : Time Series (si date dispo)
    with c1:
        if date_cols and num_cols:
            date_col = date_cols[0]
            val_col = num_cols[0]
            # Agrégation par date pour éviter le bruit
            df_agg = df.groupby(date_col)[val_col].sum().reset_index().sort_values(date_col)
            
            fig = px.line(df_agg, x=date_col, y=val_col, title=f"Évolution : {val_col}", markers=True)
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de données temporelles détectées.")

    # VIZ 2 : Bar Chart (Top Categories)
    with c2:
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            val_col = num_cols[0]
            
            # TOP 10 pour éviter les légendes infinies
            top_df = df.groupby(cat_col)[val_col].sum().reset_index().nlargest(10, val_col)
            
            fig = px.bar(top_df, x=cat_col, y=val_col, color=cat_col, title=f"Top 10 : {cat_col}")
            fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de données catégorielles pour un Top 10.")

    # VIZ 3 : Scatter Plot (Correlation)
    if len(num_cols) >= 2:
        with st.container():
            x_col, y_col = num_cols[0], num_cols[1]
            
            # Gestion intelligente de la couleur (Hue)
            color_col = None
            if cat_cols:
                # Si trop de valeurs uniques (>15), on n'utilise PAS la couleur
                if df[cat_cols[0]].nunique() < 15:
                    color_col = cat_cols[0]
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                             size=num_cols[2] if len(num_cols)>2 else None,
                             title=f"Corrélation : {x_col} vs {y_col}", hover_data=cat_cols[:1])
            
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# --- FONCTION SUGGESTIONS ---
@st.cache_data(show_spinner=False)
def get_smart_suggestions(full_table_id):
    defaults = ["Top 10 produits par ventes", "Chiffre d'affaires mensuel", "Répartition par catégorie"]
    if not api_key or not PROJECT_ID: return defaults
    try:
        client = bigquery.Client(project=PROJECT_ID)
        try: table = client.get_table(full_table_id)
        except: return defaults
        
        schema_str = "\n".join([f"{f.name} ({f.field_type})" for f in table.schema])
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Schema BigQuery: {schema_str}
        Génère 3 questions analytiques courtes pour un Dashboard (JSON array string).
        """
        response = model.generate_content(prompt)
        return json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except: return defaults

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "last_sql" not in st.session_state: st.session_state.last_sql = None # <--- NOUVEAU
if "prompt_trigger" not in st.session_state: st.session_state.prompt_trigger = None

# --- SIDEBAR ---
with st.sidebar:
    c1, c2 = st.columns([1, 4])
    with c1: st.write("📊")
    with c2: 
        st.markdown("**Agent Analytics**")
        st.caption(f"Table: `{TABLE_ID}`")
    
    st.divider()
    if st.button("🗑️ Reset Dashboard", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.last_sql = None
        st.session_state.prompt_trigger = None
        st.rerun()

    st.markdown("### 💡 Suggestions")
    suggs = get_smart_suggestions(FULL_TABLE_ID)
    for q in suggs:
        if st.button(f"👉 {q}", key=q):
            st.session_state.prompt_trigger = q
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;font-size:12px;'>Jocelyn Ndong<br>Devoteam G Cloud</div>", unsafe_allow_html=True)

# --- MAIN ---
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; padding-bottom:10px; margin-bottom:20px;">
    <h3 style="margin:0;">Dashboard Interactif</h3>
    <span style="background:#007fd4; color:white; padding:4px 10px; border-radius:15px; font-size:12px;">Live Data</span>
</div>""", unsafe_allow_html=True)

# 1. INPUT
user_input = st.chat_input("Ex: Quel est le chiffre d'affaires par ville ?")
final_prompt = st.session_state.prompt_trigger if st.session_state.prompt_trigger else user_input

if final_prompt:
    st.session_state.prompt_trigger = None
    
    # Affichage User
    with st.chat_message("user", avatar="👤"): st.write(final_prompt)

    if AGENTS_AVAILABLE:
        with st.chat_message("assistant", avatar="🤖"):
            status = st.status("Analyse en cours...", expanded=True)
            
            try:
                # Agent 1 : SQL
                status.write("🧠 Génération du SQL...")
                sql_agent = SQLGeneratorAgent()
                sql = sql_agent.generate(FULL_TABLE_ID, final_prompt)
                
                if not sql:
                    status.update(label="Échec SQL", state="error")
                    st.stop()
                
                # --- SAUVEGARDE DU SQL DANS L'ÉTAT ---
                st.session_state.last_sql = sql 
                
                # Agent 2 : Pipeline
                status.write("🚀 Exécution BigQuery...")
                pipeline = OrchestratorAgent()
                result = pipeline.act({"query": sql})
                
                if "error" in result:
                    status.update(label="Erreur Pipeline", state="error")
                    st.error(result["error"])
                else:
                    status.update(label="Terminé !", state="complete", expanded=False)
                    st.session_state.last_result = result
                    
            except Exception as e:
                status.update(label="Erreur Critique", state="error")
                st.error(str(e))

# 2. AFFICHAGE DU DASHBOARD (PERSISTANT)
if st.session_state.last_result:
    res = st.session_state.last_result
    df = res.get("data")
    report = res.get("final_report")

    # Onglets
    tab_dash, tab_report, tab_data, tab_sql = st.tabs(["📊 Dashboard Visuel", "📝 Rapport IA", "💾 Données", "⚙️ SQL"])

    with tab_dash:
        if df is not None and not df.empty:
            generate_interactive_dashboard(df)
        else:
            st.warning("Aucune donnée retournée.")

    with tab_report:
        st.markdown(report)
        st.feedback("thumbs")

    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.download_button("Télécharger CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

    with tab_sql:
        # --- AFFICHAGE DU SQL SAUVEGARDÉ ---
        if st.session_state.last_sql:
            st.code(st.session_state.last_sql, language="sql")
        else:
            st.info("Aucun code SQL disponible pour le moment.")