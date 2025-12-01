import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Agent Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(override=True)

PROJECT_ID = os.getenv("PROJECT_ID", "Inconnu")
TABLE_ID = os.getenv("TABLE_ID", "Non spécifiée")

# Lecture SQL
QUERY_FILE = Path("results/generated_query.sql")
if QUERY_FILE.exists():
    QUERY = QUERY_FILE.read_text(encoding="utf-8")
else:
    QUERY = os.getenv("QUERY", "")

# Cache Busting
if 'last_table_id' not in st.session_state: st.session_state.last_table_id = None
if st.session_state.last_table_id != TABLE_ID:
    st.cache_data.clear()
    st.session_state.last_table_id = TABLE_ID
    st.rerun()

# --- 2. CHEMINS & LIBS ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError: HAS_PLOTLY = False

BASE_DIR = Path("results")
DATA_FILE = BASE_DIR / "data" / "data.parquet"
REPORT_FILE = BASE_DIR / "rapport_final.md"
VIZ_DIR = BASE_DIR / "visualisations"

# --- 3. FONCTIONS ---

@st.cache_data
def load_data_smart(file_path, table_tag, time_tag):
    p = Path(file_path)
    if not p.exists(): return None
    try: df = pd.read_parquet(p)
    except: 
        try: df = pd.read_parquet(p, engine='pyarrow')
        except: return None
        
    for col in df.columns:
        if df[col].dtype == 'object' or 'date' in str(df[col].dtype).lower():
            try: df[col] = pd.to_datetime(df[col], errors='ignore')
            except: pass
        # Conversion string pour affichage propre
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('nan', '', regex=False)
    return df

def get_file_info(p): return os.path.getmtime(p) if p.exists() else 0
def format_big_number(n): return f"{n/1000:.1f}K" if n > 1000 else str(n)
def clean_viz_title(n): 
    n = re.sub(r'viz_\d+_', '', n.replace(".png", ""))
    return n.replace("_", " ").title()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🤖 Agent Analytics")
    st.info(f"**Table** : `{TABLE_ID}`")
    st.caption(f"Projet : {PROJECT_ID}")
    
    if DATA_FILE.exists():
        ts = get_file_info(DATA_FILE)
        st.success(f"🕒 Données : {time.strftime('%H:%M:%S', time.localtime(ts))}")

    if st.button("🔄 Reset Cache", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    if QUERY:
        with st.expander("Voir SQL", expanded=False):
            st.code(QUERY, language="sql")

    st.markdown("---")
    st.markdown("<div style='text-align: center; font-size: 11px; color: #666;'>Auteur : <b>Jocelyn NDONG</b><br>Analyst Engineer (Devoteam G Cloud)</div>", unsafe_allow_html=True)

# --- 5. MAIN ---
st.title(f"📊 Dashboard : {TABLE_ID.split('.')[-1]}")

if not DATA_FILE.exists():
    st.error("Données absentes. Lancez le pipeline.")
    st.stop()

df = load_data_smart(str(DATA_FILE), TABLE_ID, get_file_info(DATA_FILE))

if df is not None:
    # Classification des colonnes
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # --- DÉTECTION INTELLIGENTE GÉO ---
    # 1. Coordonnées précises
    geo_lat = next((c for c in df.columns if 'lat' in c.lower()), None)
    geo_lon = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
    
    # 2. Adresses / Lieux (Texte)
    # Mots clés courants pour les lieux
    geo_keywords = ['address', 'adresse', 'city', 'ville', 'country', 'pays', 'state', 'region', 'zip', 'postal', 'loc', 'purchase_address']
    geo_addr_cols = [c for c in cat_cols if any(k in c.lower() for k in geo_keywords)]

    # Scorecards
    st.markdown("### 🚀 Vue d'Ensemble")
    cols = st.columns(6)
    cols[0].metric("Lignes", len(df))
    cols[1].metric("Colonnes", len(df.columns))
    idx = 2
    for col in num_cols[:4]:
        if df[col].nunique() > 5:
            tot = df[col].sum()
            label = "Total" if tot > 1000 else "Moy."
            val = tot if tot > 1000 else df[col].mean()
            cols[idx].metric(f"{label} {col}", format_big_number(val))
            idx += 1
    st.markdown("---")

    # Onglets
    tabs = st.tabs(["📈 Comparaison", "🍩 Distribution", "🌍 Géospatial", "🖼️ Galerie Statique", "📝 Rapport", "💾 Données Brutes"])

    if not HAS_PLOTLY: st.error("Plotly manquant.")
    else:
        # TAB 1
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📊 Comparaison")
                if cat_cols and num_cols:
                    x = st.selectbox("Axe X", cat_cols, key="b_x")
                    y = st.selectbox("Axe Y", num_cols, key="b_y")
                    grp = st.selectbox("Grouper", [None]+cat_cols, key="b_g")
                    df_agg = df.groupby([x]+([grp] if grp else []))[y].sum().reset_index().sort_values(y, ascending=False).head(20)
                    fig = px.bar(df_agg, x=x, y=y, color=grp, title=f"Top {y} par {x}")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### 📈 Tendance")
                if date_cols and num_cols:
                    dx = st.selectbox("Temps", date_cols, key="l_x")
                    dy = st.selectbox("Mesure", num_cols, key="l_y")
                    df_t = df.groupby(dx)[dy].sum().reset_index().sort_values(dx)
                    fig = px.line(df_t, x=dx, y=dy, markers=True)
                    st.plotly_chart(fig, use_container_width=True)

        # TAB 2
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🍩 Répartition")
                if cat_cols and num_cols:
                    pc = st.selectbox("Dim", cat_cols, key="p_c")
                    pv = st.selectbox("Val", num_cols, key="p_v")
                    fig = px.pie(df, names=pc, values=pv, hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("#### 📦 Distribution")
                if num_cols:
                    dc = st.selectbox("Var", num_cols, key="d_c")
                    fig = px.histogram(df, x=dc, nbins=30, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # TAB 3 : GÉOSPATIAL (LOGIQUE HYBRIDE)
        # ---------------------------------------------------------
        with tabs[2]:
            st.subheader("🌍 Analyse Géographique")
            
            # CAS 1 : Lat/Lon disponibles -> Carte
            if geo_lat and geo_lon:
                st.success(f"📍 Coordonnées GPS détectées : `{geo_lat}`, `{geo_lon}`")
                
                c_map1, c_map2 = st.columns([3, 1])
                with c_map2:
                    map_size = st.selectbox("Taille point", [None] + num_cols, key="m_s")
                    map_col = st.selectbox("Couleur point", [None] + cat_cols, key="m_c")
                    style = st.selectbox("Style", ["open-street-map", "carto-positron", "white-bg"])
                
                with c_map1:
                    fig = px.scatter_mapbox(
                        df, lat=geo_lat, lon=geo_lon, 
                        size=map_size, color=map_col,
                        zoom=1, mapbox_style=style, height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # CAS 2 : Pas de Lat/Lon mais Adresses -> Analyse par Zone
            elif len(geo_addr_cols) > 0:
                st.info(f"📍 Adresses détectées : `{', '.join(geo_addr_cols)}` (Affichage analytique)")
                
                c_geo1, c_geo2 = st.columns(2)
                
                # Configuration commune
                loc_col = st.selectbox("Choisir la colonne Lieu", geo_addr_cols, key="geo_loc")
                metric_col = st.selectbox("Métrique à analyser", num_cols, key="geo_met") if num_cols else None
                
                if metric_col:
                    # Viz 1 : Treemap (Vue d'ensemble hiérarchique)
                    with c_geo1:
                        st.markdown("#### 🗺️ Poids par Zone (Treemap)")
                        fig_tree = px.treemap(df, path=[loc_col], values=metric_col)
                        fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                        st.plotly_chart(fig_tree, use_container_width=True)
                    
                    # Viz 2 : Top Lieux (Bar Chart Horizontal)
                    with c_geo2:
                        st.markdown(f"#### 🏆 Top 15 {loc_col}")
                        df_loc = df.groupby(loc_col)[metric_col].sum().reset_index()
                        df_loc = df_loc.sort_values(by=metric_col, ascending=False).head(15)
                        
                        fig_bar = px.bar(
                            df_loc, x=metric_col, y=loc_col, 
                            orientation='h', color=metric_col,
                            color_continuous_scale="Viridis"
                        )
                        # Inverser l'axe Y pour avoir le 1er en haut
                        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bar, use_container_width=True)
            
            # CAS 3 : Rien du tout
            else:
                st.warning("Aucune donnée géographique détectée (Lat/Lon ou Ville/Pays/Adresse).")

        # TAB 4
        with tabs[3]:
            if VIZ_DIR.exists():
                imgs = sorted(list(VIZ_DIR.glob("*.png")))
                if imgs:
                    c1, c2 = st.columns(2)
                    for i, img in enumerate(imgs):
                        with (c1 if i%2==0 else c2):
                            with st.container(border=True):
                                st.markdown(f"**{clean_viz_title(img.name)}**")
                                st.image(str(img), use_container_width=True)
                else: st.info("Pas d'images.")

        # TAB 5
        with tabs[4]:
            if REPORT_FILE.exists():
                with open(REPORT_FILE, "r", encoding="utf-8") as f: st.markdown(f.read())

        # TAB 6
        with tabs[5]:
            st.subheader(f"Données : {TABLE_ID}")
            if cat_cols:
                f_c = st.selectbox("Filtre", ["Tout"]+cat_cols, key="r_f")
                if f_c != "Tout":
                    val = st.selectbox("Valeur", df[f_c].unique(), key="r_v")
                    df_view = df[df[f_c] == val]
                else: df_view = df
            else: df_view = df
            
            st.dataframe(df_view, use_container_width=True)
            csv = df_view.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV", csv, "export.csv", "text/csv")