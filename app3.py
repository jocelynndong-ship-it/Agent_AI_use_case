import pandas as pd
import numpy as np
import warnings
import os
import re
from dotenv import load_dotenv

# --- Streamlit & Viz ---
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cloud & AI ---
from google.cloud import bigquery
import google.generativeai as genai

# --- Scikit-Learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_squared_error, confusion_matrix, silhouette_score, 
    accuracy_score, f1_score
)
from sklearn.impute import SimpleImputer

# --- Algorithmes ---
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch

# --- Configuration ---
st.set_page_config(page_title="Auto-ML Business Agent", layout="wide", page_icon="🛡️")
load_dotenv()
warnings.filterwarnings("ignore")

# --- Dictionnaire des Algorithmes ---
ALGORITHMS = {
    "Classification": {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVC (SVM)": SVC(probability=True, kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Disc. Analysis": LinearDiscriminantAnalysis()
    },
    "Régression": {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Bayesian Ridge": BayesianRidge()
    },
    "Clustering": {
        "K-Means": KMeans(n_clusters=3, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
        "Birch": Birch(n_clusters=3)
    }
}

# --- Fonctions Utilitaires ---

def parse_env_table_id():
    """Récupère les infos de connexion depuis .env pour pré-remplir."""
    full_id = os.getenv("TABLE_ID", "")
    project = os.getenv("PROJECT_ID", "")
    dataset = os.getenv("DATASET_ID", "")
    table = ""

    if full_id:
        parts = full_id.split('.')
        if len(parts) == 3:
            project, dataset, table = parts
        elif len(parts) == 1:
            table = parts[0]
            
    return project, dataset, table

def get_data_from_bigquery(project_id, dataset_id, table_id, limit=None):
    try:
        client = bigquery.Client(project=project_id)
        limit_clause = f" LIMIT {limit}" if limit else ""
        # Construction sécurisée de l'ID complet
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        query = f"SELECT * FROM `{full_table_id}`{limit_clause}"
        return client.query(query).to_dataframe()
    except Exception as e:
        st.error(f"❌ Erreur BigQuery : {e}")
        return None

def preprocess_data(df):
    """
    Nettoyage robuste des données.
    Gère les types mixtes, les dates, les infinis et les manquants.
    """
    df_processed = df.copy()
    encoders = {}
    
    # 1. Conversion Object si nécessaire (pour éviter les erreurs de type mixte)
    for col in df_processed.columns:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].astype(object)

    # 2. Gestion Dates (Conversion en timestamp Unix)
    for col in df_processed.columns:
        # On ignore ce qui est déjà numérique
        if pd.api.types.is_numeric_dtype(df_processed[col]): continue
        try:
            df_temp = pd.to_datetime(df_processed[col], errors='coerce')
            # Si la conversion a réussi pour la plupart
            if not df_temp.isnull().all():
                df_processed[col] = df_temp.astype('int64') // 10**9
                # Remplacement des valeurs < 0 (erreurs de conversion) par la médiane
                mask_valid = df_processed[col] > 0
                if mask_valid.any():
                    median_val = df_processed.loc[mask_valid, col].median()
                    df_processed[col] = df_processed[col].where(mask_valid, median_val)
                else:
                    df_processed[col] = 0
        except Exception: pass

    # 3. Imputation Numérique (Infinis et NaNs)
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    num_cols = df_processed.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        try:
            df_processed[num_cols] = imputer.fit_transform(df_processed[num_cols])
        except:
            df_processed[num_cols] = df_processed[num_cols].fillna(0)

    # 4. Encodage Catégories (LabelEncoder Robuste)
    cat_cols = df_processed.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        # Conversion en string pour unifier
        df_processed[col] = df_processed[col].astype(str).replace(['nan', 'None', '<NA>', ''], "Missing")
        try:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            encoders[col] = le
        except:
            # Si échec total, on drop
            df_processed = df_processed.drop(columns=[col])

    return df_processed, encoders

def suggest_targets_heuristic(df):
    """Suggère des cibles basées sur les maths (cardinalité)."""
    classif_targets = []
    reg_targets = []
    for col in df.columns:
        unique_count = df[col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Classification : Peu de valeurs uniques ou non-numérique
        if not is_numeric or (is_numeric and unique_count < 50):
            classif_targets.append(col)
        
        # Régression : Numérique et beaucoup de valeurs
        if is_numeric and unique_count > 5:
            reg_targets.append(col)
            
    return classif_targets, reg_targets

def analyze_target_with_llm(df):
    """Gemini devine la Target Sémantique et le Mode."""
    sample = df.head(5).to_markdown(index=False)
    cols = ", ".join(df.columns)
    
    prompt = f"""
    Analyse cet échantillon pour identifier la colonne CIBLE (Target) Business la plus probable.
    Colonnes : {cols}
    Données : {sample}
    
    Réponds STRICTEMENT au format suivant :
    TARGET: [Nom Exact Colonne]
    MODE: [Classification ou Régression]
    REASON: [Explication courte en 1 phrase]
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Extraction via Regex
        t_match = re.search(r"TARGET:\s*(.+)", text)
        m_match = re.search(r"MODE:\s*(.+)", text)
        r_match = re.search(r"REASON:\s*(.+)", text)
        
        if t_match and m_match:
            return {
                'target': t_match.group(1).strip(), 
                'mode': m_match.group(1).strip(), 
                'reason': r_match.group(1).strip() if r_match else "Suggestion IA"
            }
        return None
    except: return None

def get_model_drivers(model, feature_names):
    """Extrait les features importantes du modèle gagnant."""
    try:
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        
        if importances is not None:
            feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            return feat_imp.sort_values(by='importance', ascending=False).head(5)['feature'].tolist()
    except: return []
    return []

# --- Benchmark Engine ---

def run_benchmark(df, target_col, mode, selected_models_names):
    """
    Exécute le benchmark : Entraîne tous les modèles sélectionnés et compare.
    """
    try:
        df_clean, encoders = preprocess_data(df)
    except Exception as e:
        return None, f"Erreur Preprocessing : {str(e)}", None, None, None, None, None

    results = []
    best_model_obj = None
    best_score = -float('inf')
    best_metric_val = 0
    metric_name_display = "Metric"
    
    # 1. Séparation X et y
    if mode == "Clustering":
        X = df_clean.drop(columns=[target_col]) if target_col else df_clean
        y = None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test = X_scaled, X_scaled
        y_test = None
        y_train = None
        feature_names = list(X.columns)
        metric_name_display = "Silhouette"
    else:
        if target_col not in df_clean.columns:
            return None, "Target introuvable", None, None, None, None, None
            
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        feature_names = list(X.columns)
        
        # Scaling des Features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # CORRECTION CRITIQUE : Encodage de y AVANT le split pour Classification
        if mode == "Classification":
            le_target = LabelEncoder()
            y = y.astype(str) # Robustesse
            y = le_target.fit_transform(y)
            metric_name_display = "Accuracy"
        elif mode == "Régression":
            metric_name_display = "R2 Score"

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Boucle sur les Modèles
    total_models = len(selected_models_names)
    progress_bar = st.progress(0)
    
    for i, name in enumerate(selected_models_names):
        try:
            # Récupération de l'instance du modèle depuis le dictionnaire
            model = ALGORITHMS[mode][name]
            
            if mode == "Clustering":
                model.fit(X_train)
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
                # Silhouette score (plus proche de 1 est mieux)
                score = silhouette_score(X_train, labels) if len(set(labels)) > 1 else -1
                results.append({"Model": name, "Silhouette": score})
                metric_key = score
                metric_val = score
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if mode == "Classification":
                    score = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results.append({"Model": name, "Accuracy": score, "F1-Score": f1})
                    metric_key = score
                    metric_val = score
                else: # Régression
                    score = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    results.append({"Model": name, "R2 Score": score, "MSE": mse})
                    metric_key = score
                    metric_val = score

            # Sauvegarde du "Champion"
            if metric_key > best_score:
                best_score = metric_key
                best_model_obj = model
                best_metric_val = metric_val
                
        except Exception as e:
            results.append({"Model": name, "Error": str(e)})
        
        progress_bar.progress((i + 1) / total_models)

    if not results:
        return pd.DataFrame(), None, None, None, None, None, None

    results_df = pd.DataFrame(results)
    
    # Tri du tableau de résultats
    if mode != "Clustering":
        sort_col = "Accuracy" if mode == "Classification" else "R2 Score"
        if sort_col in results_df.columns:
            results_df = results_df.sort_values(by=sort_col, ascending=False)
            
    return results_df, best_model_obj, X_test, y_test, feature_names, encoders, (metric_name_display, best_metric_val)

def get_gemini_analysis_business(results_df, mode, best_model_name, best_model_obj, feature_names, df_sample, target_col, metric_info):
    """Analyse Business Finale par Gemini."""
    metric_name, metric_score = metric_info
    drivers = get_model_drivers(best_model_obj, feature_names)
    drivers_str = ", ".join(drivers) if drivers else "Indéterminé (Boîte Noire)"
    data_desc = df_sample.head(3).to_markdown(index=False)
    
    prompt = f"""
    Agis comme un Consultant Stratégique Senior.
    
    CONTEXTE DONNÉES : {", ".join(df_sample.columns)}
    EXEMPLE LIGNES : 
    {data_desc}
    
    RÉSULTAT AUTO-ML : 
    - Type : {mode} sur la cible '{target_col}'
    - Modèle Gagnant : {best_model_name}
    - Score ({metric_name}) : {metric_score:.3f}
    
    DRIVERS (FACTEURS D'INFLUENCE) : {drivers_str}
    
    TACHE :
    1. Analyse la fiabilité du score (Est-ce bon ? Suspect ?).
    2. Interprète les drivers : Pourquoi ces colonnes influencent-elles le résultat ? (Logique business).
    3. Donne 3 Actions concrètes pour l'entreprise basées sur ces facteurs.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Erreur IA : {e}"

# --- Interface Utilisateur (Main) ---

def main():
    st.title("🛡️ Auto-ML Business Agent")
    st.markdown("Plateforme avancée de Benchmark Machine Learning sur BigQuery.")
    
    # Initialisation Session State
    if 'ai_suggestion' not in st.session_state: st.session_state.ai_suggestion = None
    if 'data' not in st.session_state: st.session_state.data = None
    if 'results' not in st.session_state: st.session_state.results = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("Connexion BigQuery")
        
        # Auto-remplissage depuis .env
        def_project, def_dataset, def_table = parse_env_table_id()
        env_key = os.getenv("GOOGLE_API_KEY", "")

        project_id = st.text_input("Project ID", value=def_project)
        dataset_id = st.text_input("Dataset ID", value=def_dataset)
        table_id = st.text_input("Table ID", value=def_table)
        api_key = st.text_input("Gemini API Key", value=env_key, type="password")
        
        if st.button("🚀 Charger les Données"):
            if api_key and project_id and dataset_id and table_id:
                genai.configure(api_key=api_key)
                with st.spinner("Téléchargement BigQuery..."):
                    df = get_data_from_bigquery(project_id, dataset_id, table_id, limit=2000)
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.results = None
                        st.session_state.ai_suggestion = None
                        st.success(f"Chargé : {len(df)} lignes")
                        
                        # Analyse IA Initiale
                        with st.spinner("Analyse sémantique par l'IA..."):
                            st.session_state.ai_suggestion = analyze_target_with_llm(df)
                    else:
                        st.error("Aucune donnée retournée.")
            else: st.error("Informations de connexion manquantes.")

            # 👇 AJOUTEZ LE CODE ICI (Sortez du if st.button, mais restez dans with sidebar) 👇
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; font-size: 11px; color: #666;'>
                Auteur : <b>Jocelyn NDONG</b><br>
                Analyst Engineer (Devoteam G Cloud)
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Corps Principal ---
    if st.session_state.data is not None:
        df = st.session_state.data
        p_classif, p_reg = suggest_targets_heuristic(df)
        ai_sugg = st.session_state.ai_suggestion
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["📊 Données & Insights", "⚙️ Auto-ML Benchmark", "🧠 Rapport Stratégique"])
        
        # 1. Données
        with tab1:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.dataframe(df.head())
            with c2:
                if ai_sugg:
                    st.success("💡 Suggestion IA")
                    st.write(f"**Cible** : `{ai_sugg['target']}`")
                    st.write(f"**Mode** : {ai_sugg['mode']}")
                    st.caption(ai_sugg['reason'])
                else:
                    st.info("L'IA analyse vos données...")

        # 2. Auto-ML
        with tab2:
            st.subheader("Configuration du Benchmark")
            
            # Bouton Appliquer Suggestion IA
            if ai_sugg and ai_sugg['target'] in df.columns:
                col_ia, col_apply = st.columns([0.8, 0.2])
                with col_ia: 
                    st.markdown(f"🤖 L'IA recommande : **{ai_sugg['mode']}** sur la colonne **`{ai_sugg['target']}`**")
                with col_apply:
                    if st.button("✅ Appliquer Reco"):
                        st.session_state.auto_mode = ai_sugg['mode']
                        st.session_state.auto_target = ai_sugg['target']
                        st.rerun()
            st.divider()

            # Sélecteurs
            default_mode = st.session_state.get('auto_mode', "Classification")
            # Index safe
            valid_modes = ["Classification", "Régression", "Clustering"]
            m_idx = valid_modes.index(default_mode) if default_mode in valid_modes else 0
            
            mode = st.radio("Type d'Analyse", valid_modes, index=m_idx, horizontal=True)
            
            c1, c2 = st.columns(2)
            with c1:
                forced = st.session_state.get('auto_target', None)
                target = None
                
                # Listes intelligentes de cibles
                if mode == "Classification":
                    opts = p_classif if p_classif else df.columns
                    idx = list(opts).index(forced) if forced in opts else 0
                    target = st.selectbox("Colonne Cible (Target)", opts, index=idx)
                elif mode == "Régression":
                    opts = p_reg if p_reg else df.columns
                    idx = list(opts).index(forced) if forced in opts else 0
                    target = st.selectbox("Colonne Cible (Target)", opts, index=idx)
                else:
                    target = st.selectbox("Exclure une colonne (Optionnel)", [None]+list(df.columns))

            with c2:
                # Choix multiple des algos
                avail_algos = list(ALGORITHMS[mode].keys())
                sel_algos = st.multiselect("Algorithmes à comparer", avail_algos, default=avail_algos[:3])

            if st.button("🚀 Lancer le Benchmark", type="primary"):
                with st.spinner(f"Entraînement de {len(sel_algos)} modèles en cours..."):
                    # Appel du moteur
                    res_tuple = run_benchmark(df, target, mode, sel_algos)
                    
                    if res_tuple[0] is None: 
                        st.error(res_tuple[1]) # Afficher erreur
                    else:
                        st.session_state.results = res_tuple
                        st.session_state.mode = mode
                        st.session_state.target = target
                        st.success("Benchmark terminé !")

        # 3. Rapport
        with tab3:
            if st.session_state.get('results'):
                # Déballage des résultats
                if len(st.session_state.results) != 7:
                    st.error("Données de résultats invalides. Veuillez relancer.")
                    st.stop()
                    
                res_df, best_model, X_test, y_test, feat_names, _, metric_info = st.session_state.results
                
                st.subheader("🏆 Leaderboard des Modèles")
                st.dataframe(res_df.style.highlight_max(axis=0, subset=[metric_info[0]], color='#d1e7dd'))
                
                c_ia, c_viz = st.columns([2, 1])
                
                with c_ia:
                    st.markdown("### 🧠 Analyse Stratégique")
                    if st.button("Générer le rapport Consultant (Gemini)"):
                        with st.spinner("Rédaction du rapport..."):
                            analysis = get_gemini_analysis_business(
                                res_df, st.session_state.mode, 
                                res_df.iloc[0]['Model'], best_model, 
                                feat_names, df, st.session_state.target, metric_info
                            )
                            st.session_state.analysis_text = analysis
                    
                    if st.session_state.get('analysis_text'):
                        st.markdown(st.session_state.analysis_text)
                
                with c_viz:
                    m_name, m_val = metric_info
                    st.markdown("### 📊 Performance")
                    st.metric(f"Score ({m_name})", f"{m_val:.3f}")
                    
                    # Barre de progression visuelle
                    prog_val = max(0.0, min(1.0, m_val)) if m_val > -1 else 0
                    st.progress(prog_val)
                    
                    st.markdown("#### 🔑 Facteurs Clés (Drivers)")
                    drivers = get_model_drivers(best_model, feat_names)
                    if drivers:
                        for d in drivers: st.text(f"• {d}")
                    else:
                        st.caption("Non applicable pour ce modèle")

if __name__ == "__main__":
    main()