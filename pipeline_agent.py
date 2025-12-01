import os
import io
import base64
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib

# Backend Agg pour éviter les erreurs de thread GUI
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from google.cloud import bigquery

# Configuration Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineAgent")

class OrchestratorAgent:
    def __init__(self, validator_config=None):
        self.validator_config = validator_config or {"min_rows": 1}
        self.project_id = os.getenv("PROJECT_ID")
        
        if not self.project_id:
             raise ValueError("PROJECT_ID manquant.")
             
        self.bq_client = bigquery.Client(project=self.project_id)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            raise ValueError("GOOGLE_API_KEY manquante.")

    def act(self, context):
        query = context.get("query")
        if not query:
            return {"error": "Query manquante", "failed_step": "init"}

        logger.info("1. Exécution de la requête BigQuery...")
        
        try:
            df = self.bq_client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Erreur BigQuery: {e}")
            return {"error": str(e), "failed_step": "bigquery"}

        if df.empty or len(df) < self.validator_config.get("min_rows", 1):
             return {"error": "Données insuffisantes", "failed_step": "validation"}

        # --- NETTOYAGE INTELLIGENT ---
        df = self._clean_dataframe(df)

        logger.info(f"2. Génération des visualisations ({len(df)} lignes)...")
        viz_list = self._generate_visualizations(df)

        logger.info("3. Rédaction du rapport via Gemini...")
        report = self._generate_report(df, query)

        return {
            "data": df,
            "final_report": report,
            "generated_visualizations": viz_list
        }

    def _clean_dataframe(self, df):
        """
        Nettoyage corrigé pour ne pas effacer les colonnes Textes.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

        # Copie de sécurité
        df = df.copy()

        # 1. Gestion des types Pandas récents (Int64, Float64, StringDtype, BooleanDtype)
        # Ces types contiennent des pd.NA qui font planter les visualisations.
        # On les convertit en types "Numpy-friendly" (object, float, bool)
        for col in df.columns:
            dtype_name = str(df[col].dtype)
            
            # Gestion Booléens
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(object).fillna("Unknown")
            
            # Gestion Strings modernes
            elif "String" in dtype_name:
                df[col] = df[col].astype(object)

            # Gestion Entiers nullables (Int64) -> Float64
            elif "Int" in dtype_name and df[col].hasnans:
                df[col] = df[col].astype(float)

        # 2. Gestion des Dates (CORRECTION DU BUG 'NONE')
        # On ne convertit en datetime QUE si le nom de la colonne le suggère
        date_keywords = ['date', 'time', 'timestamp', 'jour', 'day', 'month', 'year', 'annee']
        
        for col in df.columns:
            if df[col].dtype == 'object':
                col_lower = col.lower()
                # Si le nom contient un indice de date
                if any(k in col_lower for k in date_keywords):
                    try:
                        # On tente la conversion
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except: pass
                # SINON : On ne touche pas ! C'est du texte (Pays, Segment, etc.)

        # 3. Remplacement final des pd.NA restants par np.nan (compris par tout le monde)
        # On évite de toucher aux colonnes objets pures qui n'ont pas de problème
        try:
            df = df.replace({pd.NA: np.nan})
        except: pass

        return df

    def _generate_visualizations(self, df):
        """Génère un set complet de visualisations."""
        viz_list = []
        df_clean = df.dropna(axis=1, how='all')
        
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        cat_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns
        date_cols = df_clean.select_dtypes(include=['datetime', 'datetimetz']).columns
        
        sns.set_theme(style="whitegrid", palette="viridis")

        # 1. TIME SERIES
        try:
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                d_c, n_c = date_cols[0], numeric_cols[0]
                df_s = df_clean.sort_values(by=d_c).dropna(subset=[d_c, n_c])
                if len(df_s) > 100: df_s = df_s.iloc[::len(df_s)//100, :]
                
                sns.lineplot(x=d_c, y=n_c, data=df_s, marker="o", ax=ax)
                ax.set_title(f"📈 Évolution : {n_c}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                viz_list.append(self._fig_to_base64(fig, "viz_1_evolution"))
                plt.close(fig)
        except Exception as e: logger.warning(f"Viz 1 skip: {e}")

        # 2. BAR CHART
        try:
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                c_c, n_c = cat_cols[0], numeric_cols[0]
                top = df_clean.groupby(c_c)[n_c].sum().nlargest(10).reset_index()
                sns.barplot(x=n_c, y=c_c, data=top, ax=ax, hue=c_c, legend=False)
                ax.set_title(f"🏆 Top 10 {c_c}")
                plt.tight_layout()
                viz_list.append(self._fig_to_base64(fig, "viz_2_top_cat"))
                plt.close(fig)
        except Exception as e: logger.warning(f"Viz 2 skip: {e}")

        # 3. HEATMAP
        try:
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = df_clean[numeric_cols[:10]].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
                ax.set_title("🔥 Corrélation")
                plt.tight_layout()
                viz_list.append(self._fig_to_base64(fig, "viz_3_correlation"))
                plt.close(fig)
        except: pass

        # 4. BOXPLOT
        try:
            if len(numeric_cols) > 0 and len(cat_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                c_c, n_c = cat_cols[0], numeric_cols[0]
                top_cats = df_clean[c_c].value_counts().nlargest(5).index
                df_b = df_clean[df_clean[c_c].isin(top_cats)]
                sns.boxplot(x=c_c, y=n_c, data=df_b, ax=ax, hue=c_c, legend=False)
                ax.set_title(f"📦 Distribution : {n_c}")
                plt.tight_layout()
                viz_list.append(self._fig_to_base64(fig, "viz_4_distribution"))
                plt.close(fig)
        except: pass

        # 5. SCATTER
        try:
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                x, y = numeric_cols[0], numeric_cols[1]
                hue = cat_cols[0] if len(cat_cols) > 0 else None
                sns.scatterplot(data=df_clean, x=x, y=y, hue=hue, alpha=0.7, ax=ax)
                ax.set_title(f"🔗 {y} vs {x}")
                plt.tight_layout()
                viz_list.append(self._fig_to_base64(fig, "viz_5_scatter"))
                plt.close(fig)
        except: pass

        return viz_list

    def _fig_to_base64(self, fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return {"name": name, "img_base64": img_str}

    def _generate_report(self, df, query):
        try:
            summary = df.describe(include='all').to_markdown()
            head = df.head().to_markdown()
        except:
            summary = "Données brutes."
            head = "..."

        prompt = f"""
        Analyste Senior. Requête : "{query}"
        
        Données :
        {head}
        
        Stats :
        {summary}
        
        Rapport concis (Markdown) :
        1. Synthèse (3 points).
        2. Analyse Tendances/Segments.
        3. Actions.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except: return "Rapport indisponible."