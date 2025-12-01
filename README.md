
🤖 Multi-Agent AI Data Platform (BigQuery + Gemini)

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue)
![alt text](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![alt text](https://img.shields.io/badge/Google-BigQuery-4285F4)
![alt text](https://img.shields.io/badge/AI-Gemini%20Flash-8E75B2)

Une plateforme d'analyse de données autonome pilotée par des Agents IA.
Ce projet orchestre plusieurs scripts Python pour automatiser le pipeline de données : de l'ingestion (ETL) à la génération de requêtes SQL complexes, en passant par le Reporting BI interactif et le Machine Learning automatisé (Auto-ML).



-----------------------------------------
🚀 Fonctionnalités Clés

🕵️‍♂️ Agent SQL Génératif : Transforme le langage naturel en SQL BigQuery optimisé. Nettoie automatiquement les formats financiers ($, ,) et gère les erreurs de syntaxe.

🔄 Pipeline ETL & Visualisation : Exécute les requêtes, nettoie les données (gestion des types pd.NA, dates), et génère une galerie de graphiques statiques (Matplotlib/Seaborn).

📊 Dashboard BI Intelligent :

Détection automatique des coordonnées géographiques (Cartes) ou des noms de lieux (Treemaps).

Graphiques interactifs (Plotly).

Système de cache intelligent (rechargement auto si la table change).

-----------------------------------------

🔮 Auto-ML Lab (V7.1) :

Benchmark automatique de modèles (Random Forest, SVM, XGBoost, etc.).

Modes Classification, Régression et Clustering.

Analyse sémantique par l'IA pour suggérer la meilleure cible (Target) à prédire.

-----------------------------------------

📂 Architecture du Projet

Le système est modulaire. Chaque script agit comme un agent spécialisé :

Fichier	Rôle	Description

main.py	🎮 Chef d'Orchestre	Point d'entrée (CLI). Gère le menu, l'upload CSV vers BigQuery et lance les agents.

generate_kpi_query_g3.py	🧠 Agent SQL	Analyse le schéma BigQuery et génère/corrige le SQL via Gemini. Sauvegarde dans generated_query.sql.

run_pipeline_g3.py	⚙️ Orchestrateur	Exécute l'agent pipeline, gère le nettoyage des anciens fichiers et la sauvegarde Parquet.

pipeline_agent.py	🎨 Agent Viz	Exécute la requête, nettoie les données (Robustness) et crée les visuels statiques + rapport Markdown.

dashboard.py	📊 Interface BI	Dashboard Streamlit complet (KPIs, Onglets dynamiques, Géospatial).

app3.py	🧪 Agent ML	Interface Auto-ML pour l'entraînement de modèles et l'analyse prédictive.


-----------------------------------------
🛠️ Pré-requis

Google Cloud Platform (GCP) :

Un projet actif.

BigQuery API activée.

Un fichier de clé de service (JSON) ou une authentification locale (gcloud auth application-default login).

-----------------------------------------
Gemini API :
Une clé API valide (Google AI Studio).
Python 3.9+

-----------------------------------------
📦 Installation
Cloner le dépôt :
git clone https://github.com/votre-user/votre-repo.git
cd votre-repo

-----------------------------------------
Créer un environnement virtuel :
python -m venv .venv
source .venv/bin/activate  # Mac/Linux


-----------------------------------------
Installer les dépendances :
pip install -r requirements.txt

Configurer les variables d'environnement :
Créez un fichier .env à la racine du projet :

PROJECT_ID=votre-projet-gcp-id
DATASET_ID=agent_dataset
GOOGLE_API_KEY=votre-cle-api-gemini


-----------------------------------------
▶️ Utilisation
Lancez simplement le script principal pour démarrer l'assistant :
python main.py


Vous aurez accès au menu interactif :

📂 Charger un fichier CSV local : Upload instantané vers BigQuery (création de table auto) puis lancement de l'analyse.

🗄️ Utiliser une table existante : Analyse une table BigQuery déjà présente.

🔮 Ouvrir l'Auto-ML Agent : Accès direct au laboratoire de Machine Learning.

Une fois le pipeline terminé, choisissez l'option 1 pour ouvrir le Dashboard Streamlit dans votre navigateur.



-----------------------------------------
📦 Requirements (Dépendances)

Pour recréer le fichier requirements.txt, voici les librairies nécessaires :

streamlit

pandas

numpy

google-cloud-bigquery

google-generativeai

python-dotenv

matplotlib

seaborn

plotly

scikit-learn

pyarrow

db-dtypes



-----------------------------------------
🛡️ Robustesse & Gestion d'Erreurs

Ce projet a été conçu pour la production :

Formatage Numérique : Correction automatique des erreurs SQL type Bad double value (virgules dans les nombres).

Cache Busting : Le Dashboard détecte si la table cible a changé et invalide le cache automatiquement.

Type Safety : Conversion forcée des types pd.NA (Nullables) pour éviter les crashs de visualisation.

Clean Workspace : Suppression automatique des anciens graphiques/rapports avant chaque nouvelle exécution.


-----------------------------------------
👤 Auteur
Jocelyn NDONG - Analyst Engineer (Devoteam G Cloud)
N'hésitez pas à contribuer ou à ouvrir une issue pour toute suggestion d'amélioration !




