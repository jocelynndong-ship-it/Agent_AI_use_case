import sys
import subprocess
import os
import re
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# --- CHARGEMENT CONFIGURATION ---
load_dotenv()

# Gestion chemin .env
found_path = find_dotenv()
ENV_PATH = Path(found_path) if found_path else Path(".env")

PROJECT_ID = os.getenv("PROJECT_ID", "sandbox-jndong")
DATASET_ID = os.getenv("DATASET_ID", "agent_dataset")
DEFAULT_TABLE_ID = os.getenv("TABLE_ID", "")

# --- FONCTIONS UTILITAIRES ---

def update_env_variable(key, value):
    try:
        if not ENV_PATH.exists():
            with open(ENV_PATH, "w") as f: f.write("")
        set_key(ENV_PATH, key, value)
    except Exception as e:
        print(f"⚠️ Warning: Impossible de mettre à jour .env ({e})")

def get_full_table_id(short_or_full_name):
    if "." in short_or_full_name:
        return short_or_full_name
    return f"{PROJECT_ID}.{DATASET_ID}.{short_or_full_name}"

def sanitize_column_name(name):
    clean = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    return clean.strip('_').lower()

def upload_csv_to_bq(csv_path):
    print(f"⏳ Lecture du fichier : {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        df.columns = [sanitize_column_name(c) for c in df.columns]
        
        client = bigquery.Client(project=PROJECT_ID)
        dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
        
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            print(f"   Création du dataset {dataset_ref}...")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "EU"
            client.create_dataset(dataset)

        file_stem = Path(csv_path).stem
        short_table_name = sanitize_column_name(file_stem)
        full_table_id = f"{dataset_ref}.{short_table_name}"
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        print(f"🚀 Upload vers `{full_table_id}` en cours...")
        job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        job.result()
        
        print(f"✅ Succès ! {job.output_rows} lignes chargées.")
        return short_table_name

    except Exception as e:
        print(f"❌ Erreur upload CSV : {e}")
        return None

def run_command(command_list, step_name):
    print(f"\n{'='*60}")
    print(f"🤖 ÉTAPE : {step_name}")
    print(f"{'='*60}")
    try:
        subprocess.run(command_list, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ÉCHEC : {step_name} (Code {e.returncode})")
        return False

# --- FONCTION PRINCIPALE ---

def main():
    print(f"🔹 PROJET : {PROJECT_ID}")
    print(f"🔹 DATASET : {DATASET_ID}")
    print("\n--------------------------------------------------")
    print("       🤖  MENU PRINCIPAL MULTI-AGENTS  🤖")
    print("--------------------------------------------------")
    
    # Option 1
    print("1. 📂 Charger un fichier CSV local (ETL + Analyse)")
    
    # Option 2 avec Affichage Dynamique du Défaut
    if DEFAULT_TABLE_ID:
        print(f"2. 🗄️  Utiliser une table BigQuery existante (Défaut : {DEFAULT_TABLE_ID})")
    else:
        print("2. 🗄️  Utiliser une table BigQuery existante")
        
    # Option 3
    print("3. 🔮 Ouvrir l'Auto-ML Agent (Direct)")
    print("--------------------------------------------------")
    
    choice = input("👉 Votre choix (1, 2 ou 3) : ").strip()
    
    # --- OPTION 3 : ACCÈS DIRECT AUTO-ML ---
    if choice == "3":
        print("\n🚀 Lancement de l'Auto-ML Agent...")
        try:
            # On passe les variables d'environnement au sous-processus pour être sûr
            subprocess.run(["streamlit", "run", "app3.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Arrêt Auto-ML.")
        sys.exit(0)

    # --- OPTIONS 1 & 2 : FLUX ANALYTIQUE (AGENTS) ---
    short_table_name_for_env = None
    full_table_id_for_agents = None

    if choice == "1":
        # CSV -> BQ -> Agents
        raw_path = input("📂 Chemin du fichier CSV : ").strip()
        file_path = raw_path.strip('"').strip("'").strip()
        
        if os.path.exists(file_path):
            short_name = upload_csv_to_bq(file_path)
            if short_name:
                short_table_name_for_env = short_name
                full_table_id_for_agents = get_full_table_id(short_name)
            else:
                sys.exit(1)
        else:
            print(f"❌ Fichier introuvable : {file_path}")
            sys.exit(1)

    elif choice == "2":
        # BQ -> Agents
        prompt_msg = "🗄️ Nom de la table"
        if DEFAULT_TABLE_ID:
            prompt_msg += f" [Entrée pour : {DEFAULT_TABLE_ID}]"
        
        user_input = input(f"{prompt_msg} : ").strip()
        
        if user_input:
            short_table_name_for_env = user_input
            full_table_id_for_agents = get_full_table_id(user_input)
        elif DEFAULT_TABLE_ID:
            short_table_name_for_env = DEFAULT_TABLE_ID
            full_table_id_for_agents = get_full_table_id(DEFAULT_TABLE_ID)
            print(f"✅ Table par défaut utilisée : {short_table_name_for_env}")
        else:
            print("❌ ID Table manquant.")
            sys.exit(1)
            
    else:
        print("❌ Choix invalide.")
        sys.exit(1)

    if not full_table_id_for_agents:
        sys.exit(1)

    # Sauvegarde config pour la prochaine fois
    update_env_variable("TABLE_ID", short_table_name_for_env)
    print(f"💾 Config mise à jour : TABLE_ID={short_table_name_for_env}")

    # Demande Intent
    print(f"\n✅ Cible confirmée : `{full_table_id_for_agents}`")
    intent = input("\n🔎 Intention d'analyse (laisser vide pour auto) : ").strip()
    if not intent:
        intent = "Analyse exploratoire Master Table"

    # --- LANCEMENT DES AGENTS ---
    
    # 1. Agent SQL (Génération requête)
    cmd_sql = [
        sys.executable, "generate_kpi_query_g3.py", 
        "--table", full_table_id_for_agents,
        "--intent", intent
    ]
    if not run_command(cmd_sql, "Agent 1: Générateur SQL"): sys.exit(1)

    # 2. Agent Pipeline (ETL + Viz + Rapport)
    cmd_pipeline = [sys.executable, "run_pipeline_g3.py"]
    if not run_command(cmd_pipeline, "Agent 2: Pipeline ETL & Report"): sys.exit(1)

    # --- MENU FINAL ---
    print("\n✅ Analyse terminée avec succès.")
    print("--------------------------------------------------")
    print("1. 📊 Ouvrir le Dashboard (Reporting)")
    print("2. 🔮 Ouvrir l'Auto-ML Agent")
    print("3. 👋 Quitter")
    
    final_choice = input("👉 Choix : ").strip()
    
    if final_choice == "1":
        subprocess.run(["streamlit", "run", "dashboard.py"])
    elif final_choice == "2":
        subprocess.run(["streamlit", "run", "app3.py"])
    else:
        print("👋 Au revoir !")

if __name__ == "__main__":
    main()