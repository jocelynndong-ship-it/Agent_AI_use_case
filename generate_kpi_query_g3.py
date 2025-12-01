#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import google.generativeai as genai
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

# Configuration Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from dotenv import set_key, find_dotenv, load_dotenv
    load_dotenv()
except ImportError:
    logger.error("❌ Installez python-dotenv")
    sys.exit(1)

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID_DEFAULT = os.getenv("DATASET_ID", "agent_dataset")
# Nouveau : Fichier dédié pour transmettre la requête
QUERY_FILE = Path("results/generated_query.sql")

class SQLGeneratorAgent:
    def __init__(self):
        self._setup_clients()

    def _setup_clients(self):
        if not PROJECT_ID:
            logger.critical("❌ PROJECT_ID manquant")
            sys.exit(1)
        try:
            self.bq_client = bigquery.Client(project=PROJECT_ID)
        except Exception as e:
            logger.critical(f"❌ Erreur Init BigQuery : {e}")
            sys.exit(1)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.critical("❌ GOOGLE_API_KEY manquant")
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.genai_model = genai.GenerativeModel("gemini-2.5-flash")

    def clean_sql(self, text: str) -> str:
        """Nettoie le markdown."""
        pattern = r"```(?:sql|bigquery)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sql = (match.group(1) if match else text).strip()
        return sql

    def sanitize_numeric_literals(self, sql: str) -> str:
        """Retire les virgules des nombres littéraux (ex: 1,000 -> 1000)."""
        return re.sub(r'(\d),(\d{3})', r'\1\2', sql)

    def get_schema_info(self, table_id: str) -> Optional[str]:
        try:
            table = self.bq_client.get_table(table_id)
            return "\n".join([f"- `{f.name}` ({f.field_type})" for f in table.schema])
        except Exception as e:
            logger.error(f"Erreur schéma {table_id}: {e}")
            return None

    def validate_with_dry_run(self, query: str) -> Tuple[bool, str]:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.bq_client.query(query, job_config=job_config)
            gb = job.total_bytes_processed / (1024**3)
            return True, f"Volume: {gb:.5f} GB"
        except BadRequest as e:
            return False, e.message
        except Exception as e:
            return False, str(e)

    def generate(self, table_id: str, context_intent: str = "") -> Optional[str]:
        schema_str = self.get_schema_info(table_id)
        if not schema_str: return None

        prompt = f"""
        Tu es Expert SQL BigQuery.
        TABLE: `{table_id}`
        SCHEMA: {schema_str}
        INTENT: "{context_intent}"

        RÈGLES STRICTES DE NETTOYAGE :
        1. **TYPES STRING** : Si une colonne numérique (Prix, Ventes, Quantité) est de type STRING :
           - Tu DOIS utiliser : `SAFE_CAST(REGEXP_REPLACE(colonne, r'[^0-9.-]', '') AS FLOAT64)`
           - Cela gère les `$`, `,`, et les valeurs vides.
        
        2. **FORMATAGE** :
           - NE JAMAIS mettre de virgule dans les nombres (ex: `WHERE x > 1000` est OK, `1,000` est INTERDIT).
           - Utilise des backticks ` ` pour tous les champs.
        
        STRUCTURE :
        - SELECT Dimensions, Agrégations (SUM, AVG...)
        - FROM table
        - GROUP BY Dimensions
        - ORDER BY Metrics DESC
        - LIMIT 1000

        Renvoie uniquement le SQL.
        """
        
        logger.info(f"🧠 Génération SQL pour : {table_id}")
        
        for attempt in range(3):
            try:
                resp = self.genai_model.generate_content(prompt)
                sql = self.sanitize_numeric_literals(self.clean_sql(resp.text))
                valid, msg = self.validate_with_dry_run(sql)
                if valid:
                    logger.info(f"✅ SQL Validé ({msg})")
                    return sql
                else:
                    logger.warning(f"⚠️ SQL Invalide ({msg})")
                    prompt += f"\n\nERREUR: {msg}. Corrige le SQL et attention aux conversions STRING -> FLOAT."
            except Exception as e:
                logger.error(f"Erreur API: {e}")
                break
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", help="ID Table BigQuery", default=None)
    parser.add_argument("--intent", help="Contexte", default="")
    args = parser.parse_args()

    agent = SQLGeneratorAgent()
    
    table_id = args.table
    if not table_id:
        table_id = f"{PROJECT_ID}.{DATASET_ID_DEFAULT}.sales_data2"

    final_query = agent.generate(table_id, args.intent)

    if final_query:
        print(f"\n✅ SQL GÉNÉRÉ :\n{final_query}")
        
        # Sauvegarde Fichier SQL
        QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_FILE, "w", encoding="utf-8") as f:
            f.write(final_query)
        logger.info(f"💾 Requête sauvegardée dans : {QUERY_FILE}")
        
        # Sauvegarde .env (Placeholder)
        try:
            env_path = find_dotenv() or Path(".env")
            set_key(env_path, "QUERY", "Voir results/generated_query.sql")
        except: pass
    else:
        logger.error("❌ Échec génération SQL.")
        sys.exit(1)

if __name__ == "__main__":
    main()