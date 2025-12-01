#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv

try:
    from pipeline_agent import OrchestratorAgent
except ImportError:
    print("❌ Erreur : 'pipeline_agent.py' manquant.")
    sys.exit(1)

class Config:
    BASE_DIR = Path("results")
    DATA_DIR = BASE_DIR / "data"
    VIZ_DIR = BASE_DIR / "visualisations"
    REPORT_FILE = BASE_DIR / "rapport_final.md"
    DATA_FILE = DATA_DIR / "data.parquet"
    QUERY_FILE = BASE_DIR / "generated_query.sql"

    @staticmethod
    def setup_dirs():
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cleanup_workspace():
        """Supprime TOUS les anciens fichiers générés (images, data, report)."""
        print("🧹 Nettoyage de l'espace de travail...")
        cnt = 0
        
        # 1. Supprime images
        if Config.VIZ_DIR.exists():
            for p in Config.VIZ_DIR.glob("*.png"):
                try: p.unlink(); cnt+=1
                except: pass
        
        # 2. Supprime Data & Report
        for p in [Config.DATA_FILE, Config.REPORT_FILE]:
            if p.exists():
                try: p.unlink(); cnt+=1
                except: pass
                
        print(f"✨ {cnt} fichiers nettoyés.")

def save_image_from_base64(base64_string: str, output_path: Path):
    try:
        if "," in base64_string: base64_string = base64_string.split(",")[1]
        with open(output_path, "wb") as f: f.write(base64.b64decode(base64_string))
        return True
    except: return False

def main():
    load_dotenv()
    
    # Lecture SQL Fichier
    query = None
    if Config.QUERY_FILE.exists():
        query = Config.QUERY_FILE.read_text(encoding="utf-8").strip()
    
    if not query:
        print("❌ Erreur : 'results/generated_query.sql' introuvable ou vide.")
        sys.exit(1)

    # Nettoyage AVANT exécution
    Config.setup_dirs()
    Config.cleanup_workspace()

    print("🚀 Démarrage Pipeline...")
    try:
        agent = OrchestratorAgent()
        result = agent.act({"query": query})
    except Exception as e:
        print(f"❌ CRASH Pipeline : {e}")
        sys.exit(1)

    if "error" in result:
        print(f"🚫 Erreur Agent : {result['error']}")
        sys.exit(1)

    # Sauvegardes
    if result.get("final_report"):
        with open(Config.REPORT_FILE, "w", encoding="utf-8") as f: f.write(result["final_report"])
        print(f"✅ Rapport généré.")

    if result.get("data") is not None:
        df = result["data"]
        # Conversion string pour robustesse Parquet
        for col in df.select_dtypes(['object', 'category']):
            df[col] = df[col].astype(str)
        try:
            df.to_parquet(Config.DATA_FILE, index=False)
            print(f"✅ Données sauvegardées.")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde data: {e}")

    count_viz = 0
    for viz in result.get("generated_visualizations", []):
        path = Config.VIZ_DIR / f"{viz['name']}.png"
        if save_image_from_base64(viz['img_base64'], path):
            count_viz += 1
    
    print(f"✅ {count_viz} graphiques générés.")
    print("\n✅ Analyse terminée.")

if __name__ == "__main__":
    main()