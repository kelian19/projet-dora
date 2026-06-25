# 01_vcdb_load.py
# Chargement VCDB + diagnostic de structure
from verispy import VERIS
import pandas as pd

# Chemin vers les JSON validés du repo cloné
DATA_DIR = r"C:\Users\KélianKADDOURI\projet-dora\VCDB\data\json\validated"

v = VERIS(json_dir=DATA_DIR)
print("Schema URL utilisée :", v.schema_url)

# Construction du DataFrame (peut prendre 1-2 min sur ~10k incidents)
df = v.json_to_df(verbose=True)

print("\n--- DIAGNOSTIC ---")
print("Nombre d'incidents :", df.shape[0])
print("Nombre de colonnes  :", df.shape[1])

# Sauvegarde pour ne pas recharger à chaque fois
df.to_pickle("vcdb_df.pkl")
print("\nDataFrame sauvegardé dans vcdb_df.pkl")