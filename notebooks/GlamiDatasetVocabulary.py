"""Soubor udrzuje natrenovane mappingy jednotlivych kategorii na indexy, pracuje s tim Dataset"""
import json
import os

class GlamiVocabularyManager:
    def __init__(self):
        # Start completely empty. No data, no files. Just placeholders.
        self.geo_dict = {} # label to numeric id
        self.color_dict = {} # label to numeric id
        self.dept_dict = {} # label to numeric id
        self.is_fitted = False

    def fit(self, df_transformed):
        """Builds the dictionaries from your training data."""
        unique_geos = sorted(df_transformed['geo'].dropna().unique().tolist())
        self.geo_dict = {geo: idx for idx, geo in enumerate(unique_geos,start=1)} # 0 is reserved for unknown

        def build_id_mapping(series):
            all_ids = series.dropna().astype(str).str.split(',').explode().str.strip()
            all_ids = all_ids[all_ids != '']
            
            unique_ids = sorted(all_ids.unique())
            
            id_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_ids)}
            
            return id_to_idx

        self.color_dict = build_id_mapping(df_transformed['colorTagIdsString'])
        self.dept_dict = build_id_mapping(df_transformed['departmentIds'])

        self.is_fitted = True
        return self

    def save(self, save_dir = "vocabularies"):
        """Saves the dictionaries as separate clean JSONs in a folder."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted vocabulary!")
            
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'geo_dict.json'), 'w') as f:
            json.dump(self.geo_dict, f)
        with open(os.path.join(save_dir, 'color_dict.json'), 'w') as f:
            json.dump(self.color_dict, f)
        with open(os.path.join(save_dir, 'dept_dict.json'), 'w') as f:
            json.dump(self.dept_dict, f)

    @classmethod
    def load_from_dir(cls, load_dir):
        """Creates a ready-to-use Manager directly from saved files."""
        manager = cls()
        with open(os.path.join(load_dir, 'geo_dict.json'), 'r') as f:
            manager.geo_dict = json.load(f)
        with open(os.path.join(load_dir, 'color_dict.json'), 'r') as f:
            manager.color_dict = json.load(f)
        with open(os.path.join(load_dir, 'dept_dict.json'), 'r') as f:
            manager.dept_dict = json.load(f)
            
        manager.is_fitted = True
        return manager