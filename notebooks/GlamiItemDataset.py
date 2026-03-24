import torch
from torch.utils.data import Dataset
import pandas as pd
import math

class GlamiItemDataset(Dataset):
    def __init__(self, df, vocab_manager, images_dir,price_scaled = True):
        """
        df: Pandas DataFrame s vyčištěnými daty (výstup ze scikit-learn pipeline)
        vocab_manager: Náš načtený GlamiVocabularyManager
        images_dir: Cesta ke složce s obrázky
        """
        self.df = df.reset_index(drop=True)
        self.vocab = vocab_manager
        self.images_dir = images_dir
        self.price_scaled = price_scaled
        
    def __len__(self):
        return len(self.df)

    def _to_multi_hot(self, ids_string, vocab_dict):
        """
        Bezpečně převede string (např. "14, 25") na multi-hot tenzor.
        vocab_dict obsahuje mappingy z labelu na id pochazi z GlamiVocabularyManageru
        """
        vocab_size = len(vocab_dict)
        tensor = torch.zeros(vocab_size, dtype=torch.float32)
        
        if pd.isna(ids_string) or str(ids_string).strip() == "":
            return tensor
            
        raw_ids = [i.strip() for i in str(ids_string).split(',') if i.strip()]
        
        # Přiřazení jedniček v tenzoru, pokud id neni v vocab_dict, ignorujeme ji
        for raw_id in raw_ids:
            if raw_id in vocab_dict:
                idx = vocab_dict[raw_id]
                tensor[idx] = 1.0
                
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Cena (už je škálovaná z pipeline)
        if self.price_scaled:
            price_tensor = torch.tensor([row['price_scaled']], dtype=torch.float32)
        else:
            price_tensor = torch.tensor([row['price_eur']], dtype=torch.float32)
        
        # Geo (použijeme .get s defaultní hodnotou 0, kdyby přišlo neznámé geo)
        geo_idx = self.vocab.geo_dict.get(row['geo'], 0)
        geo_tensor = torch.tensor(geo_idx, dtype=torch.long)
        
        # --- MULTI-HOT VEKTORY ---
        
        # Zde reálně probíhá ta práce s tvými dicty!
        colors_tensor = self._to_multi_hot(row['colorTagIdsString'], self.vocab.color_dict)
        depts_tensor = self._to_multi_hot(row['departmentIds'], self.vocab.dept_dict)
        
        return {
            "item_id": row['itemId'], # Vždy dobré vracet pro kontrolu
            "price": price_tensor,
            "geo": geo_tensor,
            "colors": colors_tensor,
            "departments": depts_tensor
        }