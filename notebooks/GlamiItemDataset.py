"""Dataset obsahuje klicove veci:
1/ tvorba multi one hot encodingu
2/ embedding pro geo kategorie
3/ nacitani obrazku pro konkretni datovy bod
4/ embedding pro textove popisky"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import math
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import os

class GlamiItemDataset(Dataset):
    def __init__(self, df, vocab_manager, images_dir, price_scaled = True, tokenizer_name="paraphrase-multilingual-MiniLM-L12-v2", max_len=128):
        
        self.df = df.reset_index(drop=True)
        self.vocab = vocab_manager
        self.images_dir = images_dir
        self.price_scaled = price_scaled
        
        # 1. Nastavení Tokenizeru
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
        # 2. Transformace obrázků (Resize na 224x224 a Normalizace)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        
        # jednoduche transformace
        price_tensor = torch.tensor([row['price_scaled']], dtype=torch.float32)
        geo_idx = self.vocab.geo_dict.get(row['geo'], 0)
        geo_tensor = torch.tensor(geo_idx, dtype=torch.long)
        colors_tensor = self._to_multi_hot(row['colorTagIdsString'], self.vocab.color_dict)
        depts_tensor = self._to_multi_hot(row['departmentIds'], self.vocab.dept_dict)
        
        # tokenizace
        # Spojíme title a description pro lepší kontext
        text = f"{row['title']} {row['description']}"
        encoded = self.tokenizer(
            text,
            add_special_tokens=True, # text sam o sobe neobsahuje specialni tokeny pro transformer
            max_length=self.max_len, # zbytek se usekne nejde vubec do tokenizeru
            padding='max_length', # doplneni pokud je kratsi text nez max_length
            truncation=True,  
            return_attention_mask=True, # vektor obsahujici info o tom co je skutecne a co padding
            return_tensors='pt' # vraci jako pytorch tensor
        )

        # obrazek
        item_id = str(row['itemId'])
        img_path = os.path.join(self.images_dir, f"{item_id}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.img_transform(image)
        except Exception as e:
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        return {
            "item_id": item_id,
            "price": price_tensor,
            "geo": geo_tensor,
            "colors": colors_tensor,
            "departments": depts_tensor,
            # Squeeze odstraní zbytečnou dimenzi [1, max_len] -> [max_len]
            "input_ids": encoded['input_ids'].squeeze(0), # input_ids obsahuje encoding
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "image": image_tensor
        }