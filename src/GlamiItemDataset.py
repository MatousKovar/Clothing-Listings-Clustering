import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class GlamiItemDataset(Dataset):
    def __init__(self, df, vocab_manager, tokenizer, max_len=128,
                 embeddings_dict=None,     # Pro TRAIN mód
                 images_dir=None,          # Pro PREDICT mód
                 clip_model=None,          # Pro PREDICT mód
                 clip_processor=None,      # Pro PREDICT mód
                 device="cpu"):
        
        self.df = df.reset_index(drop=True)
        self.vocab = vocab_manager
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.embeddings_dict = embeddings_dict
        self.images_dir = images_dir
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

    def __len__(self):
        return len(self.df)

    def _to_multi_hot(self, ids_string, vocab_dict):
        vocab_size = len(vocab_dict)
        tensor = torch.zeros(vocab_size, dtype=torch.float32)
        if pd.isna(ids_string) or str(ids_string).strip() == "":
            return tensor
        raw_ids = [i.strip() for i in str(ids_string).split(',') if i.strip()]
        for raw_id in raw_ids:
            if raw_id in vocab_dict:
                tensor[vocab_dict[raw_id]] = 1.0
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = str(row['itemId'])
        
        price_tensor = torch.tensor([row['price_scaled']], dtype=torch.float32)
        geo_tensor = torch.tensor(self.vocab.geo_dict.get(row['geo'], 0), dtype=torch.long)
        colors_tensor = self._to_multi_hot(row['colorTagIdsString'], self.vocab.color_dict)
        depts_tensor = self._to_multi_hot(row['departmentIds'], self.vocab.dept_dict)
        
        text = f"{row['title']} {row['description']}"
        encoded = self.tokenizer(text, max_length=self.max_len, padding='max_length', 
                                 truncation=True, return_tensors='pt')
        
        if self.embeddings_dict is not None:
            image_embedding = self.embeddings_dict.get(item_id, torch.zeros(768, dtype=torch.float32))
            
            if image_embedding.device.type != 'cpu':
                image_embedding = image_embedding.cpu()

        else:
            img_path = os.path.join(self.images_dir, f"{item_id}.jpg")
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    # Výstup je [1, 768], přes squeeze z něj uděláme [768] a hodíme na CPU
                    image_embedding = self.clip_model(**inputs).pooler_output.squeeze(0).cpu()
            except Exception as e:
                # Fallback, pokud obrázek nejde načíst
                image_embedding = torch.zeros(768, dtype=torch.float32)

        return {
            "item_id": item_id,
            "price": price_tensor,
            "geo": geo_tensor,
            "colors": colors_tensor,
            "departments": depts_tensor,
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "image_embedding": image_embedding  # Dřív to byl image_tensor [3, 224, 224]
        }