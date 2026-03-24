import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class GlamiItemDataset(Dataset):
    def __init__(self, df, image_dir, geo_to_idx, max_color_id, max_dept_id, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.geo_to_idx = geo_to_idx
        self.max_color_id = max_color_id
        self.max_dept_id = max_dept_id
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def _to_multi_hot(self, ids_string, max_id):
        """vytvori vektor idcek tak """
        tensor = torch.zeros(max_id + 1, dtype=torch.float32)
        if not ids_string or ids_string == "0":
            return tensor
            
        # Parse the comma-separated string into integers
        try:
            ids = [int(i.strip()) for i in str(ids_string).split(',') if i.strip()]
            for i in ids:
                if i <= max_id:
                    tensor[i] = 1.0
        except ValueError:
            pass # Failsafe for weird corrupted strings
            
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Numerical Features (Scaled Price)
        price_scaled = torch.tensor([row['price_scaled']], dtype=torch.float32)
        price_euro = torch.tensor([row['price_eur']], dtype=torch.float32)
        
        # 2. Single Categorical (Geo)
        # Default to 0 if geo is unknown
        geo_idx = self.geo_to_idx.get(row['geo'], 0) 
        geo_tensor = torch.tensor(geo_idx, dtype=torch.long)
        
        # MULTI CATHEGORICAL
        color_tensor = self._to_multi_hot(row['colorTagIdsString'], self.max_color_id)
        dept_tensor = self._to_multi_hot(row['departmentIds'], self.max_dept_id)
        
        # LOAD IMAGE
        image_path = os.path.join(self.image_dir, f"{row['itemId']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB') 
        except Exception as e:
            # Failsafe: if image is missing, return a blank black image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        # pro potreby croppingu, nebo rotace
        if self.transform:
            image = self.transform(image)
            
        # TODO tokenizace
        title = str(row['title'])
        description = str(row['description'])
            
        return {
            "item_id": row['itemId'],
            "price_scaled": price_scaled,
            "geo_idx": geo_tensor,
            "colors_multi_hot": color_tensor,
            "depts_multi_hot": dept_tensor,
            "image": image,
            "title": title,
            "description": description
        }