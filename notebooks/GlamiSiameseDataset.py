import torch
from torch.utils.data import Dataset

class GlamiSiameseDataset(Dataset):
    def __init__(self, item_dataset, pairs_df):
        """
        item_dataset: Tvoje už fungující instance GlamiItemDataset (ta, co má v sobě i CLIP embeddingy/obrázky)
        pairs_df: DataFrame vygenerovaný funkcí create_balanced_pairs
        """
        self.item_dataset = item_dataset
        self.pairs = pairs_df.reset_index(drop=True)
        
        # Vytvoříme si rychlou "mapu", abychom věděli, na kterém řádku v item_datasetu leží jaké ID
        # Předpokládáme, že původní data mají sloupec 'itemId'
        self.id_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.item_dataset.df['itemId'])}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        idx1 = self.id_to_idx[str(row['item_id_1'])]
        idx2 = self.id_to_idx[str(row['item_id_2'])]
        
        item1 = self.item_dataset[idx1]
        item2 = self.item_dataset[idx2]
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            "item1": item1,
            "item2": item2,
            "label": label
        }