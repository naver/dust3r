import os 
import pickle 
from PIL import Image 
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd 

class Freiburg_dataset(Dataset):
    def __init__(self,root_dir= None,split= "train"):
        super().__init__()
        """_Args_
            root_dir(str): Directory containing the dataset 
            split(str): split type (train, test) 
        """
        self.split  = split 
        self.root_dir = os.path.join(root_dir,split)
        if self.split == "train":
            png_files = sorted(list(Path(self.root_dir).rglob("*.png")))
            data_df = pd.DataFrame({"file_path": png_files})
            data_df["seq_name"] = data_df["file_path"].apply(lambda x: Path(x).parts[-4])
            data_df["image_type"] = data_df["file_path"].apply(lambda x: Path(x).parts[-2])                                            
            data_df["image_num"] = data_df["file_path"].apply(lambda x: "_".join(Path(x).stem.rsplit("_", 2)[-2:]))
            data_pivot = data_df.pivot(index="image_num", columns="image_type", values="file_path").reset_index()
            seq_name_mapping = data_df.groupby("image_num")["seq_name"].first().reset_index()
            self.data = seq_name_mapping.merge(data_pivot, on="image_num")
                    
        if self.split == "test":
            png_files = sorted(list(Path(self.root_dir).rglob("*.png")))
            data_df = pd.DataFrame({"file_path": png_files})
            data_df["image_num"] = data_df["file_path"].apply(lambda x: "_".join(Path(x).stem.split("_")[3:5]))
            data_df["image_type"] = data_df["file_path"].apply(lambda x: Path(x).parts[-2])
            data_df["image_num"] = data_df.apply(
                lambda row: row["image_num"] + "0" if row["image_type"] == "ImagesIR" else row["image_num"], 
                axis=1
            )
            self.data = data_df.pivot(index="image_num", columns="image_type", values="file_path").reset_index()
            self.data.rename(columns={"ImagesIR": "fl_ir_aligned", "ImagesRGB": "fl_rgb"}, inplace=True)
            
    def __len__(self):
        return len(self.data)
    
    def get_seq(self,idx):
        return self.data[self.data["seq_name"] == idx]

    def get_allseq(self):
        return self.data["seq_name"].unique()
    
    def __getitem__(self, idx):
        scene = self.data.iloc[idx]
        
        # Load images
        # ir_aligned_path =scene["fl_ir_aligned"]
        # rgb_path = scene["fl_rgb"]
        # seq_name = scene["seq_name"]
        # image_num = scene["image_num"]
        return scene