import os
import requests
import re
import torch
from torch.utils.data import Dataset
class DataPreparationCharacterLevel:
    chars_dict=None
    numbers_dict=None
    def __init__(self,path_name="input.txt"):
        self.data=None
        self.path_name=path_name
        self.vocab_size=0
    def prepare_data(self):
        chars = sorted(list(set(self.data)))
        counts_of_elements = {ch: self.data.count(ch) for ch in chars}
        elements_to_delete = dict(
            filter(lambda d: d[1] < 70, counts_of_elements.items()))
        elements_to_delete
        string_to_delete = "".join([e for e in elements_to_delete.keys()])
        string_to_delete = string_to_delete+'<'+'>'
        self.data = re.sub(f'[{string_to_delete}]', " ", self.data)
    
    @classmethod
    def decode(cls, numbers):
        return "".join([cls.numbers_dict[i] for i in numbers])
    
    @classmethod
    def encode(cls, text):
        return [cls.chars_dict[i] for i in text]
    def get_vocab_size(self):
        return self.vocab_size
    def split(self):
        l = len(self.data)
        return self.data[:int(l*0.98)], self.data[int(l*0.98):int(l*0.99)], self.data[int(l*0.99):]
     
    def build_datasets(self):
        file_path = os.path.join(os.path.dirname(__file__), self.path_name)
        def download():
            if not os.path.exists(file_path):
                r = requests.get("https://raw.githubusercontent.com/Poeci/project/master/pan-tadeusz.txt")
                with open(file_path,'wb') as f:
                    f.write(r.content)
            
        download()
        
        with open(file_path, 'r', encoding="utf8") as f:
            self.data=f.read().lower()
            
        self.prepare_data()
        
        train, val, test = self.split()
        chars = sorted(list(set(self.data)))
        DataPreparationCharacterLevel.chars_dict = {
            v: k for k, v in enumerate(chars)}
        DataPreparationCharacterLevel.numbers_dict = {
            k: v for k, v in enumerate(chars)}
        self.vocab_size = len(set(self.data))

        return train,val,test
                

class MickiewiczDataSetChar(Dataset):
    def __init__(self,data,len_seq) -> None:
        self.data=data
        self.len_seq=len_seq
    
    def __len__(self):
        return len(self.data)-self.len_seq
    
    def __getitem__(self, index):
        encoded = DataPreparationCharacterLevel.encode(
            self.data[index:index+self.len_seq+1])
        features = torch.tensor(encoded[0:-1])
        target = torch.tensor(encoded[1:])
        return features, target