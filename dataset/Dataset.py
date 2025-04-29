import torch
import csv
from torch.utils.data import Dataset

class csv_dataset(Dataset):
  def __init__(self, path):
    with open(path,'r') as f:
      reader = csv.reader(f, delimiter='\t')
      self.data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        self.data.append(r2)
        
  def get_input_size(self):
    return len(self.data[0])-1

  def __getitem__(self, index: int):
    #print((self.data.shape()))
    #print(index)
    item = [torch.tensor(self.data[index][:-1], dtype=torch.float), torch.tensor(self.data[index][-1], dtype=torch.long)]
    return item

  def __len__(self):
    return len(self.data)
