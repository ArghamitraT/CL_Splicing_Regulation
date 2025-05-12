from torch.utils.data import DataLoader, Dataset

class Dummy(Dataset):
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        return idx

loader = DataLoader(Dummy(), batch_size=32, num_workers=8)

for i, batch in enumerate(loader):
    if i > 10: break
    print(f"Batch {i}: {batch}")
