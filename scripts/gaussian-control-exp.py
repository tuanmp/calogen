import os
from pathlib import Path

import numpy as np
import torch
from calogen.classifiers import MLP
from tqdm import tqdm

DIM=6525
NTRAIN=900000
epochs=100

def create_synthetic_data(num_samples, dim, file_name, batch_size=100000):
    """Create synthetic data for training and validation."""
    path = Path(file_name)
    if path.exists():
        print(f"File {file_name} already exists. Skipping data creation.")
        return

    os.makedirs(path.parent, exist_ok=True)
    for i in range(0, num_samples, batch_size):
        batch = np.random.randn(min(batch_size, num_samples - i), dim).astype('float32')
        if i == 0:
            fp = np.memmap(path, dtype='float32', mode='w+', shape=(num_samples, dim))
        fp[i:i+batch.shape[0]] = batch

def train(model, train_loader, val_loader, optimizer, epochs, device):
    model.train()
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            train_loss = []
            for x in tqdm(train_loader):
                optimizer.zero_grad()
                x_fake = torch.randn_like(x)
                y = torch.cat([torch.ones(x.size(0)), torch.zeros(x_fake.size(0))], dim=0).float().to(device)
                X_in = torch.cat([x, x_fake], dim=0).to(device)
                logits = model(X_in)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y.to(device))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach().cpu().item())
            train_loss = sum(train_loss) / len(train_loss)

            val_loss = []
            for x in val_loader:
                with torch.no_grad():
                    x_fake = torch.randn_like(x)
                    y = torch.cat([torch.ones(x.size(0)), torch.zeros(x_fake.size(0))], dim=0).float().to(device)
                    X_in = torch.cat([x, x_fake], dim=0).to(device)
                    logits = model(X_in)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y.to(device))
                    val_loss.append(loss.detach().cpu().item())
            
            val_loss = sum(val_loss) / len(val_loss)
            pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
            pbar.update(1)

class MemmapDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset that reads from a memory-mapped file."""
    def __init__(self, memmap_file, shape, dtype='float32'):
        self.memmap = np.memmap(memmap_file, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return self.memmap.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.memmap[idx])

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        hidden_dim=1024,
        num_layers=4,
        dropout=0.5
    ).to(device)

    print("creating memmap datasets...")
    create_synthetic_data(NTRAIN, DIM, "/pscratch/sd/p/pmtuan/control-exp/train.dat", batch_size=10000)
    create_synthetic_data(100000, DIM, "/pscratch/sd/p/pmtuan/control-exp/val.dat", batch_size=10000)

    print("creating dataloaders...")
    train_dataset = MemmapDataset("/pscratch/sd/p/pmtuan/control-exp/train.dat", (NTRAIN, DIM))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, persistent_workers=True, num_workers=1)

    val_dataset = MemmapDataset("/pscratch/sd/p/pmtuan/control-exp/val.dat", (100000, DIM))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False,  persistent_workers=True, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("training...")
    train(model, train_loader, val_loader, optimizer, epochs, device)

if __name__ == "__main__":
    main()



