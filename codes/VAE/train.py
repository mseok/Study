import time

import torch
import torch.nn as nn
from torch.optim import Adam

from dataloader import load_data
from dataloader import get_c_to_i
from dataloader import get_dataset_dataloader
from model import VAE

# Arguments
num_epoch = 1000
maxlen = 30
n_data = 10000
batch_size = 1000

# Define dataset
smiles_list = load_data("./smiles.txt", maxlen, n_data)
c_to_i = get_c_to_i(smiles_list)
max_idx = max(list(c_to_i.values())) + 1
dataset, data_loader = get_dataset_dataloader(maxlen, c_to_i, smiles_list,
                                              batch_size, True)

# Setting about the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and initialize
model = VAE(max_idx)
for param in model.parameters():
    if param.dim() == 1:
        continue
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal_(param)
if device != "cpu":
    model.to(device)
print(f"Model loading done at {device}")
print("Number of Parameters: ",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
print("####### Finished Loading Model #######")

# Define optimizer and loss function
optimizer = Adam(params=model.parameters())
criterion = nn.CrossEntropyLoss()
print("epoch\trec\tkl\ttotal\ttime")

for epoch in range(num_epoch):
    reconstruction_losses = []
    kl_losses = []
    total_losses = []

    data = iter(data_loader)
    done = False
    sample = next(data)
    while not done:
        model.zero_grad()
        optimizer.zero_grad()
        st = time.time()
        target = sample["seq"]
        onehot = sample["onehot"]
        if device != "cpu":
            target = target.to(device)
            onehot = onehot.to(device)
        dec_onehot, kl_loss = model(onehot)
        prev_target_shape = target.shape
        prev_onehot_shape = dec_onehot.shape
        target = target.view(-1)
        dec_onehot = dec_onehot.view(-1, dec_onehot.size(-1))
        reconstruction_loss = criterion(dec_onehot, target.long())
        kl_loss = kl_loss.mean(-1)
        total_loss = reconstruction_loss + kl_loss
        reconstruction_losses.append(reconstruction_loss)
        kl_losses.append(kl_loss)
        total_losses.append(total_loss)
        total_loss.backward()
        optimizer.step()
        et = time.time()

        sample = next(data, None)
        if sample is None:
            done = True

    total_len = len(total_losses)
    reconstruction_loss = sum(reconstruction_losses) / total_len
    kl_loss = sum(kl_losses) / total_len
    total_loss = sum(total_losses) / total_len

    msg = f"{epoch}\t"
    msg += f"{reconstruction_loss:.3f}\t{kl_loss:.3f}\t{total_loss:.3f}"
    msg += f"\t{et -st:.3f}"
    print(msg)
