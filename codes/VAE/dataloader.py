import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class MolDataset(Dataset):
    def __init__(self, maxlen, c_to_i, smiles_list):
        self.smiles_list = smiles_list
        self.maxlen = maxlen
        self.c_to_i = c_to_i
        self.sequence_list = self.encode_smiles()
        self.length_list = []
        for sequence in self.sequence_list:
            self.length_list.append(len(sequence))
        self.length_list = np.array(self.length_list)
        self.length_list = torch.from_numpy(self.length_list)
        self.maxidx = max(list(self.c_to_i.values()))

    def encode_smiles(self):
        smiles_list = self.smiles_list
        sequence_list = []
        for smiles in smiles_list:
            sequence = []
            for s in smiles:
                sequence.append(self.c_to_i[s])
            sequence_list.append(torch.from_numpy(np.array(sequence)))
        return sequence_list

    def seq_to_onehot(self, seq, c_to_i):
        seq = seq.long()
        c_to_i = list(c_to_i.values())
        onehot = torch.zeros(len(seq), len(c_to_i)+1)
        seq_idx = [i for i in range(len(seq))]
        onehot[seq_idx, seq] = 1
        return onehot

    def length_to_valid(self, length):
        valid = torch.ones(length)
        return valid

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        sample = dict()
        sample['seq'] = self.sequence_list[idx]
        sample['valid'] = self.length_to_valid(self.length_list[idx])
        sample['onehot'] = self.seq_to_onehot(sample['seq'], self.c_to_i)
        sample['max'] = self.maxidx
        return sample


def load_data(directory, maxlen, num):
    f = open(directory, 'r')
    smiles_list = []
    cnt = 0
    while cnt < num:
        line = f.readline()
        words = line.strip().split('\t')
        smiles = words[-1]
        if len(smiles) <= maxlen - 1:
            cnt += 1
            smiles_list.append(words[-1])
    f.close()
    return smiles_list


def get_c_to_i(smiles_list):
    c_to_i = dict()
    for smiles in smiles_list:
        for letter in smiles:
            if letter not in c_to_i:
                c_to_i[letter] = len(c_to_i) + 1
    return c_to_i


def adjust_smiles(smiles_list, maxlen):
    for i in range(len(smiles_list)):
        smiles_list[i] = smiles_list[i].ljust(maxlen, 'X')


def add_X(smiles_list):
    for i in range(len(smiles_list)):
        smiles_list[i] += 'X'


#Collate fn
def my_collate(batch):
    sample = dict()
    x = pad_sequence([b['seq'] for b in batch],
                     batch_first=True,
                     padding_value=0)
    sample['seq'] = x
    onehot = pad_sequence([b['onehot'] for b in batch],
                          batch_first=True,
                          padding_value=0)
    sample['onehot'] = onehot
    valid = pad_sequence([b['valid'] for b in batch],
                         batch_first=True,
                         padding_value=0)
    sample['valid'] = valid
    return sample


def get_dataset_dataloader(maxlen,
                           c_to_i,
                           smiles_list,
                           batch_size=8,
                           shuffle=True):
    dataset = MolDataset(maxlen, c_to_i, smiles_list)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=4,
                             collate_fn=my_collate)
    return dataset, data_loader


if __name__ == "__main__":
    maxlen = 30
    n_data = 10000
    batch_size = 2
    smiles_list = load_data("./smiles.txt", maxlen, n_data)
    c_to_i = get_c_to_i(smiles_list)
    print(c_to_i)
    dataset = MolDataset(maxlen, c_to_i, smiles_list)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=my_collate)
    data = iter(data_loader)
    i = 0
    while True:
        sample = next(data, None)
        i += 1
        print(sample["valid"])
        print(sample["valid"].shape)
        break
        if sample is None:
            break
        print(i)
