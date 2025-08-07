### original source from https://github.com/MyungjaeSong/Paired-Library
### code updated by Alisa Kumbara to be compatible with Python 3


import sys
import torch
import h5py
import torch.nn as nn
import pandas as pd

BASES = {'A':0, 'C':1, 'G':2, 'T':3}
   
class SeqDeepCpf1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 80, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1200, 80)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(80, 40)
        self.drop3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(40, 40)
        self.drop4 = nn.Dropout(0.3)
        self.output = nn.Linear(40, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x.transpose(-1,-2))
        x = self.drop1(x)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.relu(self.fc3(x))
        x = self.drop4(x)
        return self.output(x)
     

def PREPROCESS(path_to_txt: str):
    data = pd.read_csv(path_to_txt, sep="\t")
    seqs = data.iloc[:,1].tolist()
    N = len(seqs)
    emb_matrix = torch.zeros((N, 4, len(seqs[0])), dtype=torch.float32)
    for idx, seq in enumerate(seqs):
        for i, s in enumerate(seq):
            emb_matrix[idx, BASES[s.upper()], i] = 1
    return data, emb_matrix

def load_SeqDeepCpf1_weights(model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        w_conv = f['convolution1d_157']['convolution1d_157_W'][:]
        b_conv = f['convolution1d_157']['convolution1d_157_b'][:]
        model.conv.weight.data.copy_(torch.tensor(w_conv).squeeze(1).permute(2,1,0).flip(-1))
        model.conv.bias.data.copy_(torch.tensor(b_conv))

        model.fc1.weight.data.copy_(torch.tensor(f['dense_490']['dense_490_W'][:]).T)
        model.fc1.bias.data.copy_(torch.tensor(f['dense_490']['dense_490_b'][:]))

        model.fc2.weight.data.copy_(torch.tensor(f['dense_491']['dense_491_W'][:]).T)
        model.fc2.bias.data.copy_(torch.tensor(f['dense_491']['dense_491_b'][:]))

        model.fc3.weight.data.copy_(torch.tensor(f['dense_492']['dense_492_W'][:]).T)
        model.fc3.bias.data.copy_(torch.tensor(f['dense_492']['dense_492_b'][:]))

        model.output.weight.data.copy_(torch.tensor(f['dense_493']['dense_493_W'][:]).T)
        model.output.bias.data.copy_(torch.tensor(f['dense_493']['dense_493_b'][:]))

def main():
    print("Usage: python SeqDeepCpf1.py input.txt output.txt")
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data, SEQ = PREPROCESS(input_path)

    model = SeqDeepCpf1()

    load_SeqDeepCpf1_weights(model, 'weights/Seq_deepCpf1_weights.h5')

    model.eval()

    with torch.no_grad():
        seq_scores = model(SEQ).squeeze().tolist()

    data['Seq-deepCpf1 Score'] = seq_scores

    print(f"Saving to {output_path}")
    data.to_csv(output_path, sep="\t", index=False)

if __name__ == '__main__':
    main()
