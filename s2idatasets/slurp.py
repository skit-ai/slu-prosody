import os
import pandas as pd

import torch
import torch.nn as nn
import torchaudio

import whisper
import numpy as np


def get_mel_dataloaders(batch_size=16, n_workers=5):
    train_dataset = SLURPMelDataset(
            "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/train2.csv"
        )

    val_dataset = SLURPMelDataset(
        "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/dev2.csv"
    )

    trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )
    
    valloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )

    test_dataset = SLURPMelDataset(
            "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/test2.csv"
        )
    testloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )
    
    return trainloader, valloader, testloader

def get_prosody_dataloaders(batch_size=16, n_workers=5):
    train_dataset = SLURPProsodyDataset(
            "/root/Speech2Intent/Datasets/SLURP/prosody/train"
        )

    val_dataset = SLURPProsodyDataset(
        "/root/Speech2Intent/Datasets/SLURP/prosody/dev"
    )

    trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )
    
    valloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )

    test_dataset = SLURPProsodyDataset(
            "/root/Speech2Intent/Datasets/SLURP/prosody/test"
        )
    testloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            num_workers=n_workers,
            collate_fn = train_dataset.collate_fn,
        )
    
    return trainloader, valloader, testloader


class SLURPMelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path=None):
        self.df = pd.read_csv(csv_path)

        train_path =  "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/train2.csv"
        traindf = pd.read_csv(train_path)
        intent_list = sorted(list(set(traindf['intent'])))
        self.intent_dict = {k: v for v, k in enumerate(intent_list)}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        wav_path = os.path.join(row["wav_path"])
        intent_class = int(self.intent_dict[str(row["intent"])])
        
        wav_tensor, _= torchaudio.load(wav_path) 

        # pad trim to 5 seconds
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten(), 16000*5)
        mel = whisper.log_mel_spectrogram(wav_tensor)

        return mel, intent_class

    def collate_fn(self, batch):
        (seq, label) = zip(*batch)
        data = torch.stack([x.reshape(80, -1) for x in seq])
        label = torch.tensor(list(label))
        return data, label


# -----------------------------------------------------------------------------------------------------------

class SLURPProsodyDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.files_list = os.listdir(dir_path)
        self.t = 3

    def __len__(self):
        return len(self.files_list)
    
    def norm(self, z):
        return (z-z.mean())/z.std()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_file_path = os.path.join(self.dir_path, self.files_list[idx])
        d = torch.load(pt_file_path)
        wav_path = d["wav_path"]
        pitch = d["pitch"]
        intent_class = d["intent_class"]

        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten(), 16000*self.t)
        mel = whisper.log_mel_spectrogram(wav_tensor)

        n_frames = self.t * 50

        # NCCF
        p1 = pitch[:n_frames, 0]

        # Log-Pitch
        p2 = np.log(pitch[:n_frames, 1])

        # Pitch derivative
        p3 = torch.gradient(pitch[:n_frames, 1])[0]

        # Energy 
        e1 = mel[:, :].sum(0)
        e1 = e1.unsqueeze(0)
        e1 = nn.AvgPool1d(2, stride=2)(e1).view(-1)

        e2 = mel[0:40, :].sum(0)
        e2 = self.norm(e2)
        e2 = e2.unsqueeze(0)
        e2 = nn.AvgPool1d(2, stride=2)(e2).view(-1)

        e3 = mel[40:80, :].sum(0)
        e3 = e3.unsqueeze(0)
        e3 = nn.AvgPool1d(2, stride=2)(e3).view(-1)

        mask = (p1 == 0).nonzero(as_tuple=True)[0]
        p2[mask] = 0
        p3[mask] = 0
        e1[mask] = 0
        e2[mask] = 0
        e3[mask] = 0
        
        p2 = self.norm(p2)
        p3 = self.norm(p3)
        e1 = self.norm(e1)
        e2 = self.norm(e2)
        e3 = self.norm(e3)

        p1[mask] = 0
        p2[mask] = 0
        p3[mask] = 0
        e1[mask] = 0
        e2[mask] = 0
        e3[mask] = 0

        prosodic_features = torch.cat([p1.view(-1, 1), p2.view(-1, 1), p3.view(-1, 1), e1.view(-1, 1), e2.view(-1, 1), e3.view(-1, 1)], 1)
        # prosodic_features = torch.cat([p1.view(-1, 1), p2.view(-1, 1), p3.view(-1, 1)], 1)
        # prosodic_features = torch.cat([e1.view(-1, 1), e2.view(-1, 1), e3.view(-1, 1)], 1)
        return mel, prosodic_features, intent_class

    def collate_fn(self, batch):
        (seq, prosody, label) = zip(*batch)
        data = torch.stack([x.reshape(80, -1) for x in seq])
        prosody = torch.stack([x.reshape(-1, 6) for x in prosody])
        label = torch.tensor(list(label))
        return data, prosody, label