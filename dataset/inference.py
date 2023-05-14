import os
import pandas as pd

import torch
import torch.nn as nn
import torchaudio

import whisper
import numpy as np

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
        intent_str = d["intent_str"]

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
        return mel, prosodic_features, intent_class, wav_path, intent_str

    def collate_fn(self, batch):
        (seq, prosody, label) = zip(*batch)
        data = torch.stack([x.reshape(80, -1) for x in seq])
        prosody = torch.stack([x.reshape(-1, 6) for x in prosody])
        label = torch.tensor(list(label))
        return data, prosody, label