import os
import pandas as pd

import torch
import torchaudio
from tqdm import tqdm

from shennong.audio import Audio
from prosody import PitchKaldiProcessor,  EnergyKaldiProcessor
import whisper

pitch_processor = PitchKaldiProcessor()

def save_features(csv_path, out_path):
    df = pd.read_csv(csv_path)

    train_path =  "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/train2.csv"
    traindf = pd.read_csv(train_path)
    intent_list = sorted(list(set(traindf['intent'])))
    intent_dict = {k: v for v, k in enumerate(intent_list)}

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        intent_str = str(row["intent"])
        intent_class = int(intent_dict[intent_str])

        wav_path = os.path.join(row["wav_path"])
        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten(), 5*16000)
        
        pitch = pitch_processor(Audio(wav_tensor.flatten().numpy(), 16000))

        tensors = {
            "wav_path" : wav_path,
            "pitch" : pitch,
            "intent_class" : intent_class,
            "intent_str" : intent_str
        }
        
        file_name = wav_path.split("/")[-1].split(".")[0] + ".pt"
        torch.save(tensors, os.path.join(out_path, file_name))
    print("\nCompleted set - ", csv_path)


if __name__ == "__main__":
    csv_path = "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/train2.csv"
    out_path = "/root/Speech2Intent/Datasets/SLURP/prosody/train"
    save_features(csv_path, out_path)

    csv_path = "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/dev2.csv"
    out_path = "/root/Speech2Intent/Datasets/SLURP/prosody/dev"
    save_features(csv_path, out_path)

    csv_path = "/root/Speech2Intent/s2i-unpaired-corpus/data_folder/csvs/slurp/test2.csv"
    out_path = "/root/Speech2Intent/Datasets/SLURP/prosody/test"
    save_features(csv_path, out_path)