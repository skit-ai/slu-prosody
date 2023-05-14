import os
import pandas as pd

import torch
import torchaudio
from tqdm import tqdm

from shennong.audio import Audio
from prosody import PitchKaldiProcessor, EnergyKaldiProcessor
import whisper

pitch_processor = PitchKaldiProcessor()

def save_features(csv_path, out_path):
    wavs_dir = "/root/Speech2Intent/Datasets/STOP/stop/"
    df = pd.read_csv(csv_path, sep='\t')

    train_path =  "/root/Speech2Intent/Datasets/STOP/stop/manifests/train_filtered2.tsv"
    traindf = pd.read_csv(train_path, sep='\t')
    intent_list = sorted(list(set(traindf['Intent'])))
    intent_dict = {k: v for v, k in enumerate(intent_list)}

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        intent_str = str(row["Intent"])
        intent_class = int(intent_dict[intent_str])

        wav_path = os.path.join(wavs_dir, row["file_id"])
        intent_type = wav_path.split("/")[-2]
        file_name = intent_type + "_" + wav_path.split("/")[-1].split(".")[0] + ".pt"
        

        save_path = os.path.join(out_path, file_name)

        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten(), 5*16000)
        
        pitch = pitch_processor(Audio(wav_tensor.flatten().numpy(), 16000))
        tensors = {
            "wav_path" : wav_path,
            "pitch" : pitch,
            "intent_class" : intent_class,
            "intent_str" : intent_str
        }
        torch.save(tensors, save_path)
    print("\nCompleted set - ", csv_path)


if __name__ == "__main__":
    csv_path = "/root/Speech2Intent/Datasets/STOP/stop/manifests/train_filtered2.tsv"
    out_path = "/root/Speech2Intent/Datasets/STOP/stop/prosody/train"
    save_features(csv_path, out_path)

    csv_path = "/root/Speech2Intent/Datasets/STOP/stop/manifests/eval_filtered2.tsv"
    out_path = "/root/Speech2Intent/Datasets/STOP/stop/prosody/eval"
    save_features(csv_path, out_path)

    csv_path = "/root/Speech2Intent/Datasets/STOP/stop/manifests/test_filtered2.tsv"
    out_path = "/root/Speech2Intent/Datasets/STOP/stop/prosody/test"
    save_features(csv_path, out_path)