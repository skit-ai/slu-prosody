import torch
from shennong.processor import KaldiPitchProcessor
from shennong.processor.energy import EnergyProcessor

class PitchKaldiProcessor:
    def __init__(self, ):
        options = {
                    'sample_rate': 16000,
                    'frame_shift': 0.02, 'frame_length': 0.02,
                    'min_f0': 20, 'max_f0': 500
                    }
        self.processor = KaldiPitchProcessor(**options)

    def __call__(self, audio):
        return torch.from_numpy(self.processor.process(audio).data)

class EnergyKaldiProcessor:
    def __init__(self, ):
        options = {
                    'sample_rate': 16000,
                    'frame_shift': 0.02, 'frame_length': 0.02,
                    'compression': "off"
                    }
        self.processor = EnergyProcessor(**options)

    def __call__(self, audio):
        return torch.from_numpy(self.processor.process(audio).data)