# s2i-prosody
This repository is the official implementation of the paper [Improving End-to-End SLU performance with Prosodic Attention and Distillation]() (submitted at Interspeech 2023). 
<p align="center"><img src="imgs/method.png" width="50%"/></p>

**Abstract:**
Most End-to-End SLU methods depend on the pretrained ASR or language model features for intent prediction. However, other essential information in speech, such as prosody, is often ignored. Recent research has shown improved results in classifying dialogue acts by incorporating prosodic information. The margins of improvement in these methods are minimal as the neural models ignore prosodic features. In this work, we propose prosody-attention, which uses the prosodic features differently to generate attention maps across time frames of the utterance. Then we propose prosody-distillation to explicitly learn the prosodic information in the acoustic encoder rather than concatenating the implicit prosodic features. Both the proposed methods improve the baseline results, and the prosody-distillation method gives an intent classification accuracy improvement of 8\% and 2\% on SLURP and STOP datasets over the prosody baseline.


# Installation 

```bash
git clone https://github.com/skit-ai/s2i-prosody
cd s2i-prosody

conda create -n env_name --file requirements.txt
conda activate env_name
```


To extract and save the prosodic features into ```.pt``` files, add the path to the original dataset after downloading in the scripts.
```bash
# SLURP Dataset
python data_preparation/slurp.py

# STOP Dataset
python data_preparation/stop.py

```

# Training and Evaluation
Change the dataset, training method and training hyperparamerters in ```train.py```.

Datasets:
- SLURP
- STOP

Training Methods:
- Baseline Whisper
- Baselines Prosody Concat
- Prosody-Attention
- prosody-Distillation

```train.py``` also returns the evaluation metrics after training.

```bash
python train.py
```

# Visualization of Attention Map
Code for visualization of the attention maps of the pretrained models are given in ```visualize_attention.ipynb```
<p align="center"><img src="imgs/attentionmap.png" width="50%"/></p>

# Results
<p align="center"><img src="imgs/t1.png" width="50%"/></p>

<p align="center"><img src="imgs/t2.png" width="50%"/></p>

<p align="center"><img src="imgs/t3.png" width="50%"/></p>


# Citation
```

```

# References
1. [Skit-S2I: An Indian Accented Speech to Intent dataset](https://arxiv.org/abs/2212.13015)
2. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - [Pretrained model checkpoint](https://github.com/openai/whisper) 
3. [A neural prosody encoder for end-ro-end dialogue act classification](https://arxiv.org/abs/2205.05590)
4. [SLURP: A Spoken Language Understanding Resource Package](https://arxiv.org/abs/2011.13205)
5. [STOP: A dataset for Spoken Task Oriented Semantic Parsing](https://arxiv.org/abs/2207.10643)