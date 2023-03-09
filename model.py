import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import whisper
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------
class WhisperEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = whisper.load_model("base.en").encoder
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x)

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w


# ----------------------------------------------------------------------------------------------------------------------------

class WhisperBaselineModel(nn.Module):
    def __init__(self, feature_dim=512, n_class=56):
        super().__init__()
        self.acoustic_encoder = WhisperEncoder()

        self.intent_classifier = nn.Sequential(
            nn.Linear(feature_dim, n_class),
        )

    def forward(self, x):
        z = self.acoustic_encoder(x)
        z = torch.mean(z, 1)
        intent = self.intent_classifier(z)
        return intent

# ----------------------------------------------------------------------------------------------------------------------------

class WhisperProsodyAttentionModel(nn.Module):
    def __init__(self, feature_dim=512, n_class=68):
        super().__init__()
        self.encoder = WhisperEncoder()

        self.prosody_encoder = nn.Sequential(
            nn.Conv1d(6, 128, 5,padding='same'),
            nn.GELU(),
        )

        self.self_attn = SelfAttentionPooling(128)
        self.intent_classifier = nn.Sequential(
            nn.Linear(512, n_class),
        )

    def concat_fn(self, z, p):
        return torch.cat([z, p], dim=2)
         
    def forward(self, x, p):
        
        z = self.encoder(x)
        p = self.prosody_encoder(p.transpose(1,2)).transpose(1,2)
        _, attn = self.self_attn(p)
    
        z = torch.sum(z * attn, dim=1)
        intent = self.intent_classifier(z)
        return intent, attn
        # , attn

# ----------------------------------------------------------------------------------------------------------------------------

# class WhisperProsodyConcatModel(nn.Module):
#     def __init__(self, feature_dim=512, n_class=68):
#         super().__init__()
#         self.encoder = WhisperEncoder()

#         self.acoustic_proj = nn.Sequential(
#             nn.Linear(feature_dim, 128),
#             nn.ReLU(),
#         )

#         self.prosody_encoder = nn.Sequential(
#             nn.Conv1d(6, 128, 5,padding='same'),
#             nn.GELU(),
#             nn.Conv1d(128, 128, 5,padding='same'),
#             nn.GELU(),
#             nn.Conv1d(128, 128, 5,padding='same'),
#             nn.GELU(),
#         )

#         encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.self_attn = SelfAttentionPooling(256)

#         self.intent_classifier = nn.Sequential(
#             nn.Linear(256, n_class),
#         )

#     def concat_fn(self, z, p):
#         return torch.cat([z, p], dim=2)
         
#     def forward(self, x, p):
#         z = self.encoder(x)
#         z = self.acoustic_proj(z)
#         p = self.prosody_encoder(p.transpose(1,2)).transpose(1,2)
#         z = self.concat_fn(z, p)
#         z =  self.transformer_encoder(z)
#         z, _ = self.self_attn(z)
#         intent = self.intent_classifier(z)
#         return intent

class WhisperProsodyConcatModel(nn.Module):
    def __init__(self, feature_dim=512, n_class=68):
        super().__init__()
        self.encoder = WhisperEncoder()

        self.acoustic_proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
        )

        self.prosody_encoder = nn.Sequential(
            nn.Conv1d(6, 128, 5,padding='same'),
            nn.GELU(),
            nn.Conv1d(128, 128, 5,padding='same'),
            nn.GELU(),
            # nn.Conv1d(128, 128, 5,padding='same'),
            # nn.GELU(),
        )

        # encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.rnn = nn.LSTM(256, 256, 2, batch_first=True, dropout=0.1)
        # self.self_attn = SelfAttentionPooling(6)

        self.intent_classifier = nn.Sequential(
            nn.Linear(256, n_class),
        )

    def concat_fn(self, z, p):        
        return torch.cat([z, p], dim=2)
         
    def forward(self, x, p):
        # p -> 1s -> (B, 50, 6)
        # x -> 1s -> (B, 50, 512)

        # _, attn_w = self.self_attn(p)

        z = self.encoder(x)
        z = self.acoustic_proj(z)
        p = self.prosody_encoder(p.transpose(1,2)).transpose(1,2)
        z = self.concat_fn(z, p)
        z = self.rnn(z)[0]
        
        # z = torch.sum(z * attn_w, dim=1)
        z = z[:, -1, :]
        intent = self.intent_classifier(z)
        return intent

# ----------------------------------------------------------------------------------------------------------------------------

class WhisperProsodyDistillationModel(nn.Module):
    def __init__(self, feature_dim=512, n_class=56):
        super().__init__()
        self.encoder = WhisperEncoder()

        dim = feature_dim
        self.acoustic_proj = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.GELU(),
        )

        self.prosody_encoder = nn.Sequential(
            nn.Conv1d(6, dim, 5, padding='same'),
            nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding='same'),
            nn.GELU(),
        )
        self.z_pool = SelfAttentionPooling(dim)
        self.p_pool = SelfAttentionPooling(dim)

        self.p_intent_classifier = nn.Sequential(
            nn.Linear(dim, n_class),
        )

        self.z_intent_classifier = nn.Sequential(
            nn.Linear(dim, n_class),
        )

    def forward(self, x, p):
        z = self.encoder(x)
        z = self.acoustic_proj(z) # [B, T, 128]
        z, z_attn = self.z_pool(z)

        zp = self.prosody_encoder(p.transpose(1,2)).transpose(1,2) # [B, T, 128]
        zp, zp_zttn = self.p_pool(zp)

        intent_p  = self.p_intent_classifier(zp)
        intent_z = self.z_intent_classifier(z)
               
        return intent_p, intent_z, z, zp, z_attn, zp_zttn
        # return intent_p, intent_z, z.mean(1), zp.mean(1), z_attn, zp_zttn

# ----------------------------------------------------------------------------------------------------------------------------