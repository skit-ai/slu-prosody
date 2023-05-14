import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import f1_score, accuracy_score
from model import WhisperBaselineModel, ProsodyBaselineModel, ProsodyAttentionModel, ProsodyDistillationModel

# SEED
SEED=100
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)
LR = 1e-3


# -----------------------------------------------------------------------------------------------------------
def get_checkpoint_earlystop(run_name):
    # update logger to Tensorboard if needed
    logger = WandbLogger(
        name=run_name,
        project='s2i_prosody'
    )

    model_acc_checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            monitor='val/acc', 
            mode='max',
            verbose=1,
            filename= run_name +'-epoch={epoch:02d}.ckpt')
    
    early_stop_callback = EarlyStopping(
        monitor="val/acc", 
        min_delta=0.00, 
        patience=15,
        verbose=False, 
        mode="max"
        )
    return logger, model_acc_checkpoint_callback, early_stop_callback
# -----------------------------------------------------------------------------------------------------------
class WhisperBaselineTrainer(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperBaselineModel(n_class=n_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": 1e-5},
            {"params": self.model.intent_classifier.parameters(), "lr": LR},], lr=LR)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y = y.view(-1)
        logits = self(x)        
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y = y.view(-1)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss}

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y = y.view(-1).detach().cpu().numpy().astype(int)
        y_hat = self(x)
        probs = F.softmax(y_hat, dim=1).detach().cpu().view(1, -1)
        pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
        probs = probs.numpy().astype(float).tolist()
        return {"y": y, "pred": pred}

    def test_epoch_end(self, outputs):
        true_labels = [sample["y"] for sample in outputs]
        pred_labels = [sample["pred"] for sample in outputs]

        acc = accuracy_score(true_labels, pred_labels)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        f1_macro = f1_score(true_labels, pred_labels, average='macro')

        self.log("Accuracy Score", acc)
        self.log("F1-Weighted", f1_weighted)
        self.log("F1-Macro", f1_macro)
# -----------------------------------------------------------------------------------------------------------
class ProsodyBaselineTrainer(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProsodyBaselineModel(n_class=n_class)

    def forward(self, x, p):
        return self.model(x, p)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": 1e-5},
            {"params": self.model.acoustic_proj.parameters(), "lr": LR},
            {"params": self.model.prosody_encoder.parameters(), "lr": LR},
            {"params": self.model.rnn.parameters(), "lr": LR},
            {"params": self.model.intent_classifier.parameters(), "lr": LR},
            ], 
            lr=LR)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, p, y = batch

        y = y.view(-1)
        logits = self(x, p)        
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1)

        logits = self(x, p)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss}

    def test_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1).detach().cpu().numpy().astype(int)
        y_hat = self(x, p)
        probs = F.softmax(y_hat, dim=1).detach().cpu().view(1, -1)
        pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
        probs = probs.numpy().astype(float).tolist()
        return {"y": y, "pred": pred}

    def test_epoch_end(self, outputs):
        true_labels = [sample["y"] for sample in outputs]
        pred_labels = [sample["pred"] for sample in outputs]

        acc = accuracy_score(true_labels, pred_labels)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        f1_macro = f1_score(true_labels, pred_labels, average='macro')

        self.log("Accuracy Score", acc)
        self.log("F1-Weighted", f1_weighted)
        self.log("F1-Macro", f1_macro)
# -----------------------------------------------------------------------------------------------------------------------------------------
class ProsodyAttentionTrainer(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProsodyAttentionModel(n_class=n_class)

    def forward(self, x, p):
        return self.model(x, p)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": 1e-5},
            {"params": self.model.prosody_encoder.parameters(), "lr": LR},
            {"params": self.model.self_attn.parameters(), "lr": LR},
            {"params": self.model.intent_classifier.parameters(), "lr": LR},
            ], 
            lr=LR)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, p, y = batch

        y = y.view(-1)
        logits, _ = self(x, p)        
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1)

        logits, _  = self(x, p)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss}

    def test_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1).detach().cpu().numpy().astype(int)
        y_hat, _  = self(x, p)
        probs = F.softmax(y_hat, dim=1).detach().cpu().view(1, -1)
        pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
        probs = probs.numpy().astype(float).tolist()
        return {"y": y, "pred": pred}

    def test_epoch_end(self, outputs):
        true_labels = [sample["y"] for sample in outputs]
        pred_labels = [sample["pred"] for sample in outputs]

        acc = accuracy_score(true_labels, pred_labels)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        f1_macro = f1_score(true_labels, pred_labels, average='macro')

        self.log("Accuracy Score", acc)
        self.log("F1-Weighted", f1_weighted)
        self.log("F1-Macro", f1_macro)
# -----------------------------------------------------------------------------------------------------------
class ProsodyDistillationTrainer(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProsodyDistillationModel(n_class=n_class) 

    def forward(self, x, p):
        return self.model(x, p)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": 1e-5},
                {"params": self.model.acoustic_proj.parameters(), "lr": LR},
                {"params": self.model.prosody_encoder.parameters(), "lr": LR},
                {"params": self.model.z_pool.parameters(), "lr": LR},
                {"params": self.model.p_pool.parameters(), "lr": LR},
                {"params": self.model.p_intent_classifier.parameters(), "lr": LR},
                {"params": self.model.z_intent_classifier.parameters(), "lr": LR},
            ], lr=1e-4)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss()(prediction, targets)
    
    def mse(self, prediction, targets):
        return nn.MSELoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        
        x, p, y = batch
        y = y.view(-1)

        logits_p, logits_z, z, zp, z_attn, zp_zttn = self(x, p)        

        winners = logits_z.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits_z.size(0))

        if self.current_epoch < 10:
            loss = self.loss_fn(logits_p, y)
            self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            cls_loss = self.loss_fn(logits_p, y) + self.loss_fn(logits_z, y)
            mse_loss = self.mse(z_attn, zp_zttn) + self.mse(z, zp) 

            batch_weight = F.softmax(torch.randn(2), dim=-1).to(self.device)
            loss = batch_weight[0] * cls_loss + batch_weight[1] * mse_loss 

            self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train/cls_loss', cls_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train/prosody_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1)

        _, logits, _, _, _, _ = self(x, p)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss, 
                'val_acc':acc,
                }

    def test_step(self, batch, batch_idx):
        x, p, y = batch
        y = y.view(-1).detach().cpu().numpy().astype(int)
        _, y_hat, _, _, _, _= self(x, p)
        probs = F.softmax(y_hat, dim=1).detach().cpu().view(1, -1)
        pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
        probs = probs.numpy().astype(float).tolist()
        return {"y": y, "pred": pred}

    def test_epoch_end(self, outputs):
        true_labels = [sample["y"] for sample in outputs]
        pred_labels = [sample["pred"] for sample in outputs]

        acc = accuracy_score(true_labels, pred_labels)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        f1_macro = f1_score(true_labels, pred_labels, average='macro')

        self.log("Accuracy Score", acc)
        self.log("F1-Weighted", f1_weighted)
        self.log("F1-Macro", f1_macro)



