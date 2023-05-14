from trainer import WhisperBaselineTrainer, ProsodyBaselineTrainer, ProsodyAttentionTrainer, ProsodyDistillationTrainer, get_checkpoint_earlystop

from dataset.slurp import get_prosody_dataloaders as slurp_loaders
from dataset.stop import get_prosody_dataloaders as stop_loaders

from pytorch_lightning import Trainer

import pandas as pd
import os
os.environ['WANDB_MODE'] = 'offline'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":

    '''List of data_name
    - slurp
    - stop
    '''

    '''List of training_method
    - baseline-whisper
    - baseline-prosody
    - prosody-attention
    - prosody-distillation
    '''

    data_name = "slurp"
    training_method = "baseline-whisper"
    run_name = f"{data_name}-{training_method}"
    batch_size = 2
    

    # Choosing the training dataset
    if data_name == "slurp":
        trainloader, valloader, testloader = slurp_loaders(batch_size=batch_size)
        n_class = 60
    if data_name == "stop":
        trainloader, valloader, testloader = stop_loaders(batch_size=batch_size)
        n_class = 64


    # Choosing the training method
    if training_method == "baseline-whisper":
        model = WhisperBaselineTrainer(n_class=n_class)
    if training_method == "baseline-prosody":
        model = ProsodyBaselineTrainer(n_class=n_class)
    if training_method == "prosody-attention":
        model = ProsodyAttentionTrainer(n_class=n_class)
    if training_method == "prosody-distillation":
        model = ProsodyDistillationTrainer(n_class=n_class)



    logger, checkpoint_callback, early_stop_callback = get_checkpoint_earlystop(run_name)
    trainer = Trainer(
            fast_dev_run=True, 
            gpus=1, 
            max_epochs=10, 
            checkpoint_callback=True,
            callbacks=[
                checkpoint_callback,
                early_stop_callback
            ],
            logger=logger,
            )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)
    test_report = trainer.test(model, test_dataloaders=testloader, ckpt_path=checkpoint_callback.best_model_path)
    print(f"\nTest report for run - {run_name}\n", test_report)