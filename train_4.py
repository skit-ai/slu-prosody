from trainer import LightningProsodyDistillationModel, get_checkpoint_earlystop

from s2idatasets.slurp import get_prosody_dataloaders as slurp_loaders
from s2idatasets.stop import get_prosody_dataloaders as stop_loaders
# from s2i_datasets.skits2i import get_baseline_dataloaders as skits2i_loaders
# from s2i_datasets.fsc import get_baseline_dataloaders as fsc_loaders

from pytorch_lightning import Trainer
import pandas as pd
import os
os.environ['WANDB_MODE'] = 'online'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    data_name = "stop"

    if data_name == "slurp":
        trainloader, valloader, testloader = slurp_loaders(batch_size=128)
        n_class = 60
    if data_name == "stop":
        trainloader, valloader, testloader = stop_loaders(batch_size=128)
        n_class = 64
    # if data_name == "skits2i":
    #     trainloader, valloader, testloader = skits2i_loaders(batch_size=128)
    #     n_class = 14
    # if data_name == "fsc":
    #     trainloader, valloader, testloader = fsc_loaders(batch_size=128)
    #     n_class = 31

    model = LightningProsodyDistillationModel(n_class=n_class)
    run_name = f"{data_name}-1-prosody_dist=final"
    logger, checkpoint_callback, early_stop_callback = get_checkpoint_earlystop(run_name)

    trainer = Trainer(
            fast_dev_run=False, 
            gpus=1, 
            max_epochs=20, 
            checkpoint_callback=True,
            callbacks=[
                checkpoint_callback,
                early_stop_callback
            ],
            logger=logger,
            )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)
    test_report = trainer.test(model, test_dataloaders=testloader, ckpt_path=checkpoint_callback.best_model_path)
    print(f"\n\n\n\nTest report for run - {run_name}\n", test_report)