from __future__ import annotations
import shutil
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import BrailleDataModule
from model import BrailleTagger


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    datamodule = BrailleDataModule()

    checkpoint = ModelCheckpoint(
        monitor="val/accuracy", mode="max", save_weights_only=True, save_top_k=1
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        precision=16,
        amp_backend="native",
        max_steps=6000,
        log_every_n_steps=40,
        val_check_interval=40,
        logger=WandbLogger("multilabel-braille", project="multilabel-braille"),
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )

    trainer.fit((BrailleTagger(n_training_steps=6000, n_warmup_steps=600)), datamodule)

    trainer.test(
        model=BrailleTagger(),
        ckpt_path=checkpoint.best_model_path,
        datamodule=datamodule,
    )
