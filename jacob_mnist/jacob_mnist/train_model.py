from models.model import MyAwesomeModel
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="train_loss", mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="train_loss", patience=3, verbose=True, mode="min"
)

model = MyAwesomeModel()

trainer = Trainer(max_epochs=10, precision="16-mixed", limit_train_batches=0.2, logger=WandbLogger(project="jacob_mnist"),
                   callbacks=[checkpoint_callback, early_stopping_callback])   

trainer.fit(model)

trainer.test(model)
