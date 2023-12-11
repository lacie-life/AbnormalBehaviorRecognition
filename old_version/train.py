from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything

from tools.utilsv2 import get_loader
from tools.utilsv2 import *
from jellyfishABD.modelv2 import *
from jellyfishABD.slowfast_model import *
from tools.config import *

data_path = '/home/lacie/Datasets/KISA/ver-3/frames'

def train(config):

    loaders = {
        p: get_loader(data_path, p, config.batch_size, config.num_workers, config.dynamic_frames, config.num_frames, config.slowfast)
            for p in ['train', 'val']
    }

    wandb_logger = WandbLogger(project='video-classification')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if config.slowfast:
        model = SlowFastLitFrames(drop_prob=config.drop_prob, num_frames=config.num_frames, num_classes=config.num_classes)
    else:
        model = LitFrames(drop_prob=config.drop_prob, num_frames=config.num_frames, num_classes=config.num_classes)

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        # overfit_batches=0.05,
        callbacks=[lr_monitor],
    )
    trainer.fit(model, loaders['train'], loaders['val'])


if __name__ == "__main__":
    seed_everything(CFG.seed, workers=True)
    train(CFG)