from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import os
import torch

import hydra

from src.utils import load_obj, deepconvert, resolve_wandb_entity, resolve_wandb_mode
from src.data.datamodule import ParsedDataModule
from src.data.staging import prepared_dataset_config
from src.callbacks.visualisation_callback import VisualisationCallback


def detect_accelerator():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_runtime_options(cfg_train):
    accelerator = detect_accelerator()
    has_cuda = accelerator == 'cuda'
    has_mps = accelerator == 'mps'

    gpus_cfg = cfg_train.get('gpus', 'auto')
    if gpus_cfg == 'auto':
        resolved_gpus = 1 if (has_cuda or has_mps) else 0
    elif isinstance(gpus_cfg, str) and gpus_cfg.isdigit():
        resolved_gpus = int(gpus_cfg)
    else:
        resolved_gpus = gpus_cfg

    use_amp_cfg = cfg_train.get('use_amp', 'auto')
    if use_amp_cfg == 'auto':
        resolved_use_amp = has_cuda
    elif isinstance(use_amp_cfg, str):
        resolved_use_amp = use_amp_cfg.lower() in {'1', 'true', 'yes', 'on'}
    else:
        resolved_use_amp = bool(use_amp_cfg)

    if resolved_gpus in (None, 0, '0', False):
        accelerator = 'cpu'
        resolved_gpus = 0

    return accelerator, resolved_gpus, resolved_use_amp


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(cfg):

    seed_everything(42, workers=True)

    cfg = deepconvert(cfg)

    with prepared_dataset_config(cfg['dataset'], cfg['cache_dir']) as dataset_cfg:
        cfg['dataset'] = dataset_cfg

        data_module = ParsedDataModule.load_or_create(cfg['dataset'],
                                                      cfg['cache_dir'])

        cfg['module']['len_train_ds'] = data_module.len_train_ds
        cfg['module']['len_val_ds'] = data_module.len_val_ds
        cfg['module']['len_test_ds'] = data_module.len_test_ds

        cfg['module']['input_shape'] = data_module.sample_shape_train_ds.to_tuple()[0]

        cfg_train = cfg['training']
        accelerator, resolved_gpus, resolved_use_amp = resolve_runtime_options(cfg_train)
        cfg_train['gpus'] = resolved_gpus
        cfg_train['use_amp'] = resolved_use_amp
        cfg_train['accelerator'] = accelerator
        module = load_obj(cfg['module']['class'])(cfg['module'], cfg_train)

        log_name = cfg['module']['class'] + '/' + cfg['project']
        wandb_mode = resolve_wandb_mode(cfg)
        os.environ['WANDB_MODE'] = wandb_mode
        logger = loggers.WandbLogger(
            save_dir=cfg['log_dir'],
            name=log_name,
            project=cfg['project'],
            entity=resolve_wandb_entity(cfg),
            offline=wandb_mode == 'offline',
        )

        callbacks = [
            VisualisationCallback(),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_last=True,
                save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
                filename='epoch_{epoch:02d}-step_{step}',
                auto_insert_metric_name=False)
        ]

        plugins = []
        if cfg_train.get('distr_backend') == 'ddp':
            plugins.append(DDPPlugin(find_unused_parameters=False))

        trainer_kwargs = dict(
            deterministic=True,
            logger=logger,
            callbacks=callbacks,
            plugins=plugins,
            profiler='simple',
            max_epochs=cfg_train['epochs'],
            accumulate_grad_batches=cfg_train['grad_batches'],
            precision=16 if cfg_train['use_amp'] else 32,
            auto_scale_batch_size=cfg_train.get('auto_batch_size'),
            auto_lr_find=cfg_train.get('auto_lr', False),
            check_val_every_n_epoch=cfg_train.get('check_val_every_n_epoch', 10),
            reload_dataloaders_every_epoch=False,
            fast_dev_run=cfg_train['fast_dev_run'],
            resume_from_checkpoint=cfg_train.get('from_checkpoint'),
        )

        if cfg_train.get('accelerator') == 'cuda' and cfg_train.get('gpus') not in (None, 0):
            trainer_kwargs['gpus'] = cfg_train['gpus']
        elif cfg_train.get('accelerator') == 'mps':
            trainer_kwargs['accelerator'] = 'mps'
            trainer_kwargs['devices'] = 1

        strategy = cfg_train.get('distr_backend')
        if strategy:
            trainer_kwargs['strategy'] = strategy

        trainer = Trainer(**trainer_kwargs)

        trainer.tune(module, datamodule=data_module)

        trainer.fit(module, data_module)


if __name__ == '__main__':
    main()
