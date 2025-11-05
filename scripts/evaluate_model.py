"""
Example usage:
    python3 -m scripts.evaluate_model \
        +dataset=floods_evaluation \
        ++dataset.root_folder=datasets/floods \
        +training=simple_vae \
        +normalisation=log_scale \
        +channels=high_res \
        +module=deeper_vae \
        +checkpoint=demo_assets/checkpoints/edgesat_pretrained_vae_128_small.ckpt \
        +project=edgesat_eval \
        +evaluation=vae_comprehensive \
        +name=edgesat_eval_floods
"""
from tqdm import tqdm
import hydra
import torch
import wandb
import os

from pathlib import Path
from numpy import inf
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import repeat
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from torchvision.transforms import ToPILImage
from pytorch_lightning import loggers, seed_everything

from src.data.datamodule import ParsedDataModule
from src.data.staging import prepared_dataset_config
from src.utils import load_obj, deepconvert, resolve_wandb_entity, resolve_wandb_mode
from src.evaluation.utils import get_eval_result, tassellate, as_plot
from src.models.ae_vae_models.base_vae import BaseVAE
import matplotlib.pyplot as plt


def resolve_device(cfg_train):
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    gpus_cfg = cfg_train.get('gpus', 'auto')
    if gpus_cfg == 'auto':
        wants_acceleration = True
    elif isinstance(gpus_cfg, str) and gpus_cfg.isdigit():
        wants_acceleration = int(gpus_cfg) > 0
    else:
        wants_acceleration = gpus_cfg not in (None, 0, '0', False)

    if wants_acceleration and has_cuda:
        return torch.device('cuda')
    if wants_acceleration and has_mps:
        return torch.device('mps')
    return torch.device('cpu')


TRANSFORM = ToPILImage()
def scale_original_image(image, inverse_normaliser, channels=None):
    image = inverse_normaliser(image.clone().detach())
    image[image > 2000] = 2000
    image = image / 2000
    if channels is not None:
        if len(image.shape)==3:
            image = image[channels]
        elif len(image.shape)==4:
            image = image[:, channels]
        else:
            raise NotImplementedError(f'Image shape not recognised {image.shape}')
    return TRANSFORM(image)


def __get_eval_result(method, is_vae, embs_now, data_now, recons_now,
                      embs_win, data_win, recons_win):
    return get_eval_result(method,
                           is_vae,
                           embs_now=embs_now,
                           data_now=data_now,
                           recons_now=recons_now,
                           embs_win=embs_win,
                           data_win=data_win,
                           recons_win=recons_win)


def dumb_pool_api(detection_methods, is_vae, embs_now, data_now,
                  recons_now, embs_win, data_win, recons_win):

    with Pool(1) as p:
        args = zip(detection_methods,
                   repeat(is_vae),
                   repeat(embs_now),
                   repeat(data_now),
                   repeat(recons_now),
                   repeat(embs_win),
                   repeat(data_win),
                   repeat(recons_win))

        results = p.starmap(__get_eval_result, args)
        return results


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(cfg):
    # Load configs
    cfg = deepconvert(cfg)

    ### DON'T seed_everything(42, workers=True)

    with prepared_dataset_config(cfg['dataset'], cfg['cache_dir']) as dataset_cfg:
        cfg['dataset'] = dataset_cfg

        # Load and setup the dataloaders
        data_module = ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])

        # Setting up module and loading weights
        cfg_train = cfg['training']
        cfg_module = cfg['module']
        cfg_eval = cfg['evaluation']
        cfg_module['input_shape'] = [data_module.sample_shape_train_ds.to_tuple()[0][0]] + cfg['dataset']['input_shape'] # [channels] + [input shape] ~ ex. [10]+[32,32]

        module = load_obj(cfg['module']['class'])(cfg_module, cfg_train)

        # load transformation modules
        input_trans_cls = cfg_eval.pop("input_trans_cls", "src.data.transformations.NoTransformer")
        input_trans_args = cfg_eval.pop("input_trans_args", list())
        target_trans_cls = cfg_eval.pop("target_trans_cls", "src.data.transformations.NoTransformer")
        target_trans_args = cfg_eval.pop("target_trans_args", list())

        input_transformer = load_obj(input_trans_cls)(*input_trans_args)
        target_transformer = load_obj(target_trans_cls)(*target_trans_args)

        checkpoint_root_path = Path(cfg['checkpoint']).parent.parent
        hparams_path = checkpoint_root_path / 'hparams.yaml'
        checkpoint_path = cfg['checkpoint']

        device = resolve_device(cfg_train)

        print(f'Loading checkpoint {checkpoint_path} on {device}')
        load_kwargs = dict(
            checkpoint_path=str(checkpoint_path),
            cfg=cfg_module,
            train_cfg=cfg_train,
        )
        if hparams_path.exists():
            load_kwargs['hparams_file'] = str(hparams_path)

        module = module.load_from_checkpoint(**load_kwargs)
        module = module.to(device).eval()

        log_name = cfg.get('name', None)
        wandb_mode = resolve_wandb_mode(cfg)
        os.environ['WANDB_MODE'] = wandb_mode
        logger = loggers.WandbLogger(
            save_dir=cfg['log_dir'],
            name=log_name,
            project=cfg['project'],
            entity=resolve_wandb_entity(cfg),
            offline=wandb_mode == 'offline',
        )
        logger.experiment.config.update(cfg) # store config used

        evaluate_model(module, data_module, cfg['evaluation'], cfg_module['input_shape'], logger, input_transformer, target_transformer, device)


def evaluate_model(module, data_module, cfg_evaluation, input_shape, logger, input_transformer, target_transformer, device):
    dataloaders = data_module.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = list(dataloaders)

    is_vae = isinstance(module.model, BaseVAE)
    vis_channels = cfg_evaluation['visualisation_channels']
    plot_with_colormaps = cfg_evaluation['with_colormaps']
    mask_out_clouds = cfg_evaluation['mask_out_clouds']
    
    detection_methods = [load_obj(m['method'])(**m['method_args']) 
                        for m in cfg_evaluation['detection_methods']]
    summary_stat_methods = [load_obj(m['method'])(**m['method_args']) 
                        for m in cfg_evaluation['summary_stat_methods']]

    save_plots_locally = cfg_evaluation['save_plots_locally'] # boolean flag to save plots and results locally

    print("Will save results locally:", save_plots_locally, "(otherwise these will only be sent to Wandb).")

    # IDs and methods for table made for each location sequence (ie each folder dataset)
    plot_sequences = cfg_evaluation['plot_sequences'] # boolean flag to plot sequences or not
    sequence_ids = ['Position in sequence', 'input', 'reconstruction']
    for m in detection_methods:
        sequence_ids += [m.name]
    # Each table is made inside loop ahead
        
    # IDs for image table made for all change images (each row is for different folder dataset)
    detection_method_image_names = [m.name for m in detection_methods if m.returns=='image']
    change_ids = (
        ['Dataloader #', 'image before', 'image after', 'change mask'] 
        + detection_method_image_names + ['summary stats']
    )
    change_image_table = wandb.Table(columns=change_ids)
    
    # IDs for image table made for summary statistics across all datasets
    summary_stat_method_names = [m.name for m in summary_stat_methods if m.returns=='value']
    summary_stat_ids = ['Detection method'] + summary_stat_method_names
    summary_statistics_table = wandb.Table(columns=summary_stat_ids)
    
    # Store the output of the change/anom detection methods in this for images we have 
    # ground truth for
    detection_results = []
    
    for dataloader_index, dataloader in tqdm(enumerate(dataloaders), total=len(dataloaders)):
        dataset = dataloader.dataset
        grid_shape = dataset.tiling_strategy.grid_shape
        inv_norm = dataset.datasets[0].normaliser.inverse
        total_len = len(dataset)

        if is_vae:
            embs_all = torch.zeros(total_len, 2 * module.model.latent_dim, 1, 1)
        else:
            embs_all = torch.zeros(total_len, module.model.latent_dim, 1, 1)

        data_all = torch.zeros(total_len, *input_shape)
        recons = torch.zeros(total_len, *input_shape)
        changes_all = torch.zeros(total_len, 1, *input_shape[1:])

        # Populate data, recs and embs
        n_tiles = 0
        for batch in tqdm(dataloader, leave=False, desc='Populating data'):
            batch_size = batch[0].shape[0]

            changes = torch.nan_to_num(batch[1], nan=2) #  invalid pixels given value 2
            changes = target_transformer(changes)
            data = torch.nan_to_num(batch[0])
            data = input_transformer(data)
            
            data_all[n_tiles: n_tiles+batch_size] = data
            changes_all[n_tiles: n_tiles+batch_size] = changes

            data_on_device = data.to(device)

            rec = module(data_on_device)
            if is_vae:
                rec = rec[0]
            rec = rec.detach().cpu()
            recons[n_tiles: n_tiles+batch_size] = rec
            
            emb = module.model.encode(data_on_device)
            if is_vae:
                emb = torch.cat(emb, dim=1)
            emb = emb.detach().cpu()

            embs_all[n_tiles: n_tiles+batch_size] = emb[..., None, None]

            n_tiles += batch_size
        
        assert total_len==n_tiles, f"{total_len}!={n_tiles}. Something strange is afoot"


        embs_all = tassellate(embs_all, *grid_shape, [0, 0])
        data_all = tassellate(data_all, *grid_shape, dataset.tiling_strategy.overlap)
        recons = tassellate(recons, *grid_shape, dataset.tiling_strategy.overlap)
        changes_all = tassellate(changes_all, *grid_shape, dataset.tiling_strategy.overlap)

        # Create table for logging this location only
        if plot_sequences:
            sequence_table = wandb.Table(columns=sequence_ids)
        
        # Stores just the results from the changed image, not the whole sequence
        change_results = []
        
        n_images = data_all.shape[0]
        for i in tqdm(range(n_images), desc='Analysing sequence', leave=False):

            input_image_data = scale_original_image(data_all[i], inv_norm, vis_channels)
            reconstructed_image_data = scale_original_image(recons[i], inv_norm, vis_channels)
            
            sequence_row = [f"{i}", 
                   wandb.Image(input_image_data),
                   wandb.Image(reconstructed_image_data),
                   ]

            embs_now = embs_all[i:i+1]
            data_now = data_all[i:i+1]
            recons_now = recons[i:i+1]
            change_now = changes_all[i:i+1]
            embs_win = embs_all[:i]
            data_win = data_all[:i]
            recons_win = recons[:i]

            results = dumb_pool_api(detection_methods, is_vae, embs_now,
                     data_now, recons_now, embs_win, data_win, recons_win)
                
            for method, result in zip(detection_methods, results):
                #print("Debug~", method, "=> shape:", result.shape)
                if method.returns=='image':
                    if plot_with_colormaps:
                        # Saving as a matplotlib plot:
                        results_as_plot = as_plot(result, "")
                        sequence_row += [wandb.Image(results_as_plot)]
                        plt.close()

                    else:
                        # Regular image saving:
                        sequence_row += [wandb.Image(result)]

                else:
                    sequence_row += [result]
                
                # store last changes/anom images
                if (change_now != 2).any() & (method.returns=='image'):
                    change_index = i
                    change_results += [result] # < these are saved for later visualization together ... maybe mask only these?
                   
            # Plots table
            if plot_sequences:
                sequence_table.add_data(*sequence_row)
        
        if plot_sequences:
            # Upload sequence table for this location
            logger.experiment.log({f'Sequence Qualitative results dl #{dataloader_index}': sequence_table})
        
        # Construct the summary statistics dataframe for each location
        stat_dict = {d:[] for d in detection_method_image_names}
        
        changes_in_this_location = [changes_all[change_index].squeeze()]
        for detect_name, detections in tqdm(zip(detection_method_image_names, change_results), desc='Aggregating', leave=False):
            these_detections = [detections.squeeze()]
            for stat_method in summary_stat_methods:
                stat_dict[detect_name] += [stat_method(changes_in_this_location, these_detections)]
        stat_df = pd.DataFrame(stat_dict, index=summary_stat_method_names)
        stats_as_image = wandb.Image(sns.heatmap(stat_df, annot=True, cbar=None, cmap='cool', fmt='.4g'))
        
        # Add row for change table at this location
        before_event_image = scale_original_image(data_all[change_index-1], inv_norm, vis_channels)
        after_event_image = scale_original_image(data_all[change_index], inv_norm, vis_channels)

        change_row = [dataloader_index, 
                      wandb.Image(before_event_image),
                      wandb.Image(after_event_image),
        ]

        if save_plots_locally:
            images_to_save = [dataloader_index, before_event_image, after_event_image]

        # Ground truth visualization also using the same colormap
        true_change_map = changes_all[change_index:change_index+1][0][0]

        change_results_np = change_results.copy() # We may mask these out, but only for the visualizations
        change_results_np = [np.asarray(t) for t in change_results_np]
        if mask_out_clouds:
            for x in change_results_np:
                x[true_change_map == 2] = np.nan

        if plot_with_colormaps:
            change_as_plot = as_plot(true_change_map, "")
            change_row += [wandb.Image(change_as_plot)]
            if save_plots_locally:
                images_to_save += [change_as_plot]
            plt.close()

            for x in change_results_np:
                change_as_plot = as_plot(x, "")
                change_row = change_row + [wandb.Image(change_as_plot)]
                if save_plots_locally:
                    images_to_save += [change_as_plot]
                plt.close()
        else:
            # Regular image saving:
            change_row += [wandb.Image(true_change_map)]
            if save_plots_locally:
                images_to_save += [true_change_map]

            change_row = change_row + [wandb.Image(x) for x in change_results_np]
            if save_plots_locally:
                images_to_save += [change_results_np]

        change_row = change_row + [stats_as_image]
        if save_plots_locally:
            images_to_save += [stats_as_image]        
        
        change_image_table.add_data(*change_row)

        # stores some results so we can aggregate scores over all locations
        detection_results += [changes_in_this_location + change_results]

        if save_plots_locally:
            # Save results also locally
            index_of_event = images_to_save[0]
            index_of_event_str = str(index_of_event).zfill(3)
            before = images_to_save[1]
            after = images_to_save[2]
            #print("Saving images locally into: {0}".format(os.getcwd()))
            path_abs_outputs = os.path.dirname(os.path.dirname(os.getcwd()))

            before.save(os.path.join(path_abs_outputs, index_of_event_str+"_1_before.png"))
            after.save(os.path.join(path_abs_outputs, index_of_event_str+"_2_after.png"))

            for idx in range(3, len(images_to_save)-1):
                img = images_to_save[idx]
                desc = change_ids[idx]
                print("Saved", index_of_event_str+"_"+str(idx)+"_"+desc+".png")
                img.savefig(os.path.join(path_abs_outputs, index_of_event_str+"_"+str(idx)+"_"+desc+".png"))
                idx += 1
            
            stat_df.to_csv(os.path.join(path_abs_outputs, index_of_event_str+"_stats.csv"))
                    
    # Upload change table
    logger.experiment.log({f'Change event qualitative results': change_image_table})
        
    # Compute the summary statistics across all locations
    changes = [detection[0].squeeze() for detection in detection_results]
    for i, detect_name in enumerate(detection_method_image_names):
        # detection_results is ordered [[change_truth, detection method1, ...2, etc] for location in locations]
        these_detections = [detection[i+1].squeeze() for detection in detection_results]
        
        stat_row = [detect_name]
        for stat_method in summary_stat_methods:
            stat_row += [stat_method(changes, these_detections)]
        summary_statistics_table.add_data(*stat_row)
    
    logger.experiment.log({f'Detection technique summary statistics': summary_statistics_table})

    print("Detection technique summary statistics:")
    print(summary_statistics_table)


if __name__ == '__main__':
    main()
