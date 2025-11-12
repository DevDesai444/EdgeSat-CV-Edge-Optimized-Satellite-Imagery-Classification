# Training and evaluation

Training-oriented dataset configs in this repository are intended for WorldFloods-style data prepared through [`ml4floods`](https://github.com/spaceml-org/ml4floods).
Evaluation-oriented configs such as `floods_evaluation` are intended for the annotated event data shared in the project Google Drive folder:

- [Annotated evaluation events](https://drive.google.com/drive/folders/1VEf49IDYFXGKcfvMsfh33VSiyx5MpHEn?usp=sharing)

The main scripts support temporary staged datasets for the default demo-sized flows:

- `floods_evaluation` automatically downloads the selected event archive, extracts it into `.cache/staged_datasets`, runs evaluation, and removes the extracted event folder after a successful run.
- `alpha_multiscene_tiny` automatically downloads the public `train_minisubset.zip` archive, extracts it into `.cache/staged_datasets/train_minisubset`, runs training, and removes the extracted subset after a successful run.
- `alpha_multiscene` and `alpha_singlescene` now expose the same staging hooks. They keep their local `datasets/...` defaults, but can temporarily stage a Google Drive archive when you enable `dataset.staging` and provide a real `dataset.staging.archive_id` at runtime.

If you prefer to fetch evaluation data manually, you can still create those local paths directly with:

```
pip install gdown
python3 -m scripts.download_eval_events floods
```

## Training:
Trainign of a model is done through the `script/train_model.py` script, as in
the following command:
```
python3 -m scripts.train_model +dataset=alpha_multiscene_tiny \
                               +normalisation=log_scale \
                               +channels=high_res \
                               +training=simple_vae \
                               +module=deeper_vae \
                               +project=edgesat_train
```

In this example, the model defined in `config/module/deeper_vae.yaml` is trained
on the demo-sized dataset defined in `config/dataset/alpha_multiscene_tiny.yaml`, `config/channels/high_res.yaml`
and `config/normalisation/log_scale.yaml` for the project `edgesat_train`, which
matches the `wandb` project name.

The default W&B entity is `devdesai444-university-at-buffalo`, and `wandb_mode: auto` will use online logging when your local shell already has W&B credentials available. Use `wandb login` or set `WANDB_API_KEY` locally if you want the run to sync to W&B instead of falling back to offline mode.

The trained model is going to be saved as a checkpoint in the result folder
defined in `config/config.yaml`.

For the larger dataset presets such as `alpha_multiscene` and `alpha_singlescene`, you can either point `root_folder` at a prepared local dataset or enable temporary staging with a real archive ID, for example:

```
python3 -m scripts.train_model +dataset=alpha_multiscene \
                               ++dataset.staging.enabled=true \
                               ++dataset.staging.archive_id=<google-drive-file-id> \
                               +normalisation=log_scale \
                               +channels=high_res \
                               +training=simple_vae \
                               +module=deeper_vae \
                               +project=edgesat_train
```

To train a model with data augmentations, one can specify as training config the file `config/training/da.yaml` and append the chosen transformation config, e.g. `config/transform/random.yaml` for applying random band shifts to the inputs and a center crop to the outputs/targets. 
The full command:
```
python3 -m scripts.train_model +dataset=alpha_singlescene \
                               +normalisation=log_scale \
                               +channels=all \
                               +training=da \
                               +module=simple_ae \
                               +project=da_ae_fullgrid \
                               +transform=random
```

## Evaluation:
Once a model is trained, you can use the `scripts/evaluate_model.py` script to
evaluate it.
The model takes additional config parameters: `checkpoint` is the position of the
checkpoint of the model you want to evaluate and `evaluation` is a config file
where the list of  metrics are defined.

```
python3 -m scripts.evaluate_model \
    +dataset=floods_evaluation \
    +training=simple_ae \
    +normalisation=log_scale \
    +channels=rgb \
    +module=simple_ae_with_linear \
    +checkpoint=demo_assets/checkpoints/edgesat_pretrained_vae_128_small.ckpt \
    +project=edgesat_eval \
    +evaluation=ae_base \
    #+name=whatever_name_you_want
```
