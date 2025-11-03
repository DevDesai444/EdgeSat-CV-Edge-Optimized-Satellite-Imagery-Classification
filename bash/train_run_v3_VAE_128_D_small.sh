# Example training run
# Small model ~ VAE 128 with 0 skip connections and less hidden channels

train () {
    rm -rf .cache
    python3 -m scripts.train_model +dataset=alpha_multiscene \
         +normalisation=log_scale +channels=high_res +training=$1 +module=$2 +project=edgesat_train_vae_128_small +name="${3}" \
         module.model_cls_args.latent_dim=128 module.model_cls_args.extra_depth_on_scale=0 module.model_cls_args.hidden_channels=[16,32,64]

}

training=simple_vae
module=deeper_vae
name=vae_128_small

train $training $module $name
