# Example training run
# Medium model ~ VAE 128 with 0 skip connections

train () {
    rm -rf .cache
    python3 -m scripts.train_model +dataset=alpha_multiscene \
         +normalisation=log_scale +channels=high_res +training=$1 +module=$2 +project=edgesat_train_vae_128_medium +name="${3}" \
         module.model_cls_args.latent_dim=128 module.model_cls_args.extra_depth_on_scale=0

}

training=simple_vae
module=deeper_vae
name=vae_128_medium

train $training $module $name
