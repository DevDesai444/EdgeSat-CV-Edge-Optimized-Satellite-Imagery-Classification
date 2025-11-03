# Example training run
# Large model ~ VAE 128

train () {
    rm -rf .cache
    python3 -m scripts.train_model +dataset=alpha_multiscene \
         +normalisation=log_scale +channels=high_res +training=$1 +module=$2 +project=edgesat_train_vae_128_large +name="${3}" module.model_cls_args.latent_dim=128


}

training=simple_vae
module=deeper_vae
name=vae_128_large

train $training $module $name
