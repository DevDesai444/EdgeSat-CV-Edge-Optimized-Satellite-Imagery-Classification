# Example evaluation run for a manually selected checkpoint

eval_all_disasters () {
    for event in floods fires hurricanes landslides
    do
        rm -rf .cache
        python3 -m scripts.evaluate_model \
                +dataset=floods_evaluation \
                ++dataset.root_folder="datasets/$event" \
                +training=$1 \
                +normalisation=log_scale \
                +channels=high_res \
                +module=$2\
                +checkpoint=$3 \
                +project="edgesat_eval_vae_128_medium" \
                +evaluation=$4 \
                ++evaluation.plot_sequences=$plot_sequences \
                +name="${5}_${event}" +dataset.test_overlap=[0,0] module.model_cls_args.latent_dim=128 module.model_cls_args.extra_depth_on_scale=0

    done
}

plot_sequences=true

###############
# VAEs
###############
evaluation=vae_comprehensive
training=simple_vae
###############

checkpoint=demo_assets/checkpoints/edgesat_pretrained_vae_128_medium.ckpt
module=deeper_vae
name=vae_128_medium_checkpoint
eval_all_disasters $training $module $checkpoint $evaluation $name
