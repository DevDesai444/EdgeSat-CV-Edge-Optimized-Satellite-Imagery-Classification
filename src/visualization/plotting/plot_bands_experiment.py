import argparse
import json

from pylab import plt

from wandb_functions import make_plot_from_ids


DEFAULT_METRICS = [
    ("cos_emb | memory 3 | 32x32 - mean", "cos_emb_3"),
    ("cos_pixel | memory 3 | 32x32 - mean", "cos_pix_3"),
    ("cos_emb | memory 1 | 32x32 - mean", "cos_emb_1"),
    ("cos_pixel | memory 1 | 32x32 - mean", "cos_pix_1"),
]


def main():
    parser = argparse.ArgumentParser(
        description='Plot band-ablation experiment summaries from W&B run ids.'
    )
    parser.add_argument('--entity', required=True, help='W&B entity or team name.')
    parser.add_argument(
        '--spec',
        required=True,
        help='Path to a JSON file with title, evaluation_ids, dataset_names, config_names, and wandb_project_names.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional output image path. Defaults to lastplot_<title>.png.',
    )
    args = parser.parse_args()

    with open(args.spec, 'r', encoding='utf-8') as fh:
        spec = json.load(fh)

    title = spec['title']
    evaluation_ids = spec['evaluation_ids']
    dataset_names = spec['dataset_names']
    config_names = spec['config_names']
    wandb_project_names = spec['wandb_project_names']
    output_path = args.output or f"lastplot_{title}.png"

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title)

    for ax, (metric, metric_human_name) in zip(axs.flat, DEFAULT_METRICS):
        make_plot_from_ids(
            ax,
            args.entity,
            evaluation_ids,
            config_names,
            dataset_names,
            wandb_project_names,
            metric,
            metric_human_name,
        )

    plt.savefig(output_path)


if __name__ == '__main__':
    main()
