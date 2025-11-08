import argparse

from wandb_functions import get_runs_from_project


def main():
    parser = argparse.ArgumentParser(
        description='List W&B run names and ids for an entity/project pair.'
    )
    parser.add_argument('--entity', required=True, help='W&B entity or team name.')
    parser.add_argument('--project', required=True, help='W&B project name.')
    parser.add_argument(
        '--filter-name',
        default='',
        help='Optional substring filter applied to run names.',
    )
    args = parser.parse_args()

    runs = get_runs_from_project(args.entity, args.project, args.filter_name)

    print("PROJECT", args.project, "has:")
    for run in runs:
        print(run.name, run.id)


if __name__ == '__main__':
    main()
