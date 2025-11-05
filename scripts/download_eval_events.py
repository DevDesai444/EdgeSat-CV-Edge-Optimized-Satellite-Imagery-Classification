from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


EVENT_ARCHIVES = {
    "fires": "1UCNnxaL9pQSkkZQx0aDEWQL0UBXPXkv0",
    "landslides": "1CbNGrpK66Hos_TtOEut510k7CSHvSwkl",
    "hurricanes": "1VP3SYgh3bj6uPa4r_bKP-5zFP3JdGin8",
    "floods": "1scjd4gIB_eiNS-CsOyb7Q8rYWnl9TM-L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract evaluation event archives from the shared Google Drive folder."
    )
    parser.add_argument(
        "events",
        nargs="*",
        choices=sorted(EVENT_ARCHIVES),
        default=sorted(EVENT_ARCHIVES),
        help="Event archives to download. Defaults to all available events.",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Directory where extracted event folders should be created.",
    )
    parser.add_argument(
        "--archive-root",
        default="demo_assets/eval_archives",
        help="Directory where downloaded zip archives should be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download archives and re-extract event folders even if they already exist.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete zip archives after successful extraction.",
    )
    return parser.parse_args()


def require_gdown():
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gdown is required for evaluation dataset downloads. "
            "Install it with `pip install gdown` or add it to your environment first."
        ) from exc
    return gdown


def download_archive(gdown_module, event: str, archive_root: Path, overwrite: bool) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{event}.zip"
    if archive_path.exists() and not overwrite:
        print(f"Using existing archive: {archive_path}")
        return archive_path

    file_id = EVENT_ARCHIVES[event]
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {event} archive to {archive_path}")
    gdown_module.download(url, str(archive_path), quiet=False)
    return archive_path


def extract_archive(event: str, archive_path: Path, dataset_root: Path, overwrite: bool) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    event_root = dataset_root / event

    if event_root.exists() and overwrite:
        shutil.rmtree(event_root)

    if event_root.exists():
        print(f"Using existing extracted dataset: {event_root}")
        return event_root

    print(f"Extracting {archive_path} into {dataset_root}")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(dataset_root)

    if not event_root.exists():
        raise SystemExit(
            f"Archive {archive_path} was extracted, but expected folder {event_root} was not created."
        )

    return event_root


def main() -> None:
    args = parse_args()
    gdown_module = require_gdown()

    dataset_root = Path(args.dataset_root)
    archive_root = Path(args.archive_root)

    for event in args.events:
        archive_path = download_archive(gdown_module, event, archive_root, args.overwrite)
        event_root = extract_archive(event, archive_path, dataset_root, args.overwrite)
        print(f"{event} ready at {event_root}")
        if args.delete_archives and archive_path.exists():
            archive_path.unlink()


if __name__ == "__main__":
    main()
