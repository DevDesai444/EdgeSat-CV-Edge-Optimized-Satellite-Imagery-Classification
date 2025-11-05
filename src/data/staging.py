from __future__ import annotations

import copy
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path


EVAL_EVENT_ARCHIVES = {
    "fires": "1UCNnxaL9pQSkkZQx0aDEWQL0UBXPXkv0",
    "landslides": "1CbNGrpK66Hos_TtOEut510k7CSHvSwkl",
    "hurricanes": "1VP3SYgh3bj6uPa4r_bKP-5zFP3JdGin8",
    "floods": "1scjd4gIB_eiNS-CsOyb7Q8rYWnl9TM-L",
}

TRAINING_ARCHIVES = {
    "train_minisubset": {
        "archive_id": "1rl3Clf0c7HlXnlPXO837Pjr2iCjwak0Y",
        "archive_name": "train_minisubset.zip",
        "expected_root_name": "train_minisubset",
    },
}


def require_gdown():
    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gdown is required for automatic dataset staging. "
            "Install it with `pip install gdown` or use the repo environment from env.yaml."
        ) from exc
    return gdown


def ensure_path(path_value, default_path: Path) -> Path:
    if path_value in (None, ""):
        return default_path
    return Path(path_value)


def download_gdrive_archive(archive_id: str, archive_path: Path, overwrite: bool = False) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and not overwrite:
        print(f"Using existing archive: {archive_path}")
        return archive_path

    gdown = require_gdown()
    url = f"https://drive.google.com/uc?id={archive_id}"
    print(f"Downloading dataset archive to {archive_path}")
    gdown.download(url, str(archive_path), quiet=False)
    return archive_path


def extract_archive(archive_path: Path, extract_root: Path, expected_root_name: str, overwrite: bool = False) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted_root = extract_root / expected_root_name

    if extracted_root.exists() and overwrite:
        shutil.rmtree(extracted_root)

    if extracted_root.exists():
        print(f"Using existing extracted dataset: {extracted_root}")
        return extracted_root

    print(f"Extracting {archive_path} into {extract_root}")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_root)

    if not extracted_root.exists():
        raise SystemExit(
            f"Expected extracted dataset folder {extracted_root} was not created from {archive_path}."
        )

    return extracted_root


def resolve_stage_spec(dataset_cfg: dict) -> dict | None:
    staging_cfg = dataset_cfg.get("staging")
    if not staging_cfg or not staging_cfg.get("enabled", False):
        return None

    provider = staging_cfg.get("provider")
    if provider == "eval_event_gdrive":
        event_name = staging_cfg.get("event_name") or Path(dataset_cfg["root_folder"]).name
        if event_name not in EVAL_EVENT_ARCHIVES:
            raise SystemExit(
                f"Unsupported evaluation event '{event_name}'. "
                f"Supported events: {sorted(EVAL_EVENT_ARCHIVES)}"
            )
        return {
            "archive_id": EVAL_EVENT_ARCHIVES[event_name],
            "archive_name": f"{event_name}.zip",
            "expected_root_name": event_name,
        }

    if provider == "training_subset_gdrive":
        subset_name = staging_cfg.get("subset_name") or Path(dataset_cfg["root_folder"]).name
        if subset_name not in TRAINING_ARCHIVES:
            raise SystemExit(
                f"Unsupported training subset '{subset_name}'. "
                f"Supported subsets: {sorted(TRAINING_ARCHIVES)}"
            )
        return TRAINING_ARCHIVES[subset_name]

    if provider == "gdrive_archive":
        archive_id = staging_cfg.get("archive_id")
        if not archive_id:
            raise SystemExit(
                "Missing required staging key for provider 'gdrive_archive': archive_id"
            )
        root_name = Path(dataset_cfg["root_folder"]).name
        return {
            "archive_id": archive_id,
            "archive_name": staging_cfg.get("archive_name") or f"{root_name}.zip",
            "expected_root_name": staging_cfg.get("expected_root_name") or root_name,
        }

    raise SystemExit(f"Unsupported dataset staging provider: {provider}")


@contextmanager
def prepared_dataset_config(dataset_cfg: dict, cache_dir: str):
    cfg = copy.deepcopy(dataset_cfg)
    staging_cfg = cfg.get("staging")
    if not staging_cfg or not staging_cfg.get("enabled", False):
        yield cfg
        return

    spec = resolve_stage_spec(cfg)

    stage_root = ensure_path(
        staging_cfg.get("stage_root"),
        Path(cache_dir) / "staged_datasets",
    )
    archive_root = ensure_path(
        staging_cfg.get("archive_root"),
        Path(cache_dir) / "staged_archives",
    )
    cleanup_policy = staging_cfg.get("cleanup_policy", "on_success")
    delete_archive_after_extract = staging_cfg.get("delete_archive_after_extract", True)
    overwrite = staging_cfg.get("overwrite", False)

    archive_path = archive_root / spec["archive_name"]
    extracted_root = None
    run_succeeded = False

    try:
        archive_path = download_gdrive_archive(spec["archive_id"], archive_path, overwrite=overwrite)
        extracted_root = extract_archive(
            archive_path,
            stage_root,
            spec["expected_root_name"],
            overwrite=overwrite,
        )
        cfg["root_folder"] = str(extracted_root)

        if delete_archive_after_extract and archive_path.exists():
            archive_path.unlink()

        yield cfg
        run_succeeded = True
    finally:
        should_cleanup = (
            cleanup_policy == "always"
            or (cleanup_policy == "on_success" and run_succeeded)
        )
        if should_cleanup and extracted_root is not None and extracted_root.exists():
            print(f"Cleaning up staged dataset: {extracted_root}")
            shutil.rmtree(extracted_root)
