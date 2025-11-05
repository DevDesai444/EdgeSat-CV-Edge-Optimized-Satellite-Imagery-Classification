import importlib
import os
from pathlib import Path

import omegaconf


DEFAULT_WANDB_ENTITY = "devdesai444-university-at-buffalo"
DEFAULT_WANDB_MODE = "auto"


def load_obj(obj_path):
    """
    Call an object from a string
    """

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0)

    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object '{obj_name}' cannot be loaded from '{obj_path}'."
        )
    return getattr(module_obj, obj_name)


def deepconvert(omega_conf):
    if isinstance(omega_conf, omegaconf.dictconfig.DictConfig):
        not_omega_conf = {}
        for k, v in omega_conf.items():
            not_omega_conf.update({k: deepconvert(v)})
        return not_omega_conf

    if isinstance(omega_conf, omegaconf.listconfig.ListConfig):
        not_omega_conf = []
        for v in omega_conf:
            not_omega_conf.append(deepconvert(v))
        return not_omega_conf

    return omega_conf


def wandb_credentials_available():
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return True

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False

    try:
        return "api.wandb.ai" in netrc_path.read_text()
    except OSError:
        return False


def resolve_wandb_entity(cfg=None):
    if cfg is not None:
        entity = cfg.get("entity")
        if entity:
            return entity

    return os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)


def resolve_wandb_mode(cfg=None):
    mode = None
    if cfg is not None:
        mode = cfg.get("wandb_mode")

    if not mode:
        mode = os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE)

    mode = str(mode).lower()
    if mode == "dryrun":
        mode = "offline"
    if mode == "auto":
        return "online" if wandb_credentials_available() else "offline"
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode: {mode}")
    return mode


def maybe_wandb_login(wandb_module):
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb_module.login(key=api_key, relogin=False)
