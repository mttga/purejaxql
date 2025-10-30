import os
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Union
from omegaconf import OmegaConf
import jax


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def save(
    state: Dict,
    config: Dict,
    save_dir: Union[str, os.PathLike],
    save_name: str,
    vmaps: int = 0,
) -> None:
    # save config and vmapped states

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{save_name}_config.yaml",
    )
    OmegaConf.save(config, save_path)
    print(f"Config saved at {save_path}")

    if vmaps > 0:
        for i in range(vmaps):
            vmap_state = jax.tree_util.tree_map(lambda x: x[i], state)
            save_path = os.path.join(
                save_dir,
                f"{save_name}_vmap{i}.safetensors",
            )
            save_params(vmap_state, save_path)
            print(f"Vmap state saved at {save_path}")
    else:
        save_path = os.path.join(save_dir, f"{save_name}.safetensors")
        save_params(state, save_path)
        print(f"State saved at {save_path}")
