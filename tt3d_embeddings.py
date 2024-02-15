### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Optional, Tuple, List

import argparse
import os
import shutil

from copy import deepcopy
from pathlib import Path

from utils import Utils
from launch import main as launch_script_main_fn

from threestudio.models.prompt_processors.base import hash_prompt

###

_ = Utils.Cuda.init()

DEEPFLOYD_MODEL_NAME = "DeepFloyd/IF-I-XL-v1.0"
STABLEDIFFUSION_MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"

###


def _build_default_args() -> Tuple[dict, list]:
    default_args = {
        "gpu": "0",
        "train": True,
        "validate": False,
        "test": False,
        "export": False,
        "gradio": False,
        "verbose": False,
        "typecheck": False,
    }

    default_extra_args = [
        "seed=42",
        "use_timestamp=False",
        "system.prompt_processor.spawn=false",
        "trainer.val_check_interval=10000",  ### avoid performing validation
        # "system.cleanup_after_validation_step=true",
        # "system.cleanup_after_test_step=true",
        "system.guidance.enable_sequential_cpu_offload=true",
    ]

    return default_args, default_extra_args


def _generate_deepfloyd_embeddings(
    prompt: str,
    out_rootpath: Path,
    train_steps: List[int],
    skip_existing: bool,
) -> None:
    prompt_hashed = hash_prompt(model=DEEPFLOYD_MODEL_NAME, prompt=prompt)
    prompt_hashed_filepath = Path(".").joinpath(
        ".threestudio_cache/text_embeddings",
        f"{prompt_hashed}.pt",
    )

    if skip_existing and prompt_hashed_filepath.exists():
        print("")
        print("========================================")
        print("Embedding already exists -> ", prompt_hashed_filepath)
        print("========================================")
        print("")
        return

    #

    args_configs: List[Tuple[dict, list]] = []

    args_configs += Utils.Models.dreamfusion(
        args_builder_fn=_build_default_args,
        prompt=prompt,
        out_rootpath=out_rootpath,
        train_steps=train_steps,
        mode="if",
    )

    #

    assert len(args_configs) > 0

    for args_config in args_configs:
        run_args, run_extra_args = args_config
        __run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


def _generate_stablediffusion_embeddings(
    prompt: str,
    out_rootpath: Path,
    train_steps: List[int],
    skip_existing: bool,
) -> None:
    prompt_hashed = hash_prompt(model=STABLEDIFFUSION_MODEL_NAME, prompt=prompt)
    prompt_hashed_filepath = Path(".").joinpath(
        ".threestudio_cache/text_embeddings",
        f"{prompt_hashed}.pt",
    )

    if skip_existing and prompt_hashed_filepath.exists():
        print("")
        print("========================================")
        print("Embedding already exists -> ", prompt_hashed_filepath)
        print("========================================")
        print("")
        return

    #

    args_configs: List[Tuple[dict, list]] = []

    args_configs += Utils.Models.dreamfusion(
        args_builder_fn=_build_default_args,
        prompt=prompt,
        out_rootpath=out_rootpath,
        train_steps=train_steps,
        mode="sd",
    )

    #

    assert len(args_configs) > 0

    for args_config in args_configs:
        run_args, run_extra_args = args_config
        __run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


def __run_launch_script(run_args: dict, run_extra_args: List[str]) -> None:
    REQUIRED_ARGS = ["config", "gpu", "train", "export"]
    REQUIRED_EXTRA_ARGS = ['system.prompt_processor.prompt', 'trainer.max_steps']

    assert isinstance(run_args, dict)
    assert isinstance(run_extra_args, list)
    assert all((isinstance(p, str) for p in run_extra_args))

    for k in REQUIRED_ARGS:
        assert k in run_args
    for k in REQUIRED_EXTRA_ARGS:
        assert any((p.startswith(k) for p in run_extra_args))

    print("")
    print("")
    print("========================================")
    print(run_args)
    print(run_extra_args)
    print("========================================")
    print("")
    print("")

    launch_script_main_fn(
        args=argparse.Namespace(**run_args),
        extras=run_extra_args,
    )


###


def main(
    prompt_filepath: Path,
    out_rootpath: Path,
    train_steps: List[int],
    skip_existing: bool,
):
    assert isinstance(out_rootpath, Path)
    assert isinstance(train_steps, list)
    assert all((isinstance(step, int) for step in train_steps))
    assert all((0 <= step <= 10000 for step in train_steps))
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True)

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        _generate_deepfloyd_embeddings(
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
            skip_existing=skip_existing,
        )

        _generate_stablediffusion_embeddings(
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
            skip_existing=skip_existing,
        )


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--train-steps", type=str, required=True)
    # parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    arg_train_steps = args.train_steps.split(",")
    arg_train_steps = filter(lambda step: len(step) > 0, arg_train_steps)
    arg_train_steps = map(int, arg_train_steps)
    arg_train_steps = list(arg_train_steps)

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        train_steps=arg_train_steps,
        skip_existing=True,
    )
