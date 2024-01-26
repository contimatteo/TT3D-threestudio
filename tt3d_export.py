### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Optional, Tuple, List, Literal

import argparse
import os

from copy import deepcopy
from pathlib import Path

from utils import Utils
from launch import main as launch_script_main_fn

###

_ = Utils.Cuda.init()

###


def _build_default_args(
        goal: Literal["quality", "speed", "tradeoff"]) -> Tuple[dict, list]:
    default_args = {
        'gpu': '0',
        'train': False,
        'validate': False,
        'test': False,
        'export': True,
        'gradio': False,
        'verbose': False,
        'typecheck': False,
    }

    default_extra_args = [
        "system.exporter_type=mesh-exporter",
        "system.exporter.context_type=cuda",
        "system.geometry.isosurface_threshold=auto"
        ### TODO: ask to @andrea (for me it's ok without this option)
        # "system.exporter.fmt=obj",
    ]

    # if goal == "speed":
    #     default_extra_args.append("system.exporter.save_uv=false")

    return default_args, default_extra_args


###


def _export(
    model: str,
    result_path: Path,
    goal: Literal["quality", "speed", "tradeoff"],
) -> None:
    # 'config':'./outputs/dreamfusion-sd/a_shark@20240124-171111/configs/parsed.yaml',
    # 'resume=./outputs/dreamfusion-sd/a_shark@20240124-171111/ckpts/last.ckpt',

    run_args, run_extra_args = _build_default_args(goal=goal)

    run_args["config"] = str(result_path.joinpath("configs/parsed.yaml"))
    run_extra_args += [
        f"resume={str(result_path.joinpath('ckpts/last.ckpt'))}",
    ]

    run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


###


def run_launch_script(run_args: dict, run_extra_args: List[str]) -> None:
    REQUIRED_ARGS = ["config", "gpu", "train", "export"]
    REQUIRED_EXTRA_ARGS = [
        "resume", "system.exporter_type", "system.exporter.context_type",
        "system.geometry.isosurface_threshold"
    ]

    assert isinstance(run_args, dict)
    assert isinstance(run_extra_args, list)
    assert all((isinstance(p, str) for p in run_extra_args))

    for k in REQUIRED_ARGS:
        assert k in run_args
    for k in REQUIRED_EXTRA_ARGS:
        assert any((p.startswith(k) for p in run_extra_args))

    launch_script_main_fn(
        args=argparse.Namespace(**run_args),
        extras=run_extra_args,
    )


def main(
    model: str,
    source_rootpath: Path,
    goal: Literal["quality", "speed", "tradeoff"],
):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()
    assert isinstance(goal, str)
    assert goal in ["quality", "speed", "tradeoff"]

    out_model_dirname: str = None
    if model == "dreamfusion-sd":
        out_model_dirname = "dreamfusion-sd"
    if model == "fantasia3d":
        out_model_dirname = "fantasia3d"
    else:
        raise Exception("Model output dirname not configured.")

    #

    source_model_rootpath = source_rootpath.joinpath(out_model_dirname)
    assert source_model_rootpath.exists()
    assert source_model_rootpath.is_dir()

    for result_path in source_model_rootpath.iterdir():
        if not result_path.is_dir():
            continue

        _export(model=model, result_path=result_path, goal=goal)

    # prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)
    # for prompt in prompts:
    #     if not isinstance(prompt, str) or len(prompt) < 2:
    #         continue
    #     if model == "dreamfusion-sd":
    #         __dreamfusionsd(
    #             prompt=prompt,
    #             out_rootpath=out_rootpath,
    #             train_steps=train_steps,
    #         )
    #         continue
    #     raise Exception("Model is supported but still not implemented.")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=Utils.Configs.MODELS_SUPPORTED,
        required=True,
    )
    parser.add_argument('--source-path', type=Path, required=True)
    parser.add_argument(
        "--goal",
        type=str,
        choices=["quality", "speed", "tradeoff"],
        default="tradeoff",
    )

    args = parser.parse_args()

    #

    main(
        model=args.model,
        source_rootpath=args.source_path,
        goal=args.goal,
    )
