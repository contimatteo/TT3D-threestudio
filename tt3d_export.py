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


def _build_default_args(goal: Literal["quality", "speed", "tradeoff"]) -> Tuple[dict, list]:
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
        # "system.geometry.isosurface_threshold=auto"
        ### TODO: ask to @andrea (for me it's ok without this option)
        # "system.exporter.fmt=obj",
    ]

    # if goal == "speed":
    #     default_extra_args.append("system.exporter.save_uv=false")

    return default_args, default_extra_args


def skip_exporting(
    skip_existing: bool,
    result_path: Path,
    out_rootpath: Path,
) -> bool:
    # out_model_dirname = Utils.Storage.get_model_final_dirname_from_id(model=model)
    # out_result_final_path = Utils.Storage.build_result_path_by_prompt(
    #     model_dirname=out_model_dirname,
    #     prompt=prompt,
    #     out_rootpath=out_rootpath,
    #     assert_exists=False,
    # )
    out_result_obj_filepath = Utils.Storage.build_result_export_obj_path(
        result_path=result_path,
        assert_exists=False,
    )

    if skip_existing and out_result_obj_filepath.exists():
        print("")
        print("========================================")
        print("Path already exists -> ", result_path)
        print("========================================")
        print("")
        return True

    return False


def _export(
    model: str,
    result_path: Path,
    goal: Literal["quality", "speed", "tradeoff"],
) -> None:
    run_args, run_extra_args = _build_default_args(goal=goal)

    run_args["config"] = str(result_path.joinpath("configs/parsed.yaml"))
    run_extra_args += [
        f"resume={str(result_path.joinpath('ckpts/last.ckpt'))}",
    ]

    #

    if model == "dreamfusion-sd" or model == "dreamfusion-if":
        run_extra_args += [
            "system.geometry.isosurface_threshold=auto",
            # "system.geometry.isosurface_method=mc-cpu",
            # "system.geometry.isosurface_resolution=256",
        ]
    elif model == "fantasia3d":
        pass
    elif model == "prolificdreamer":
        # run_extra_args += [
        #     "system.geometry.isosurface_method=mc-cpu",
        #     "system.geometry.isosurface_resolution=256",
        # ]
        pass
    elif model == "magic3d-sd" or model == "magic3d-if":
        pass
    elif model == "textmesh-sd" or model == "textmesh-if":
        pass
    elif model == "hifa":
        run_extra_args += [
            "system.geometry.isosurface_threshold=auto",
        ]
    else:
        ### just for safety ...
        raise Exception("Model custom run arguments not configured.")

    #

    run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


###


def run_launch_script(run_args: dict, run_extra_args: List[str]) -> None:
    REQUIRED_ARGS = ["config", "gpu", "train", "export"]
    REQUIRED_EXTRA_ARGS = ["resume", "system.exporter_type", "system.exporter.context_type"]

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


def main(
    model: str,
    source_rootpath: Path,
    goal: Literal["quality", "speed", "tradeoff"],
    skip_existing: bool,
):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()
    assert isinstance(goal, str)
    assert goal in ["quality", "speed", "tradeoff"]
    assert isinstance(skip_existing, bool)

    out_model_dirname = Utils.Storage.get_model_final_dirname_from_id(model=model)
    source_model_rootpath = source_rootpath.joinpath(out_model_dirname)

    assert source_model_rootpath.exists()
    assert source_model_rootpath.is_dir()

    for result_path in source_model_rootpath.iterdir():
        if not result_path.is_dir():
            continue

        skip_export = skip_exporting(
            skip_existing=skip_existing,
            result_path=result_path,
            source_rootpath=source_rootpath,
        )

        if skip_export:
            continue

        _export(model=model, result_path=result_path, goal=goal)


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
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        model=args.model,
        source_rootpath=args.source_path,
        goal=args.goal,
        skip_existing=args.skip_existing,
    )
