### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Optional, Tuple, List

import argparse
import os
import shutil

from copy import deepcopy
from pathlib import Path

from utils import Utils
from launch import main as launch_script_main_fn

###

_ = Utils.Cuda.init()

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
        "use_timestamp=False",
        "system.prompt_processor.spawn=false",
    ]

    return default_args, default_extra_args


# def _delete_unnecessary_ckpts(
#     model_dirname: str,
#     prompt: str,
#     out_rootpath: Path,
# ) -> None:
#     result_path = Utils.Storage.build_result_path_by_prompt(
#         model_dirname=model_dirname,
#         prompt=prompt,
#         out_rootpath=out_rootpath,
#     )

#     ckpts_path = result_path.joinpath("ckpts")
#     assert ckpts_path.exists()
#     assert ckpts_path.is_dir()
#     ### "last.ckpt" is a symlink to the last checkpoint.
#     last_ckpt_path = ckpts_path.joinpath("last.ckpt")
#     assert last_ckpt_path.exists()
#     assert last_ckpt_path.is_symlink()  ### INFO: notice this ...

#     ckpts_names_to_keep = [
#         "last.ckpt",
#         Path(os.readlink(last_ckpt_path)).name,
#     ]

#     for ckpt_path in ckpts_path.glob("*.ckpt"):
#         if ckpt_path.name in ckpts_names_to_keep:
#             continue
#         ckpt_path.unlink()

###

# def __dreamfusionsd(prompt: str, out_rootpath: Path, train_steps: int) -> None:
#     args_builder_fn = lambda: _build_default_args()

#     def __step1_run() -> None:
#         CONFIG_NAME = "dreamfusion-sd"
#         run_args, run_extra_args = args_builder_fn()
#         run_args["config"] = f"configs/{CONFIG_NAME}.yaml"
#         run_extra_args += [
#             f"exp_root_dir={str(out_rootpath)}",
#             f"system.prompt_processor.prompt={prompt}",
#             f"trainer.max_steps={train_steps}",
#         ]
#         run_launch_script(run_args=run_args, run_extra_args=run_extra_args)
#         _delete_unnecessary_ckpts(
#             model_dirname=CONFIG_NAME,
#             prompt=prompt,
#             out_rootpath=out_rootpath,
#         )

#     __step1_run()


def skip_generation_or_delete_existing_model_version(
    skip_existing: bool,
    model: str,
    prompt: str,
    out_rootpath: Path,
) -> bool:
    build_result_path_fn = lambda modeldirname: Utils.Storage.build_result_path_by_prompt(
        model_dirname=modeldirname, prompt=prompt, out_rootpath=out_rootpath)

    out_model_final_dirname = Utils.Storage.get_model_final_dirname_from_id(model=model)
    out_result_final_path = build_result_path_fn(out_model_final_dirname)

    if skip_existing and out_result_final_path.exists():
        return True

    if out_result_final_path.exists():
        model_dirnames_to_delete = Utils.Storage.get_model_intermediate_dirnames_from_id(model=model)
        model_dirnames_to_delete += [out_model_final_dirname]
        for model_dirname in model_dirnames_to_delete:
            shutil.rmtree(build_result_path_fn(model_dirname))

    return False


def _configure_and_run_model(model: str, prompt: str, out_rootpath: Path, train_steps: int) -> None:
    args_configs: List[Tuple[dict, list]] = None

    if model == "dreamfusion-sd":
        args_configs = Utils.Models.dreamfusionsd(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

    if model == "fantasia3d":
        args_configs = Utils.Models.fantasia3d(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

    if args_configs is None:
        raise Exception("Model is supported but still not implemented.")

    for args_config in args_configs:
        run_args, run_extra_args = args_config
        __run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


###


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


def main(model: str, prompt_filepath: Path, out_rootpath: Path, train_steps: int, skip_existing: bool):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(out_rootpath, Path)
    assert out_rootpath.exists()
    assert out_rootpath.is_dir()
    assert isinstance(train_steps, int)
    assert train_steps > 0
    assert isinstance(skip_existing, bool)

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        skip_generation = skip_generation_or_delete_existing_model_version(
            skip_existing=skip_existing,
            model=model,
            prompt=prompt,
            out_rootpath=out_rootpath,
        )

        if skip_generation:
            continue

        _configure_and_run_model(
            model=model,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

        # args_configs: List[Tuple[dict, list]] = None
        # if model == "dreamfusion-sd":
        #     # __dreamfusionsd(
        #     #     prompt=prompt, out_rootpath=out_rootpath, train_steps=train_steps,
        #     # )
        #     args_configs = Utils.Models.dreamfusionsd(
        #         args_builder_fn=_build_default_args,
        #         prompt=prompt,
        #         out_rootpath=out_rootpath,
        #         train_steps=train_steps,
        #     )
        # if args_configs is None:
        #     raise Exception("Model is supported but still not implemented.")
        # for args_config in args_configs:
        #     run_args, run_extra_args = args_config
        #     run_launch_script(run_args=run_args, run_extra_args=run_extra_args)


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=Utils.Configs.MODELS_SUPPORTED,
        required=True,
    )
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        train_steps=args.train_steps,
        skip_existing=args.skip_existing,
    )
