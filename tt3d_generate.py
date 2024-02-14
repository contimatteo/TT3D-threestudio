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
        # "seed=42",
        # "system.cleanup_after_validation_step=true",
        # "system.cleanup_after_test_step=true",
        # "system.prompt_processor.use_perp_neg=true",
        # "system.guidance.enable_sequential_cpu_offload=true",
        # "data.batch_size=1",
    ]

    return default_args, default_extra_args


def skip_generation_or_delete_existing_model_version(
    skip_existing: bool,
    model: str,
    prompt: str,
    out_rootpath: Path,
) -> bool:
    build_result_path_fn = lambda modeldirname: Utils.Storage.build_result_path_by_prompt(
        model_dirname=modeldirname, prompt=prompt, out_rootpath=out_rootpath, assert_exists=False)

    out_model_final_dirname = Utils.Storage.get_model_final_dirname_from_id(model=model)
    out_result_final_path = build_result_path_fn(out_model_final_dirname)

    if skip_existing and out_result_final_path.exists():
        print("")
        print("========================================")
        print("Path already exists -> ", out_result_final_path)
        print("========================================")
        print("")
        return True

    if out_result_final_path.exists():
        model_dirnames_to_delete = Utils.Storage.get_model_intermediate_dirnames_from_id(model=model)
        model_dirnames_to_delete += [out_model_final_dirname]
        print("")
        print("========================================")
        for model_dirname in model_dirnames_to_delete:
            _path_to_delete = build_result_path_fn(model_dirname)
            print("Overwriting path -> ", _path_to_delete)
            shutil.rmtree(_path_to_delete)
        print("========================================")
        print("")

    return False


def _configure_and_run_model(model: str, prompt: str, out_rootpath: Path, train_steps: List[int]) -> None:
    args_configs: List[Tuple[dict, list]] = None

    if model == "dreamfusion-sd" or model == "dreamfusion-if":
        args_configs = Utils.Models.dreamfusion(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
            mode="if" if model == "dreamfusion-if" else "sd",
        )

    if model == "fantasia3d":
        args_configs = Utils.Models.fantasia3d(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

    if model == "prolificdreamer":
        args_configs = Utils.Models.prolificdreamer(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

    if model == "magic3d":
        args_configs = Utils.Models.magic3d(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
        )

    if model == "textmesh-sd" or model == "textmesh-if":
        args_configs = Utils.Models.textmesh(
            args_builder_fn=_build_default_args,
            prompt=prompt,
            out_rootpath=out_rootpath,
            train_steps=train_steps,
            mode="if" if model == "textmesh-if" else "sd",
        )

    if model == "hifa":
        args_configs = Utils.Models.hifa(
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


def _delete_prompt_embeddings_cache():
    prompt_embeddings_cache_path = Path(".threestudio_cache/text_embeddings")
    if prompt_embeddings_cache_path.exists():
        shutil.rmtree(prompt_embeddings_cache_path)


###


def main(model: str, prompt_filepath: Path, out_rootpath: Path, train_steps: List[int], skip_existing: bool):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
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

        _delete_prompt_embeddings_cache()

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

    _delete_prompt_embeddings_cache()


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
    parser.add_argument("--train-steps", type=str, required=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    arg_train_steps = args.train_steps.split(",")
    arg_train_steps = filter(lambda step: len(step) > 0, arg_train_steps)
    arg_train_steps = map(int, arg_train_steps)
    arg_train_steps = list(arg_train_steps)

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        train_steps=arg_train_steps,
        skip_existing=args.skip_existing,
    )
