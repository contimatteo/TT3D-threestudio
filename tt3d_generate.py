### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Optional, Tuple, List

import argparse

from copy import deepcopy
from pathlib import Path

from utils import Utils
from launch import main as launch_script_main_fn

###

_ = Utils.Cuda.init()

###


def _build_default_args() -> Tuple[dict, list]:
    default_args = {
        'gpu': '0',
        'train': True,
        'validate': False,
        'test': False,
        'export': False,
        'gradio': False,
        'verbose': False,
        'typecheck': False,
    }

    default_extra_args = [
        'system.prompt_processor.spawn=false',
    ]

    return default_args, default_extra_args


def _dreamfusionsd(prompt: str, train_steps: int) -> None:
    ### STEP #1
    run_args, run_extra_args = _build_default_args()
    run_args["config"] = "configs/dreamfusion-sd.yaml"
    run_extra_args += [
        f"system.prompt_processor.prompt={prompt}",
        f"trainer.max_steps={train_steps}",
    ]
    generate(run_args=run_args, run_extra_args=run_extra_args)


###


def generate(run_args: dict, run_extra_args: List[str]) -> None:
    REQUIRED_ARGS = ["config", "gpu", "train", "export"]
    REQUIRED_EXTRA_ARGS = [
        'system.prompt_processor.prompt', 'trainer.max_steps'
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
    prompt_filepath: Path,
    train_steps: int,
):
    assert isinstance(model, str)
    assert len(model) > 0
    assert model in Utils.Configs.MODELS_SUPPORTED
    assert isinstance(train_steps, int)
    assert train_steps > 0

    # default_args, default_extra_args = _build_default_args()
    # prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)
    # for prompt in prompts:
    #     prompt_args = deepcopy(default_args)
    #     prompt_args["train"] = True
    #     prompt_args["export"] = False
    #     prompt_extra_args = deepcopy(default_extra_args)
    #     prompt_extra_args += [
    #         f"system.prompt_processor.prompt={prompt}",
    #         f"trainer.max_steps={train_steps}",
    #     ]
    #     # launch_script_main_fn(
    #     #     args=argparse.Namespace(**prompt_args),
    #     #     extras=prompt_extra_args,
    #     # )
    #     _generate(prompt_args=prompt_args, prompt_extra_args=prompt_extra_args)

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        if model == "dreamfusion-sd":
            _dreamfusionsd(prompt=prompt, train_steps=train_steps)
            continue

        raise Exception("Model is supported but still not implemented.")


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
    # parser.add_argument('--out-path', type=Path, required=True)
    # parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)

    args = parser.parse_args()

    #

    main(
        model=args.model,
        prompt_filepath=args.prompt_file,
        train_steps=args.train_steps,
    )
