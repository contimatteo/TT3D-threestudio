### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Optional

import argparse

from copy import deepcopy
from pathlib import Path

from utils import Utils
from launch import main as launch_script_main_fn

###

device = Utils.Cuda.init()

###


def main(
    prompt_filepath: Path,
    # out_path: Path,
    config: Path,
    train_steps: int,
):
    assert isinstance(config, Path)
    assert config.exists()
    assert config.is_file()
    assert isinstance(train_steps, int)
    assert train_steps > 0

    generic_args = {
        'config': str(config),
        'gpu': '0',
        # 'train': True,
        'validate': False,
        'test': False,
        # 'export': False,
        'gradio': False,
        'verbose': False,
        'typecheck': False,
    }

    generic_extra_args = [
        # 'system.prompt_processor.prompt=a shark',
        # 'trainer.max_steps=100',
        'system.prompt_processor.spawn=false',
    ]

    #

    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    for prompt in prompts:
        prompt_args = deepcopy(generic_args)
        prompt_args["train"] = True
        prompt_args["export"] = False

        prompt_extra_args = deepcopy(generic_extra_args)
        prompt_extra_args += [
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
        ]

        launch_script_main_fn(
            args=argparse.Namespace(**prompt_args),
            extras=argparse.Namespace(**prompt_extra_args),
        )


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    # parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, required=True)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        # out_path=args.out_path,
        config=args.config,
        train_steps=args.train_steps,
    )
