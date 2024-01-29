### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, List, Callable

import os
import torch

from pathlib import Path
from datetime import datetime

###


class _Cuda():

    @staticmethod
    def is_available() -> bool:
        _cuda = torch.cuda.is_available()
        _cudnn = torch.backends.cudnn.enabled
        return _cuda and _cudnn

    @classmethod
    def device(cls) -> torch.cuda.device:
        assert cls.is_available()
        return torch.device('cuda')

    @classmethod
    def count_devices(cls) -> int:
        assert cls.is_available()
        return torch.cuda.device_count()

    @classmethod
    def get_current_device_info(cls) -> Tuple[int, str]:
        _idx = torch.cuda.current_device()
        _name = torch.cuda.get_device_name(_idx)
        return _idx, _name

    @staticmethod
    def get_visible_devices_param() -> str:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    @classmethod
    def init(cls) -> torch.cuda.device:
        """
        We run all the experiments on server which have 4 different GPUs.
        Unfortunately, we cannot use all of them at the same time, since many other people are 
        using the server. Therefore, we have to specify which GPU we want to use.
        In particular, we have to use the GPU #1 (Nvidia RTX-3090).
        In order to avoid naive mistakes, we also check that the {CUDA_VISIBLE_DEVICES} environment 
        variable is set.
        """
        assert cls.is_available()
        assert isinstance(cls.get_visible_devices_param(), str)
        # assert cls.get_visible_devices_param() == "1"
        assert cls.count_devices() == 1

        device_idx, _ = cls.get_current_device_info()
        assert device_idx == 0

        return cls.device()


###


class _Prompt():

    ENCODING_CHAR: str = "_"

    @classmethod
    def encode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(" ", cls.ENCODING_CHAR)
        return prompt

    @classmethod
    def decode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(cls.ENCODING_CHAR, " ")
        return prompt

    @staticmethod
    def extract_from_file(filepath: Path) -> List[str]:
        assert isinstance(filepath, Path)
        assert filepath.exists()
        assert filepath.is_file()
        assert filepath.suffix == ".txt"

        with open(filepath, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        prompts = map(lambda p: p.strip(), prompts)
        prompts = filter(lambda p: len(p) > 1, prompts)
        ### TODO: filter out prompts with special chars ...
        prompts = list(prompts)

        return prompts


###


class _Configs():
    MODELS_SUPPORTED: List[str] = [
        "dreamfusion-sd",
        "fantasia3d",
        "prolificdreamer",
        "magic3d",
        "textmesh",
    ]

    # @classmethod
    # def model_name_to_output_model_dir_name(cls, model: str) -> str:
    #     """
    #     In some cases the model name is different from the model output directory name.
    #     """
    #     assert isinstance(model, str)
    #     assert len(model) > 0
    #     assert model in cls.MODELS_SUPPORTED
    #     if model == "dreamfusion-sd":
    #         return "dreamfusion-sd"
    #     raise NotImplementedError("Model name not supported.")


class _Storage():

    @staticmethod
    def build_result_path_by_prompt(
        model_dirname: str,
        prompt: str,
        out_rootpath: Path,
        assert_exists: bool,
    ) -> Path:
        assert "_" not in prompt

        out_model_path = out_rootpath.joinpath(model_dirname)

        prompt_enc = Utils.Prompt.encode(prompt=prompt)
        out_model_prompt_path = out_model_path.joinpath(prompt_enc)

        if assert_exists:
            assert out_model_path.exists()
            assert out_model_path.is_dir()

        return out_model_prompt_path

    @staticmethod
    def search_last_result_output_path_over_timestamps(
        model_dirname: str,
        prompt: str,
        out_rootpath: Path,
    ) -> Path:
        """
        There may be multiple subdirs related to the same prompt.
        This may be caused by multiple runs using the same prompt.
        We want to find the most recent one.
        Examples:
        -  outputs/dreamfusion-sd/a_shark@20231217-110220
        -  outputs/dreamfusion-sd/a_shark@20240124-120330
        Moreover we may have multiple subdirs which have in common some parts of the same prompt.
        Examples (sharing "a shark" prompt part):
        -  outputs/dreamfusion-sd/a_shark@...
        -  outputs/dreamfusion-sd/a_big_shark@...
        -  outputs/dreamfusion-sd/a_shark_with_red_nose@...
        """

        # output_rootdir = Path("outputs")
        out_model_path = out_rootpath.joinpath(model_dirname)

        assert out_model_path.exists()
        assert out_model_path.is_dir()

        #

        ### first, try to collect all paths which refers EXACLTY to the {prompt}.
        results_candidates_dirnames: List[str] = []

        for result_path in out_model_path.iterdir():
            if not result_path.is_dir():
                continue

            prompt_enc = Utils.Prompt.encode(prompt=prompt)
            result_dirname = result_path.name

            ### {startswith} is not a sufficient condtion, but it's
            ### a good start to filter out wrong results.
            ### examples:
            ###   -  outputs/dreamfusion-sd/a_shark@...
            ###   -  outputs/dreamfusion-sd/a_shark_with_red_nose@...
            if not result_dirname.startswith(prompt_enc):
                continue
            ### ok now we are sure that the dirname starts with the prompt.
            ### let's check if it's the exact prompt.
            if result_dirname.split("@")[0] != prompt_enc:
                continue

            results_candidates_dirnames.append(result_dirname)

        #

        assert len(results_candidates_dirnames) > 0

        ### Now we are sure that we have at least one result which refers to the exact prompt.
        ### The issue now is that we may have multiple results which refers to the exact prompt
        ### due to multiple runs with different timestamps.
        ### We have to find the most recent one

        last_datetime_obj: datetime = None
        last_dirname: str = None

        for result_dirname in results_candidates_dirnames:
            result_dirname_splits = result_dirname.split("@")
            prompt_enc = result_dirname_splits[0]  ### "a_shark"
            datetime_enc = result_dirname_splits[1]  ### "20231217-110220"
            datetime_obj = datetime.strptime(datetime_enc, '%Y%m%d-%H%M%S')

            if (last_dirname is None) or (datetime_obj > last_datetime_obj):
                last_datetime_obj = datetime_obj
                last_dirname = result_dirname

        assert last_dirname is not None

        #

        last_result_path = out_model_path.joinpath(last_dirname)

        assert last_result_path.exists()
        assert last_result_path.is_dir()

        return last_result_path

    @staticmethod
    def delete_unnecessary_ckpts(
        model_dirname: str,
        prompt: str,
        out_rootpath: Path,
    ) -> None:
        result_path = Utils.Storage.build_result_path_by_prompt(
            model_dirname=model_dirname,
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=True,
        )

        ckpts_path = result_path.joinpath("ckpts")
        assert ckpts_path.exists()
        assert ckpts_path.is_dir()
        ### "last.ckpt" is a symlink to the last checkpoint.
        last_ckpt_path = ckpts_path.joinpath("last.ckpt")
        assert last_ckpt_path.exists()
        assert last_ckpt_path.is_symlink()  ### INFO: notice this ...

        ckpts_names_to_keep = [
            "last.ckpt",
            Path(os.readlink(last_ckpt_path)).name,
        ]

        for ckpt_path in ckpts_path.glob("*.ckpt"):
            if ckpt_path.name in ckpts_names_to_keep:
                continue
            ckpt_path.unlink()

    @staticmethod
    def get_model_final_dirname_from_id(model: str) -> str:
        assert isinstance(model, str)
        assert len(model) > 0
        assert model in Utils.Configs.MODELS_SUPPORTED

        if model == "dreamfusion-sd":
            return "dreamfusion-sd"

        if model == "fantasia3d":
            return "fantasia3d-texture"

        if model == "prolificdreamer":
            return "prolificdreamer-texture"

        if model == "magic3d":
            return "magic3d-refine-sd"

        if model == "textmesh":
            return "textmesh-sd"

        raise Exception("Model output final dirname not configured.")

    @staticmethod
    def get_model_intermediate_dirnames_from_id(model: str) -> List[str]:
        assert isinstance(model, str)
        assert len(model) > 0
        assert model in Utils.Configs.MODELS_SUPPORTED

        if model == "dreamfusion-sd":
            return []

        if model == "fantasia3d":
            return ["fantasia3d"]

        if model == "prolificdreamer":
            return ["prolificdreamer", "prolificdreamer-geometry"]

        if model == "magic3d":
            return ["magic3d-coarse-sd"]

        if model == "textmesh":
            return []

        raise Exception("Model output intermediate dirnames not configured.")


###


class _Models():

    @staticmethod
    def dreamfusionsd(
        args_builder_fn: Callable[[], Tuple[dict, list]],
        prompt: str,
        out_rootpath: Path,
        train_steps: int,
    ) -> List[Tuple[dict, list]]:

        args_configs: List[Tuple[dict, list]] = []

        ###
        ### STEP #1
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/dreamfusion-sd.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
        ]

        args_configs.append((run_args, run_extra_args))

        ###

        return args_configs

    @staticmethod
    def fantasia3d(
        args_builder_fn: Callable[[], Tuple[dict, list]],
        prompt: str,
        out_rootpath: Path,
        train_steps: int,
    ) -> List[Tuple[dict, list]]:

        args_configs: List[Tuple[dict, list]] = []

        ###
        ### STEP #1
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/fantasia3d.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            "system.renderer.context_type=cuda",
        ]

        args_configs.append((run_args, run_extra_args))

        result_path = _Storage.build_result_path_by_prompt(
            model_dirname="fantasia3d",
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        ###
        ### STEP #2
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/fantasia3d-texture.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            f"system.geometry_convert_from={str(result_path.joinpath('ckpts', 'last.ckpt'))}",
            "system.renderer.context_type=cuda",
        ]

        args_configs.append((run_args, run_extra_args))

        ###

        return args_configs

    @staticmethod
    def prolificdreamer(
        args_builder_fn: Callable[[], Tuple[dict, list]],
        prompt: str,
        out_rootpath: Path,
        train_steps: int,
    ) -> List[Tuple[dict, list]]:

        args_configs: List[Tuple[dict, list]] = []

        ###
        ### STEP #1
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/prolificdreamer.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            # "system.renderer.context_type=cuda",
            "data.width=64",  ### TODO: prefers memory optimization over quality
            "data.height=64",  ### TODO: prefers memory optimization over quality
            "data.batch_size=1",  ### TODO: prefers memory optimization over quality
            # "system.guidance.pretrained_model_name_or_path_lora='stabilityai/stable-diffusion-2-1-base'",
        ]

        args_configs.append((run_args, run_extra_args))

        result_path = _Storage.build_result_path_by_prompt(
            model_dirname="prolificdreamer",
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        ###
        ### STEP #2
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/prolificdreamer-geometry.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            "system.renderer.context_type=cuda",
            f"system.geometry_convert_from={str(result_path.joinpath('ckpts', 'last.ckpt'))}",
            "system.geometry_convert_override.isosurface_threshold=auto",
        ]

        args_configs.append((run_args, run_extra_args))

        result_path = _Storage.build_result_path_by_prompt(
            model_dirname="prolificdreamer-geometry",
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        ###
        ### STEP #3
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/prolificdreamer-texture.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            "system.renderer.context_type=cuda",
            f"system.geometry_convert_from={str(result_path.joinpath('ckpts', 'last.ckpt'))}",
        ]

        args_configs.append((run_args, run_extra_args))

        ###

        return args_configs

    @staticmethod
    def magic3d(
        args_builder_fn: Callable[[], Tuple[dict, list]],
        prompt: str,
        out_rootpath: Path,
        train_steps: int,
    ) -> List[Tuple[dict, list]]:

        args_configs: List[Tuple[dict, list]] = []

        ###
        ### STEP #1
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/magic3d-coarse-sd.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
        ]

        args_configs.append((run_args, run_extra_args))

        result_path = _Storage.build_result_path_by_prompt(
            model_dirname="magic3d-coarse-sd",
            prompt=prompt,
            out_rootpath=out_rootpath,
            assert_exists=False,
        )

        ###
        ### STEP #2
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/magic3d-refine-sd.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
            f"system.geometry_convert_from={str(result_path.joinpath('ckpts', 'last.ckpt'))}",
            "system.renderer.context_type=cuda",
        ]

        args_configs.append((run_args, run_extra_args))

        ###

        return args_configs

    @staticmethod
    def textmesh(
        args_builder_fn: Callable[[], Tuple[dict, list]],
        prompt: str,
        out_rootpath: Path,
        train_steps: int,
    ) -> List[Tuple[dict, list]]:

        args_configs: List[Tuple[dict, list]] = []

        ###
        ### STEP #1
        ###

        run_args, run_extra_args = args_builder_fn()

        run_args["config"] = "configs/textmesh-sd.yaml"
        run_extra_args += [
            f"exp_root_dir={str(out_rootpath)}",
            f"system.prompt_processor.prompt={prompt}",
            f"trainer.max_steps={train_steps}",
        ]

        args_configs.append((run_args, run_extra_args))

        ###

        return args_configs


###


class Utils():

    Configs = _Configs
    Cuda = _Cuda
    Prompt = _Prompt
    Storage = _Storage
    Models = _Models
