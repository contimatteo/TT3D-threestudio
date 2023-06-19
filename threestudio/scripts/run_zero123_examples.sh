# Reconstruct Anya using latest Zero123XL, in <2000 steps.
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range;bs=4;min/max=0.02/0.98;lr=0.05;max_step=1999;wbgr=0.5;random_bg" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.guidance.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05

# Narrow the range of noise level. Converges to a slightly better result, but has
# more spikes in total loss during training.
# XL
debugpy-run launch.py -- --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl2" \
    system.loggers.wandb.name="anya_zero123XL_20230604" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# default zero123 ckpt
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl2" \
    system.loggers.wandb.name="anya_zero123_105000" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# Anya
# -----------------

# default:
# `rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])`
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="anya_default" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# don't add final term:
# rgb = rgba[..., :3] * rgba[..., 3:] #+ (1 - rgba[..., 3:])
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="anya_dont_add" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# don't premultiply - i.e. leave RGB unchanged
# rgb = rgba[..., :3] #* rgba[..., 3:] #+ (1 - rgba[..., 3:])
python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="anya_dont_premultiply" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# Bollywood

# default:
# `rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])`
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="bollywood_default" \
    data.image_path=./load/images/bollywood_actress_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# don't add final term:
# rgb = rgba[..., :3] * rgba[..., 3:] #+ (1 - rgba[..., 3:])
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="bollywood_dont_add" \
    data.image_path=./load/images/bollywood_actress_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# don't premultiply - i.e. leave RGB unchanged
# rgb = rgba[..., :3] #* rgba[..., 3:] #+ (1 - rgba[..., 3:])
python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-dont_composite" \
    system.loggers.wandb.name="bollywood_dont_premultiply" \
    data.image_path=./load/images/bollywood_actress_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9
