###

exit 0

CUDA_VISIBLE_DEVICES=3 python3 launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a shark" trainer.max_steps=100 system.prompt_processor.spawn=false
