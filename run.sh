###

exit 0

python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="$prompt" trainer.max_steps=100 system.prompt_processor.spawn=false
