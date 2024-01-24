###

exit 0

### DREAMFUSION
CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "dreamfusion-sd" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --train-steps=100
