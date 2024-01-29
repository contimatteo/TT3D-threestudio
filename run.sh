###

exit 0

### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "dreamfusion-sd" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/Threestudio-DreamFusionSD/outputs \
  --train-steps=100

CUDA_VISIBLE_DEVICES=3 python3 tt3d_export.py \
  --model "dreamfusion-sd" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-DreamFusionSD/outputs


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "fantasia3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/Threestudio-Fantasia3D/outputs \
  --train-steps=100

CUDA_VISIBLE_DEVICES=3 python3 tt3d_export.py \
  --model "fantasia3d" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Fantasia3D/outputs


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "prolificdreamer" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/Threestudio-ProlificDreamer/outputs \
  --train-steps=100

CUDA_VISIBLE_DEVICES=3 python3 tt3d_export.py \
  --model "prolificdreamer" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-ProlificDreamer/outputs


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "magic3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/Threestudio-Magic3D/outputs \
  --train-steps=100

CUDA_VISIBLE_DEVICES=3 python3 tt3d_export.py \
  --model "magic3d" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Magic3D/outputs


### 
### TEXT-MESH
### 

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --model "textmesh" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.v1.n2.txt \
  --out-path /media/data2/mconti/TT3D/models/Threestudio-TextMesh/outputs \
  --train-steps=100

CUDA_VISIBLE_DEVICES=3 python3 tt3d_export.py \
  --model "textmesh" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-TextMesh/outputs
