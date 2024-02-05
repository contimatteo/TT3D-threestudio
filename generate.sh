###

exit 0

### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "dreamfusion-sd" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-DreamFusionSD/ \
  --train-steps="1000"


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "fantasia3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-Fantasia3D/ \
  --train-steps="900,100"


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "prolificdreamer" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-ProlificDreamer/ \
  --train-steps="800,100,100"


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "magic3d" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-Magic3D/ \
  --train-steps="900,100"


### 
### TEXT-MESH
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "textmesh" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-TextMesh/ \
  --train-steps="1000"


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_generate.py \
  --model "hifa" \
  --prompt-file /media/data2/mconti/TT3D/prompts/test.t3bench.n1.txt \
  --out-path /media/data2/mconti/TT3D/outputs/Threestudio-HiFA/ \
  --train-steps="1000"
