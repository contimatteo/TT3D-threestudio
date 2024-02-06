###

exit 0

ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_DIR="${ROOT_DIR}/prompts"
OUT_DIR="${ROOT_DIR}/outputs"

GPU=0
PROMPT_FILE="${PROMPT_DIR}/test.t3bench.n10.txt"

export HF_HUB_OFFLINE=1
export HF_HOME="${ROOT_DIR}/cache/huggingface"


### 
### DREAMFUSION
### 

#  --model "dreamfusion-sd" \
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "dreamfusion-if" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-DreamFusionSD/" \
  --train-steps="1000"


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "fantasia3d" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-Fantasia3D/" \
  --train-steps="900,100"


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "prolificdreamer" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-ProlificDreamer/" \
  --train-steps="800,100,100"


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "magic3d" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-Magic3D/" \
  --train-steps="900,100"


### 
### TEXT-MESH
### 

#  --model "textmesh-sd" \
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "textmesh-if" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-TextMesh/" \
  --train-steps="1000"


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "hifa" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-HiFA/" \
  --train-steps="1000"
