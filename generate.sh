###

exit 0


GPU=0
PROMPT="test_t3bench_n1"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${PROMPT}"
PROMPT_FILE="${ROOT_DIR}/prompts/${PROMPT}.txt"


export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export HF_HOME="${ROOT_DIR}/cache/huggingface"


### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "dreamfusion-sd" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-DreamFusion/" \
  --train-steps="1000"

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "dreamfusion-if" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-DreamFusion/" \
#   --train-steps="1000"


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
  --train-steps="400,400,200"


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

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "textmesh-sd" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-TextMesh/" \
  --train-steps="1000"

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
