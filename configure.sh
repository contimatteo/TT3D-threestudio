###

exit 0

ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_DIR="${ROOT_DIR}/prompts"
OUT_DIR="${ROOT_DIR}/outputs/configure_script_cache"

GPU=0
PROMPT_FILE="${PROMPT_DIR}/test.t3bench.n1.txt"

export HF_HUB_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HOME="${ROOT_DIR}/cache/huggingface"


### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "dreamfusion-sd" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-DreamFusionSD/" \
  --train-steps="100"


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "fantasia3d" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-Fantasia3D/" \
  --train-steps="100,100"


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "prolificdreamer" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-ProlificDreamer/" \
  --train-steps="100,100,100"


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "magic3d" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-Magic3D/" \
  --train-steps="100,100"


### 
### TEXT-MESH
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "textmesh" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-TextMesh/" \
  --train-steps="100"


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --model "hifa" \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/Threestudio-HiFA/" \
  --train-steps="100"


###

# remove generated samples (they are out-of-scope ...)
rm -rf $OUT_DIR
