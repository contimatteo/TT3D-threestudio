###

exit 1


###############################################################################


GPU=1
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
PROMPT_FILE="${ROOT_DIR}/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"


export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export HF_HOME="${ROOT_DIR}/cache/huggingface"


###############################################################################


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_embeddings.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${ROOT_DIR}/outputs/cache/embeddings/Threestudio/" \
  --train-steps="1"


###############################################################################  

### 
### DREAMFUSION
### 

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "dreamfusion-sd" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-DreamFusion/" \
#   --train-steps="2650" \
#   --skip-existing

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "dreamfusion-if" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-DreamFusion/" \
#   --train-steps="3500" \
#   --skip-existing

### 
### FANTASIA-3D
### 

### TODO: need to add support for priors shape initialization
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "fantasia3d" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-Fantasia3D/" \
#   --train-steps="1600,1350" \
#   --use-priors \
#   --skip-existing

### 
### PROFILIC-DREAMER
### 

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "prolificdreamer" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-ProlificDreamer/" \
#   --train-steps="650,800,500" \
#   --skip-existing

### CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
###   --model "prolificdreamer" \
###   --prompt-file $PROMPT_FILE \
###   --out-path "${OUT_DIR}/Threestudio-ProlificDreamer/" \
###   --train-steps="950,550,300" \
###   --skip-existing

### 
### MAGIC-3D
### 

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "magic3d-sd" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-Magic3D/" \
#   --train-steps="1300,1300" \
#   --skip-existing

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "magic3d-if" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-Magic3D/" \
#   --train-steps="1700,1300" \
#   --skip-existing

### 
### TEXT-MESH
### 

### TODO: need to add support for priors shape initialization
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "textmesh-sd" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-TextMesh/" \
#   --train-steps="2500" \
#   --use-priors \
#   --skip-existing

### TODO: need to add support for priors shape initialization
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "textmesh-if" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-TextMesh/" \
#   --train-steps="3500" \
#   --use-priors \
#   --skip-existing

### 
### HIFA
### 

### INFO: OK
# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
#   --model "hifa" \
#   --prompt-file $PROMPT_FILE \
#   --out-path "${OUT_DIR}/Threestudio-HiFA/" \
#   --train-steps="2650" \
#   --skip-existing
