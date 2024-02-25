###

exit 1


GPU=3
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"


export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export HF_HOME="${ROOT_DIR}/cache/huggingface"


### 
### DREAMFUSION
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "dreamfusion-sd" \
#   --source-path "${OUT_DIR}/Threestudio-DreamFusion/" \
#   --skip-existing

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "dreamfusion-if" \
#   --source-path "${OUT_DIR}/Threestudio-DreamFusion/" \
#   --skip-existing

### 
### FANTASIA-3D
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "fantasia3d" \
#   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-Fantasia3D/" \
#   --skip-existing

### 
### PROFILIC-DREAMER
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "prolificdreamer" \
#   --source-path "${OUT_DIR}/Threestudio-ProlificDreamer/" \
#   --skip-existing

### 
### MAGIC-3D
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "magic3d-sd" \
#   --source-path "${OUT_DIR}/Threestudio-Magic3D/" \
#   --skip-existing

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "magic3d-if" \
#   --source-path "${OUT_DIR}/Threestudio-Magic3D/" \
#   --skip-existing

### 
### TEXT-MESH
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "textmesh-sd" \
#   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-TextMesh/" \
#   --skip-existing

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "textmesh-sd" \
#   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-TextMesh-nopriors/" \
#   --skip-existing

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "textmesh-if" \
#   --source-path "${OUT_DIR}/Threestudio-TextMesh-nopriors/" \
#   --skip-existing

### CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
###   --model "textmesh-if" \
###   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-TextMesh/" \
###   --skip-existing

### 
### HIFA
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "hifa" \
#   --source-path "${OUT_DIR}/Threestudio-HiFA/" \
#   --skip-existing

### 
### SJC
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "sjc" \
#   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-SJC/" \
#   --skip-existing

### 
### LATENT-NERF
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "latentnerf" \
#   --source-path "/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}/Threestudio-LatentNerf/" \
#   --skip-existing
