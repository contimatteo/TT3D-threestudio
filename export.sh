###

exit 0

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs"

GPU=0

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export HF_HOME="${ROOT_DIR}/cache/huggingface"


### 
### DREAMFUSION
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "dreamfusion-sd" \
#   --source-path "${OUT_DIR}/Threestudio-DreamFusion/"

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "dreamfusion-if" \
  --source-path "${OUT_DIR}/Threestudio-DreamFusion/"


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "fantasia3d" \
  --source-path "${OUT_DIR}/Threestudio-Fantasia3D/"


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "prolificdreamer" \
  --source-path "${OUT_DIR}/Threestudio-ProlificDreamer/"


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "magic3d" \
  --source-path "${OUT_DIR}/Threestudio-Magic3D/"


### 
### TEXT-MESH
### 

# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
#   --model "textmesh-sd" \
#   --source-path "${OUT_DIR}/Threestudio-TextMesh/"

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "textmesh-if" \
  --source-path "${OUT_DIR}/Threestudio-TextMesh/"


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
  --model "hifa" \
  --source-path "${OUT_DIR}/Threestudio-HiFA/"
