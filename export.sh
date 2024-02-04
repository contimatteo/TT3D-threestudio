###

exit 0

### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "dreamfusion-sd" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-DreamFusionSD/outputs


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "fantasia3d" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Fantasia3D/outputs


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "prolificdreamer" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-ProlificDreamer/outputs


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "magic3d" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-Magic3D/outputs


### 
### TEXT-MESH
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "textmesh" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-TextMesh/outputs


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "hifa" \
  --source-path /media/data2/mconti/TT3D/models/Threestudio-HiFA/outputs
