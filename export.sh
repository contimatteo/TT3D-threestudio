###

exit 0

### 
### DREAMFUSION
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "dreamfusion-sd" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-DreamFusionSD/


### 
### FANTASIA-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "fantasia3d" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-Fantasia3D/


### 
### PROFILIC-DREAMER
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "prolificdreamer" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-ProlificDreamer/


### 
### MAGIC-3D
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "magic3d" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-Magic3D/


### 
### TEXT-MESH
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "textmesh" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-TextMesh/


### 
### HIFA
### 

CUDA_VISIBLE_DEVICES=1 python3 tt3d_export.py \
  --model "hifa" \
  --source-path /media/data2/mconti/TT3D/outputs/Threestudio-HiFA/
