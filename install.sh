#!/bin/bash

pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install -r requirements-custom.txt
pip install "git+https://github.com/ashawkey/envlight.git"
pip install "git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2"
pip install "git+https://github.com/NVlabs/nvdiffrast.git"
pip install -v "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
