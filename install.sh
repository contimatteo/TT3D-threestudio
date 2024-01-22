#!/bin/bash

pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install -r requirements.txt
