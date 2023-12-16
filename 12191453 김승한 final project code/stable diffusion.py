!git clone https://github.com/seunghan11/stable-diffusion.git
%pip install omegaconf
%pip install invisible-watermark
%pip install einops
%pip install pytorch_lightning
%pip install diffusers
%pip install taming-transformers
%pip install clip
%pip install kornia

%mkdir -p models/ldm/stable-diffusion-v1/

from google.colab import drive
drive.mount('/content/drive')

import shutil
import os

# 원본 파일 경로
source = '/content/drive/MyDrive/Colab Notebooks/DL_set3/sd-v1-4.ckpt'

# # 목적지 디렉토리 경로
# destination = 'models/ldm/stable-diffusion-v1'

# # 파일 이동
# shutil.move(source, os.path.join(destination, 'sd-v1-1.ckpt'))
# # 목적지 파일 경로
# destination_path = '/content/drive/MyDrive/Colab Notebooks/DL_set3/model.ckpt'

# # 파일 복사 및 이름 변경
# shutil.copyfile(source_path, destination_path)

destination = 'models/ldm/stable-diffusion-v1/model.ckpt'

shutil.copyfile(source, destination)

!python /content/stable-diffusion/scripts/txt2img.py --prompt "a photograph of a green banana" --plms