# 3-2-DeepLearning-Project
Object detection using stable diffusion model  
https:/github.com/seunghan11 폴더는 참조한 stable diffusion에 수정된 version입니다.  
TinySSD.py가 사용된 모델이며, master branch는 사용된 dataset 및 각 dataset에 대한 결과입니다.

## Model
reference site: https://d2l.ai/chapter_computer-vision/ssd.html  
<img width="341" alt="TinySSD" src="https://github.com/seunghan11/3-2-DeepLearning-Project/assets/88572826/6e191785-a9a4-444a-b8d4-6ce78715e2fc">

## Dataset&Hyperparameter
making_transform_image.py가 이미지 증강을 위해 사용된 코드입니다.
dataset1은 crop data  
dataset2는 crop data + augmented data  
dataset3은 crop data + generated data  
dataset4은 crop data + augmented data + generated data 입니다.

각 폴더명이 사용된 hyperparmeter입니다.  
master branch 내 pdf가 프로젝트 결과 보고서입니다.

## Evaluation
model&dataset evaluation code:  
https://colab.research.google.com/drive/1GvWJv_4-veql7QZloot9gckei9jZb_Ln?usp=sharing

Bounding Box label code:  
https://drive.google.com/file/d/1Fqq45OTED3WQawWCcd_jF3UmR85qW0I5/view?usp=sharing

