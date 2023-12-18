from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
for num in range(0,8):
    # 이미지 로드
    image_path = 'C:/Users/82105/Desktop/GI/{}.png'.format(200+num)
    original_image = Image.open(image_path)

    # 이미지 변환
    # 1. Flipping (수평 뒤집기)
    flipped_image = ImageOps.mirror(original_image)

    # 2. Saturation 조절 (이미지를 RGB로 변환 후 적용해야 함)
    converter = ImageEnhance.Color(original_image)
    saturated_image = converter.enhance(2.0)  # 2.0배로 색상 강화

    # 3. Brightness 조절
    brightness_converter = ImageEnhance.Brightness(original_image)
    brightened_image = brightness_converter.enhance(1.5)  # 1.5배 밝기 증가
    
    # 4. Rotation (90도 회전)
    rotated_image = original_image.rotate(90, expand=True)
    

    # 이미지를 표시하기 위한 목록 생성
    images = [flipped_image, saturated_image, brightened_image, rotated_image]
    titles = [207+4*num+1, 207+4*num+2, 207+4*num+3, 207+4*num+4]

    # 수정된 이미지들을 파일로 저장하고 파일 경로를 저장하는 리스트 생성
    file_paths = []
    for i, img in enumerate(images, 1):
        img.save(f'C:/Users/82105/Desktop/test/{titles[i-1]}.png')
        file_paths.append(f'C:/Users/82105/Desktop/train/{titles[i-1]}.png')

    file_paths # 파일 경로 반환

