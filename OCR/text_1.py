import cv2
import pytesseract
from datetime import datetime

# Tesseract 경로 설정 (필요한 경우)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# 이미지 파일 경로
image_path = '/Users/leechanhyeon/Desktop/ocr/test5.png'

# 이미지 로딩 및 전처리
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tesseract를 사용한 텍스트 추출
custom_config = r'--oem 3 --psm 6 -l kor'  # 한글+영어 인식 설정
extracted_text = pytesseract.image_to_string(gray_image, config=custom_config)

# 현재 날짜와 시간 가져오기
now = datetime.now()
formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')

# 로그 파일에 인식 결과 저장
log_file = 'ocr_log.txt'
with open(log_file, 'a') as file:
    file.write(f'[{formatted_date}] {extracted_text}\n')

print("OCR 완료. 결과가 로그 파일에 저장되었습니다.")
