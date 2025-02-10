from google.cloud import vision_v1
import io

# 이미지에서 텍스트 추출
def detect_text_in_image(image_path, client):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        print(f"Detected text in {image_path}:")
        for text in texts:
            print(text.description)
    else:
        print(f"No text detected in {image_path}.")

# Google Cloud Vision 클라이언트 설정
def setup_vision_client():
    return vision_v1.ImageAnnotatorClient.from_service_account_file('myKey.json')

# 메인 함수
def main():
    image_path = './a.png'  # 텍스트를 추출할 이미지 파일의 경로

    client = setup_vision_client()
    detect_text_in_image(image_path, client)

if __name__ == '__main__':
    main()
