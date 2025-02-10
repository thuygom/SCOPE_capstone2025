import pandas as pd

def preprocess_and_save_excel(file_path):
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path, engine='openpyxl')

    # 6열과 7열 데이터 전처리
    df.iloc[:, 5] = df.iloc[:, 5].apply(preprocess_column)  # 6열 전체 데이터 전처리
    df.iloc[:, 6] = df.iloc[:, 6].apply(preprocess_column2)  # 7열 전체 데이터 전처리

    # 전처리된 데이터를 원본 파일에 저장
    df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"전처리된 데이터가 {file_path}에 저장되었습니다.")

def preprocess_column(data):
    # 6열 데이터 전처리
    # data가 NaN인 경우 처리
    if pd.isna(data):
        return "중립"

    # 문자열이 포함된 경우 처리
    if isinstance(data, str):
        emotions = ['행복', '분노', '슬픔', '웃김', '중립']
        for emotion in emotions:
            if emotion in data:
                return emotion

    # 그 외의 경우 처리 (문자열이 포함되지 않은 경우)
    return "중립"

def preprocess_column2(data):
    # 7열 데이터 전처리
    # data가 NaN인 경우 처리
    if pd.isna(data):
        return "other"

    # 문자열이 포함된 경우 처리
    if isinstance(data, str):
        # youtuber, context, other 키워드 확인
        keywords = ['youtuber', 'context', 'other']
        for keyword in keywords:
            if keyword in data:
                return keyword

    # 그 외의 경우 처리 (문자열이 포함되지 않은 경우)
    return "other"


# 원본 엑셀 파일 경로 설정
file_path = '../xlsx/labeling_0709.xlsx'

# 전처리 및 저장 함수 실행
preprocess_and_save_excel(file_path)
