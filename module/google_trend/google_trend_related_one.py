import time
from pytrends.request import TrendReq
import pandas as pd
from openpyxl import load_workbook
import os

# pytrends 세션 시작
pytrends = TrendReq(hl='ko', tz=360)

# 유튜버 키워드 리스트에서 첫 번째 키워드 선택
keyword = "때잉 플레이리스트"

# 파일 이름
excel_filename = './related_queries.xlsx'

# 연관 검색어를 저장할 데이터프레임 초기화
related_queries_df = pd.DataFrame(columns=['유튜버', '연관 검색어'])

# 선택한 키워드에 대해 연관 검색어 가져오기
pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
related_queries = pytrends.related_queries()[keyword]['top']

if related_queries is not None:
    # 각 연관 검색어를 데이터프레임에 추가
    for _, row in related_queries.iterrows():
        new_row = pd.DataFrame({'유튜버': [keyword], '연관 검색어': [row['query']]})
        related_queries_df = pd.concat([related_queries_df, new_row], ignore_index=True)

# 데이터프레임을 Excel 파일로 저장
with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='w') as writer:
    # 새 시트에 데이터프레임 쓰기
    related_queries_df.to_excel(writer, sheet_name=f'{keyword}_연관검색어', index=False)

print(f"'{keyword}'의 연관 검색어가 '{excel_filename}'에 추가되었습니다.")
