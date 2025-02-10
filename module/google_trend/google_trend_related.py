import time
from pytrends.request import TrendReq
import pandas as pd

# pytrends 세션 시작
pytrends = TrendReq(hl='ko', tz=360)

# 유튜버 키워드 리스트
keywords = ["오킹", "한동숙", "뻑가", "깡 스타일리스트", "때잉 플레이리스트"]

# 연관 검색어를 저장할 데이터프레임 초기화
related_queries_df = pd.DataFrame(columns=['유튜버', '연관 검색어'])

# 각 유튜버 키워드에 대해 연관 검색어 가져오기
for keyword in keywords:
    pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
    related_queries = pytrends.related_queries()[keyword]['top']
    
    if related_queries is not None:
        # 각 연관 검색어를 데이터프레임에 추가
        for _, row in related_queries.iterrows():
            new_row = pd.DataFrame({'유튜버': [keyword], '연관 검색어': [row['query']]})
            related_queries_df = pd.concat([related_queries_df, new_row], ignore_index=True)
    print(related_queries)
    time.sleep(910)  # 요청 간 대기

# 데이터프레임을 Excel 파일로 저장
related_queries_df.to_excel('related_queries.xlsx', index=False)

print("연관 검색어가 'related_queries.xlsx'로 저장되었습니다.")
