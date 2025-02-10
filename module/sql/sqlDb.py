import mysql.connector
import pandas as pd

# MySQL 데이터베이스에 연결
connection = mysql.connector.connect(
    host="localhost",        # 데이터베이스 호스트 이름
    user="root",             # 데이터베이스 사용자 이름
    password="1234",         # 데이터베이스 비밀번호
    database="followSupport" # 데이터베이스 이름
)

# 연결 확인
if connection.is_connected():
    print("Successfully connected to the database")

cursor = connection.cursor()

# 엑셀 파일 읽기
excel_file_path = '../../xlsx/stats0709.xlsx'  # 실제 파일 경로로 변경
df = pd.read_excel(excel_file_path)

excel_file_path = '../../xlsx/labeling_0709.xlsx'  # 실제 파일 경로로 변경
df2 = pd.read_excel(excel_file_path)

#데이터프레임의 각 행을 Youtuber 테이블에 삽입
for index, row in df.iterrows():
    insert_query = """
    INSERT INTO VideoStat (video_id, upload_date, date, view_count, like_count, comment_count, subscriber_count, channel_id, channel_title, channel_description, topic_categories, title, description, tags, thumbnails)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = (
        row['video_id'] if pd.notnull(row['video_id']) else None,
        row['upload_date'] if pd.notnull(row['upload_date']) else None,
        row['date'] if pd.notnull(row['date']) else None,
        row['view_count'] if pd.notnull(row['view_count']) else None,
        row['like_count'] if pd.notnull(row['like_count']) else None,
        row['comment_count'] if pd.notnull(row['comment_count']) else None,
        row['subscriber_count'] if pd.notnull(row['subscriber_count']) else None,
        row['channel_id'] if pd.notnull(row['channel_id']) else None,
        row['channel_title'] if pd.notnull(row['channel_title']) else None,
        row['channel_description'] if pd.notnull(row['channel_description']) else None,
        row['topic_categories'] if pd.notnull(row['topic_categories']) else None,
        row['title'] if pd.notnull(row['title']) else None,
        row['description'] if pd.notnull(row['description']) else None,
        row['tags'] if pd.notnull(row['tags']) else None,
        row['thumbnails'] if pd.notnull(row['thumbnails']) else None
    )
    cursor.execute(insert_query, data)

for index, row in df2.iterrows():
        insert_query = """
        INSERT INTO Comments (comment, author, date, num_likes, video_id, emotion, object)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        data = (
            row['comment'] if pd.notnull(row['comment']) else None,
            row['author'] if pd.notnull(row['author']) else None,
            row['date'] if pd.notnull(row['date']) else None,
            row['num_likes'] if pd.notnull(row['num_likes']) else None,
            row['video_id'] if pd.notnull(row['video_id']) else None,
            row['emotion'] if pd.notnull(row['emotion']) else None,
            row['object'] if pd.notnull(row['object']) else None,
        )
        cursor.execute(insert_query, data)
        


# 커밋하여 데이터베이스에 변경 사항 적용
connection.commit()
print("데이터 삽입 완료!")

# 커서와 연결 종료
cursor.close()
connection.close()

print("Data inserted successfully.")
