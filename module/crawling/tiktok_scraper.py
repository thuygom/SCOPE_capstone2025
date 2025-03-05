import os
import time
import logging
import warnings
import re
import pandas as pd
import undetected_chromedriver as uc
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By

warnings.filterwarnings('ignore')

# ✅ 상대 날짜 문자열을 실제 날짜(YYYY-MM-DD)로 변환
def parse_relative_date(rel_date_str, current_date=None):
    if current_date is None:
        current_date = datetime.now()
    rel_date_str = rel_date_str.strip()

    if re.match(r'^\d{1,2}-\d{1,2}$', rel_date_str):
        month, day = map(int, rel_date_str.split('-'))
        return f"{current_date.year:04d}-{month:02d}-{day:02d}"

    if "-" in rel_date_str:
        return rel_date_str

    if "방금" in rel_date_str or "분" in rel_date_str:
        return current_date.strftime("%Y-%m-%d")

    if "시간" in rel_date_str:
        hours = int("".join(filter(str.isdigit, rel_date_str))) if any(c.isdigit() for c in rel_date_str) else 1
        return (current_date - timedelta(hours=hours)).strftime("%Y-%m-%d")

    if "일" in rel_date_str:
        days = int("".join(filter(str.isdigit, rel_date_str))) if any(c.isdigit() for c in rel_date_str) else 1
        return (current_date - timedelta(days=days)).strftime("%Y-%m-%d")

    if "주" in rel_date_str:
        weeks = int("".join(filter(str.isdigit, rel_date_str))) if any(c.isdigit() for c in rel_date_str) else 1
        return (current_date - timedelta(weeks=weeks)).strftime("%Y-%m-%d")

    if "개월" in rel_date_str:
        months = int("".join(filter(str.isdigit, rel_date_str))) if any(c.isdigit() for c in rel_date_str) else 1
        return (current_date - timedelta(days=30 * months)).strftime("%Y-%m-%d")

    if "년" in rel_date_str:
        years = int("".join(filter(str.isdigit, rel_date_str))) if any(c.isdigit() for c in rel_date_str) else 1
        return (current_date - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    return current_date.strftime("%Y-%m-%d")

# === 전역 시간 상수 (필요에 따라 조정하기) ===
PAGE_LOAD_WAIT = 5         # 프로필 페이지 로딩 대기 시간
VIDEO_LOAD_WAIT = 7        # 동영상 페이지 로딩 대기 시간
COMMENTS_LOAD_WAIT = 5     # 댓글 페이지 로딩 대기 시간
SCROLL_DELAY = 0.5         # slow_scroll 함수 내 각 단계 대기 시간

# ✅ `undetected_chromedriver`를 활용한 드라이버 초기화
def initialize_driver():
    options = uc.ChromeOptions()
    options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 1  # 1: 허용, 2: 차단
    })
    options.add_argument("--disable-popup-blocking")  # 팝업 차단 해제
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")  # 자동화 탐지 방지

    driver = uc.Chrome(options=options, enable_cdp_events=True, incognito=True)
    driver.implicitly_wait(5)
    
    return driver

# ✅ 페이지를 천천히 스크롤
def slow_scroll(driver, steps=10, delay=0.5):
    total_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(1, steps + 1):
        scroll_position = total_height * i / steps
        driver.execute_script("window.scrollTo(0, arguments[0]);", scroll_position)
        time.sleep(delay)

# ✅ 틱톡 프로필에서 최신 동영상 링크 추출
def get_latest_videos(driver, user_profile_url, max_results=5):
    driver.get(user_profile_url)
    #time.sleep(1)  # 페이지 로딩 대기

    video_urls = set()  # 집합으로 선언

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0

    while len(video_urls) < max_results and scroll_attempts < 10:
        video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/video/')]")
        for video_elem in video_elements:
            url = video_elem.get_attribute('href')
            if url and '/video/' in url:
                try:
                    # 고정 게시물일 경우 "고정됨"이라는 텍스트가 포함된 요소를 찾는다.
                    pinned_indicator = video_elem.find_element(By.XPATH, ".//*[contains(text(), '고정')]")
                    pinned = True
                except:
                    pinned = False
                
                if not pinned:
                    video_urls.add(url)  # 집합에서는 add() 사용
                
            if len(video_urls) >= max_results:
                break

        slow_scroll(driver, steps=5, delay=SCROLL_DELAY)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1

    print(f"수집된 동영상 개수: {len(video_urls)}")
    return list(video_urls)[:max_results]

# ✅ 틱톡 동영상 정보 추출
def get_video_info(driver, video_url):
    driver.get(video_url)
    time.sleep(7)

    try:
        more_button = driver.find_element(By.XPATH, "//button[contains(text(), '더보기')]")
        driver.execute_script("arguments[0].click();", more_button)
        time.sleep(1)
    except Exception:
        pass


    try:
        desc_elem = driver.find_element(By.XPATH, "//h1[contains(@data-e2e, 'video-desc')]")
        description = desc_elem.text.strip()
    except:
        description = ""

    try:
        like_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='like-count']")
        like_count = like_elem.text.strip()
    except:
        like_count = "0"

    try:
        comment_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='comment-count']")
        comment_count = comment_elem.text.strip()
    except:
        comment_count = "0"

    try:
        share_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='share-count']")
        share_count = share_elem.text.strip()
    except:
        share_count = "0"

    try:
        user_elem = driver.find_element(By.XPATH, "//h2[contains(@data-e2e, 'user-title')]")
        channel_title = user_elem.text.strip()
    except:
        channel_title = ""

    try:
        date_elem = driver.find_element(By.XPATH, "//span[@data-e2e='browser-nickname']/span[last()]")
        upload_date = date_elem.text.strip()
    except:
        upload_date = datetime.now().strftime('%Y-%m-%d')
    
    upload_date = parse_relative_date(upload_date)
    
    return {
        "video_url": video_url,
        "description": description,
        "like_count": like_count,
        "comment_count": comment_count,
        "share_count": share_count,
        "channel_title": channel_title,
        "upload_date": upload_date
    }

def extract_video_comments(driver, video_url, max_comments=50):
    """틱톡 동영상 댓글을 추출하는 함수."""
    logging.info(f"댓글 추출 시작: {video_url}")
    driver.get(video_url)
    time.sleep(5)

    # ✅ 댓글을 더 불러오기 위해 스크롤 추가 (더보기 버튼이 없을 때 사용)
    slow_scroll(driver, steps=5, delay=2)  

    comments = []
    comment_blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'DivCommentObjectWrapper')]")
    logging.info(f"찾은 댓글 블록 개수: {len(comment_blocks)}")

    for block in comment_blocks[:max_comments]:
        try:
            comment_text = block.find_element(By.XPATH, ".//span[@data-e2e='comment-level-1']/p").text.strip()
            username = block.find_element(By.XPATH, ".//div[@data-e2e='comment-username-1']//p").text.strip()
            raw_date = block.find_element(By.XPATH, ".//div[contains(@class, 'DivCommentSubContentWrapper')]//span").text.strip()
            comments.append([comment_text, username, raw_date, video_url])
        except Exception as e:
            logging.warning(f"댓글 추출 오류: {e}")

    return pd.DataFrame(comments, columns=['comment', 'username', 'date', 'video_url'])

# ✅ TikTok 크롤링 실행
def extract_latest_videos_from_channel(user_profile_url, stats_file_path, comments_file_path, max_results=5):
    driver = initialize_driver()
    try:
        video_urls = get_latest_videos(driver, user_profile_url, max_results)
        
        stats_data = []
        comments_data = []
        
        for video_url in video_urls:
            video_info = get_video_info(driver, video_url)
            stats_data.append(video_info)

            comments_df = extract_video_comments(driver, video_url)
            comments_data.append(comments_df)
        
    finally:
        driver.quit()  # ✅ 크롬 드라이버를 안전하게 종료
    
    stats_df = pd.DataFrame(stats_data) if stats_data else pd.DataFrame(columns=["video_url", "description", "like_count", "comment_count", "share_count", "channel_title", "upload_date"])
    stats_df.to_excel(stats_file_path, index=False)
    print(f"통계 정보 {stats_file_path} 저장 완료.")

    if comments_data:
        comments_df = pd.concat(comments_data, ignore_index=True)
    else:
        comments_df = pd.DataFrame(columns=["comment", "username", "date", "video_url"])
    comments_df.to_excel(comments_file_path, index=False)
    print(f"댓글 정보 {comments_file_path} 저장 완료.")



# ✅ 메인 실행 코드
if __name__ == "__main__":
    target_date = "2025-03-04"
    file_suffix = target_date[2:4] + target_date[5:7] + target_date[8:10]
    
    STATS_FILE_PATH = os.path.join(os.getcwd(), f'tiktok_stats_ex1_{file_suffix}.xlsx')
    comments_file = f'tiktok_comments_{file_suffix}.xlsx'
    user_profiles = ["https://www.tiktok.com/@__ralral__"]

    for profile_url in user_profiles:
        extract_latest_videos_from_channel(profile_url, STATS_FILE_PATH, comments_file, max_results=5)
