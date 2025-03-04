import pandas as pd
import time
import warnings
import re
from datetime import datetime, timedelta

from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
#from selenium import webdriver  # 기존 webdriver 대신 undetected_chromedriver 사용

# 캡챠 라이브러리
from tiktok_captcha_solver import SeleniumSolver
import undetected_chromedriver as uc

warnings.filterwarnings('ignore')

# 상대 날짜 문자열을 실제 날짜(YYYY-MM-DD)로 변환
def parse_relative_date(rel_date_str, current_date=None):

    if current_date is None:
        current_date = datetime.now()
    rel_date_str = rel_date_str.strip()

    # 만약 rel_date_str가 "M-D" 형식(ex)2-5)이라면 현재 연도를 붙여 반환
    if re.match(r'^\d{1,2}-\d{1,2}$', rel_date_str):
        parts = rel_date_str.split('-')
        month = int(parts[0])
        day = int(parts[1])
        return f"{current_date.year:04d}-{month:02d}-{day:02d}"

    # 이미 절대 날짜 형식이면 그대로 반환함
    if "-" in rel_date_str:
        return rel_date_str

    # "방금 전"인 경우
    if "방금" in rel_date_str:
        return current_date.strftime("%Y-%m-%d")
    
    # "분 전"인 경우 (날짜 변동 없이 현재 날짜)
    if "분" in rel_date_str:
        return current_date.strftime("%Y-%m-%d")
    
    # "시간 전"인 경우
    if "시간" in rel_date_str:
        try:
            if "한" in rel_date_str:
                hours = 1
            else:
                hours = int("".join(filter(str.isdigit, rel_date_str)))
        except:
            hours = 0
        new_date = current_date - timedelta(hours=hours)
        return new_date.strftime("%Y-%m-%d")
    
    # "일 전"인 경우
    if "일" in rel_date_str:
        try:
            if "한" in rel_date_str:
                days = 1
            else:
                days = int("".join(filter(str.isdigit, rel_date_str)))
        except:
            days = 0
        new_date = current_date - timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")
    
    # "주 전"인 경우 (1주 7일로 설정)
    if "주" in rel_date_str:
        try:
            weeks = int("".join(filter(str.isdigit, rel_date_str)))
        except:
            weeks = 1
        new_date = current_date - timedelta(weeks=weeks)
        return new_date.strftime("%Y-%m-%d")
    
    # "개월 전"인 경우 (1개월 30일로 설정)
    if "개월" in rel_date_str:
        try:
            months = int("".join(filter(str.isdigit, rel_date_str)))
        except:
            months = 1
        new_date = current_date - timedelta(days=30*months)
        return new_date.strftime("%Y-%m-%d")
    
    # "년 전"인 경우 (1년 365일로 설정)
    if "년" in rel_date_str:
        try:
            years = int("".join(filter(str.isdigit, rel_date_str)))
        except:
            years = 1
        new_date = current_date - timedelta(days=365*years)
        return new_date.strftime("%Y-%m-%d")
    
    # 변환 실패 시 현재 날짜 반환
    return current_date.strftime("%Y-%m-%d")

PAGE_LOAD_WAIT = 5         # 프로필 페이지 로딩 대기 시간
VIDEO_LOAD_WAIT = 3        # 동영상 페이지 로딩 대기 시간
COMMENTS_LOAD_WAIT = 4     # 댓글 페이지 로딩 대기 시간
SCROLL_DELAY = 0.5         # slow_scroll 함수 내 각 단계 대기 시간

# undetected_chromedriver를 사용하여 Selenium driver 초기화 및 캡챠 솔버 연결
def initialize_driver():
    chrome_options = Options()
    # 기존 experimental_option 제거 --> 캡챠 솔버랑 같이 불가능
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    
    # undetected_chromedriver 사용
    driver = uc.Chrome(options=chrome_options)
    driver.implicitly_wait(4)
    
    # 캡챠 솔버 설정 (개인 API 키 사용)
    api_key = ""
    driver.captcha_solver = SeleniumSolver(
        driver,
        api_key,
        mouse_step_size=1,   # 마우스 속도 조정
        mouse_step_delay_ms=10
    )
    
    return driver

# 페이지 로드 후 캡챠가 있으면 해결하는 함수
def get_with_captcha(driver, url, wait_time):
    driver.get(url)
    time.sleep(wait_time)
    try:
        driver.captcha_solver.solve_captcha_if_present()
    except Exception as e:
        print("Captcha solve error:", e)

# slow_scroll 함수 추가: 페이지를 여러 단계로 나누어 스크롤
def slow_scroll(driver, steps=10, delay=SCROLL_DELAY):
    """
    페이지를 여러 단계로 나누어 스크롤합니다.
    steps: 전체 페이지를 몇 단계로 나눌 것인지 (기본 10단계)
    delay: 각 단계마다 대기할 시간(초)
    """
    total_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(1, steps + 1):
        scroll_position = total_height * i / steps
        driver.execute_script("window.scrollTo(0, arguments[0]);", scroll_position)
        time.sleep(delay)

# 틱톡 인플루언서 프로필 페이지에서 동영상 링크 (현 5개) 추출
def get_latest_videos(driver, user_profile_url, max_results=5):
    get_with_captcha(driver, user_profile_url, PAGE_LOAD_WAIT)  # 캡챠 확인 포함

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

        slow_scroll(driver, steps=10, delay=SCROLL_DELAY)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1

    print(f"수집된 동영상 개수: {len(video_urls)}")
    return list(video_urls)[:max_results]

# 틱톡 동영상 페이지에서 영상 정보와 채널(사용자)정보 추출
def get_video_info(driver, video_url):
    get_with_captcha(driver, video_url, VIDEO_LOAD_WAIT)  # 캡챠 확인 포함

    try:
        more_button = driver.find_element(By.XPATH, "//button[contains(text(), '더보기')]")
        driver.execute_script("arguments[0].click();", more_button)
        time.sleep(1)
    except Exception:
        pass

    try:
        desc_elem = driver.find_element(By.XPATH, "//h1[contains(@data-e2e, 'video-desc')]")
        description = desc_elem.text.strip()
    except Exception:
        description = ""
    
    try:
        like_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='like-count']")
        like_count = like_elem.text.strip()
    except Exception:
        like_count = "0"
    
    try:
        comment_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='comment-count']")
        comment_count = comment_elem.text.strip()
    except Exception:
        comment_count = "0"
    
    try:
        share_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='share-count']")
        share_count = share_elem.text.strip()
    except Exception:
        share_count = "0"
    
    try:
        user_elem = driver.find_element(By.XPATH, "//h2[contains(@data-e2e, 'user-title')]")
        channel_title = user_elem.text.strip()
    except Exception:
        channel_title = ""
    
    try:
        date_elem = driver.find_element(By.XPATH, "//span[@data-e2e='browser-nickname']/span[last()]")
        upload_date = date_elem.text.strip()
    except Exception:
        upload_date = datetime.now().strftime('%Y-%m-%d')
    
    # 업로드 날짜가 상대 표현일 경우 실제 날짜로 변환
    upload_date = parse_relative_date(upload_date)
    
    snippet = {
        'description': description,
        'title': "",
        'publishedAt': upload_date
    }
    statistics = {
        'likeCount': like_count,
        'commentCount': comment_count,
        'shareCount': share_count
    }
    channel_info = {
        'channel_title': channel_title
    }
    
    return snippet, statistics, channel_info

# 틱톡 동영상 페이지 댓글 추출. 텍스트, 작성자, 날짜 정보
def extract_video_comments(driver, video_url, max_comments=50, target_date=None):
    get_with_captcha(driver, video_url, COMMENTS_LOAD_WAIT)  # 캡챠 확인 포함

    prev_count = 0
    while True:
        comment_blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'DivCommentObjectWrapper')]")
        current_count = len(comment_blocks)
        if current_count <= prev_count:
            break
        prev_count = current_count
        slow_scroll(driver, steps=10, delay=SCROLL_DELAY)
        time.sleep(4)

    comment_blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'DivCommentObjectWrapper')]")
    print("찾은 댓글 블록 개수:", len(comment_blocks))
    
    comments = []
    for block in comment_blocks:
        try:
            comment_text = block.find_element(By.XPATH, ".//span[@data-e2e='comment-level-1']/p").text.strip()
        except Exception as e:
            print("댓글 텍스트 추출 실패:", e)
            comment_text = ""
        
        try:
            username = block.find_element(By.XPATH, ".//div[@data-e2e='comment-username-1']//p").text.strip()
        except Exception as e:
            print("작성자 추출 실패:", e)
            username = "Unknown"
        
        try:
            raw_date = block.find_element(By.XPATH, ".//div[contains(@class, 'DivCommentSubContentWrapper')]//span").text.strip()
            comment_date = parse_relative_date(raw_date)
        except Exception as e:
            print("날짜 추출 실패:", e)
            comment_date = "Unknown"
        
        comments.append([comment_text, username, comment_date, video_url])
        if len(comments) >= max_comments:
            break

    comments_df = pd.DataFrame(comments, columns=['comment', 'username', 'date', 'video_url'])
    if target_date:
        comments_df = comments_df[comments_df['date'] == target_date]
        print(f"Filtered 댓글 개수 for {target_date}: {len(comments_df)}")
        
    print(f"수집된 댓글 개수: {len(comments_df)}")
    return comments_df

# 정보 엑셀 파일에 저장. 기존 파일이 있을 경우 기존 데이터에 추가
def save_statistics_to_excel(stats_df, file_path):
    try:
        existing_stats_df = pd.read_excel(file_path)
        stats_df = pd.concat([existing_stats_df, stats_df], ignore_index=True)
    except FileNotFoundError:
        pass
    stats_df.to_excel(file_path, index=False)
    print(f"통계 정보 {file_path} 저장.")

# excel에 댓글 정보 저장. 기존 파일이 있을 경우 기존 데이터에 추가
def save_comments_to_excel(comments_df, file_path):
    try:
        existing_comments_df = pd.read_excel(file_path)
        comments_df = pd.concat([existing_comments_df, comments_df], ignore_index=True)
    except FileNotFoundError:
        pass
    comments_df.to_excel(file_path, index=False)
    print(f"댓글 정보 {file_path}에 저장")

# 틱톡 사용자 프로필 URL에서 최신 동영상 (현재엔 5개) 정보 추출, 댓글 추출 -> 엑셀 파일 두 개에 저장
def extract_latest_videos_from_channel(user_profile_url, stats_file_path, comments_file_path, max_results=5, target_date=None):
    driver = initialize_driver()
    video_urls = get_latest_videos(driver, user_profile_url, max_results)
    
    for video_url in video_urls:
        snippet, statistics, channel_info = get_video_info(driver, video_url)
        current_date = datetime.now().strftime('%Y-%m-%d')
        upload_date = snippet['publishedAt']
        
        stats_data = {
            'video_url': video_url,
            'upload_date': [upload_date],
            'date': [current_date],
            'like_count': [statistics.get('likeCount')],
            'comment_count': [statistics.get('commentCount')],
            'share_count': [statistics.get('shareCount')],
            'description': [snippet.get('description', '')]
        }
        stats_df = pd.DataFrame(stats_data)
        save_statistics_to_excel(stats_df, stats_file_path)
        
        comments_df = extract_video_comments(driver, video_url, target_date=target_date)
        save_comments_to_excel(comments_df, comments_file_path)
    
    driver.quit()

if __name__ == "__main__":
    target_date = "2025-02-15"
    file_suffix = target_date[2:4] + target_date[5:7] + target_date[8:10]
    STATS_FILE_PATH = f'/Users/syb/Desktop/tiktok_crawling/tiktok_data_date/tiktok_stats_ex1_{file_suffix}.xlsx'
    COMMENTS_FILE_PATH = f'/Users/syb/Desktop/tiktok_crawling/tiktok_data_date/tiktok_comments_ex2_{file_suffix}.xlsx'
    
    user_profiles = [
        "https://www.tiktok.com/@__ralral__"
    ]
    
    for profile_url in user_profiles:
        extract_latest_videos_from_channel(profile_url, STATS_FILE_PATH, COMMENTS_FILE_PATH, max_results=5, target_date=target_date)
