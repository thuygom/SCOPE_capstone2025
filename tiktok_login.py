import os
import time
import logging
import warnings
import re
import pandas as pd
import undetected_chromedriver as uc
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

def initialize_driver():

    options = uc.ChromeOptions()
    user_data_dir = os.path.join(os.getcwd(), "chrome_user_data")
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    options.add_argument(f"--user-data-dir={user_data_dir}")
    
    options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 1
    })
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    driver = uc.Chrome(options=options, enable_cdp_events=True)
    driver.implicitly_wait(5)
    return driver

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

def fast_scroll(driver, steps=20, delay=1):
    total_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(1, steps + 1):
        scroll_position = total_height * i / steps
        driver.execute_script("window.scrollTo(0, arguments[0]);", scroll_position)
        time.sleep(delay)

def get_latest_videos(driver, user_profile_url, max_results=30):
    logging.info("프로필 페이지 로드: %s", user_profile_url)
    driver.get(user_profile_url)
    video_urls = set()
    scroll_attempts = 0
    while len(video_urls) < max_results and scroll_attempts < 10:
        video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/video/')]")
        for video_elem in video_elements:
            url = video_elem.get_attribute('href')
            if url and '/video/' in url:
                try:
                    video_elem.find_element(By.XPATH, ".//*[contains(text(), '고정')]")
                    pinned = True
                except Exception:
                    pinned = False
                if not pinned:
                    video_urls.add(url)
            if len(video_urls) >= max_results:
                break
        fast_scroll(driver, steps=20, delay=1)
        scroll_attempts += 1
    logging.info("수집된 동영상 개수: %d", len(video_urls))
    return list(video_urls)[:max_results]

def get_video_info(driver, video_url):
    logging.info("동영상 정보 수집: %s", video_url)
    driver.get(video_url)
    time.sleep(3)
    try:
        more_button = driver.find_element(By.XPATH, "//button[contains(text(), '더보기')]")
        driver.execute_script("arguments[0].click();", more_button)
        time.sleep(1)
    except Exception as e:
        logging.debug("더보기 버튼 없음: %s", e)
    try:
        desc_elem = driver.find_element(By.XPATH, "//h1[contains(@data-e2e, 'video-desc')]")
        description = desc_elem.text.strip()
    except Exception as e:
        logging.debug("동영상 설명 수집 실패: %s", e)
        description = ""
    try:
        like_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='like-count']")
        like_count = like_elem.text.strip()
    except Exception as e:
        logging.debug("좋아요 수 수집 실패: %s", e)
        like_count = "0"
    try:
        comment_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='comment-count']")
        comment_count = comment_elem.text.strip()
    except Exception as e:
        logging.debug("댓글 수 수집 실패: %s", e)
        comment_count = "0"
    try:
        share_elem = driver.find_element(By.XPATH, "//strong[@data-e2e='share-count']")
        share_count = share_elem.text.strip()
    except Exception as e:
        logging.debug("공유 수 수집 실패: %s", e)
        share_count = "0"
    try:
        user_elem = driver.find_element(By.XPATH, "//h2[contains(@data-e2e, 'user-title')]")
        channel_title = user_elem.text.strip()
    except Exception as e:
        logging.debug("채널 이름 수집 실패: %s", e)
        channel_title = ""
    try:
        date_elem = driver.find_element(By.XPATH, "//span[@data-e2e='browser-nickname']/span[last()]")
        upload_date = date_elem.text.strip()
    except Exception as e:
        logging.debug("업로드 날짜 수집 실패: %s", e)
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

def extract_video_comments(driver, video_url, max_scrolls=100):
    logging.info("댓글 추출 시작: %s", video_url)
    driver.get(video_url)
    time.sleep(3)
    last_count = 0
    scroll_attempts = 0
    while scroll_attempts < max_scrolls:
        fast_scroll(driver, steps=20, delay=1)
        comment_blocks = driver.find_elements(By.XPATH, "//div[contains(@class, 'DivCommentObjectWrapper')]")
        current_count = len(comment_blocks)
        logging.info("스크롤 시도 %d, 댓글 블록 개수: %d", scroll_attempts+1, current_count)
        if current_count == last_count:
            break
        last_count = current_count
        scroll_attempts += 1
    comments = []
    for block in comment_blocks:
        try:
            comment_text = block.find_element(By.XPATH, ".//span[@data-e2e='comment-level-1']/p").text.strip()
            username = block.find_element(By.XPATH, ".//div[@data-e2e='comment-username-1']//p").text.strip()
            raw_date = block.find_element(By.XPATH, ".//div[contains(@class, 'DivCommentSubContentWrapper')]//span").text.strip()
            comments.append([comment_text, username, raw_date, video_url])
        except Exception as e:
            logging.warning("댓글 추출 오류: %s", e)
    df = pd.DataFrame(comments, columns=['comment', 'username', 'date', 'video_url'])
    logging.info("동영상 %s - 추출된 댓글 개수: %d", video_url, len(df))
    return df

def extract_latest_videos_from_channel(driver, user_profile_url, stats_file_path, comments_file_path, max_results=30):
    logging.info("프로필에서 동영상 및 댓글 추출 시작: %s", user_profile_url)
    video_urls = get_latest_videos(driver, user_profile_url, max_results)
    stats_data = []
    comments_data = []
    for video_url in video_urls:
        try:
            video_info = get_video_info(driver, video_url)
            stats_data.append(video_info)
            comments_df = extract_video_comments(driver, video_url)
            comments_data.append(comments_df)
        except Exception as e:
            logging.error("동영상 처리 중 오류 발생: %s", e)
    stats_df = pd.DataFrame(stats_data) if stats_data else pd.DataFrame(
        columns=["video_url", "description", "like_count", "comment_count", "share_count", "channel_title", "upload_date"]
    )
    stats_df.to_excel(stats_file_path, index=False)
    logging.info("동영상 통계 정보 저장 완료: %s", stats_file_path)
    if comments_data:
        all_comments_df = pd.concat(comments_data, ignore_index=True)
    else:
        all_comments_df = pd.DataFrame(columns=["comment", "username", "date", "video_url"])
    all_comments_df.to_excel(comments_file_path, index=False)
    logging.info("댓글 정보 저장 완료: %s", comments_file_path)

if __name__ == "__main__":
    driver = initialize_driver()
    
    driver.get("https://www.tiktok.com/login")
    logging.info("브라우저가 열렸습니다. 직접 로그인한 후, 로그인 완료 시 엔터키를 누르세요.")
    input("로그인 완료 후 엔터키를 눌러주세요.")
    
    driver.get("https://www.tiktok.com/foryou")
    time.sleep(3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    user_profiles = [
        "https://www.tiktok.com/@__ralral__",
        "https://www.tiktok.com/@horseking88",
        "https://www.tiktok.com/@joodoonge",
        "https://www.tiktok.com/@calmdownman_plus1",
        "https://www.tiktok.com/@trytoeat222",
        "https://www.tiktok.com/@under_world_b1",
        "https://www.tiktok.com/@heebab.tiktok",
        "https://www.tiktok.com/@jihan_ee",
        "https://www.tiktok.com/@yellow_aquarium",
        "https://www.tiktok.com/@park_esther",
        "https://www.tiktok.com/@simcook_",
        "https://www.tiktok.com/@minsco_official",
        "https://www.tiktok.com/@zaddnognoaw",
        "https://www.tiktok.com/@yooxicman.fi",
        "https://www.tiktok.com/@dudely_08",
        "https://www.tiktok.com/@fundamental_kr",
        "https://www.tiktok.com/@lesuho_",
        "https://www.tiktok.com/@reneirubinr2",
        "https://www.tiktok.com/@yonamism",
        "https://www.tiktok.com/@mejoocats",
        "https://www.tiktok.com/@jeon_unni",
        "https://www.tiktok.com/@bboyongnyong",
        "https://www.tiktok.com/@salim_nam_official",
        "https://www.tiktok.com/@mincheol1022",
        "https://www.tiktok.com/@enjoycouple",
        "https://www.tiktok.com/@hxxax__",
        "https://www.tiktok.com/@g_movie_official",
        "https://www.tiktok.com/@daldalsisters",
        "https://www.tiktok.com/@ipduck_official",
        "https://www.tiktok.com/@baby_pig_rabbit"
    ]
    
    for profile_url in user_profiles:
        influencer_name = profile_url.rstrip('/').split('@')[-1]
        stats_file_path = f"/Users/syb/Desktop/tiktok_crawling/tiktok_stats1/tiktok_stats_{influencer_name}_{timestamp}.xlsx"
        comments_file_path = f"/Users/syb/Desktop/tiktok_crawling/tiktok_comments1/tiktok_comments_{influencer_name}_{timestamp}.xlsx"
        
        extract_latest_videos_from_channel(
            driver=driver,
            user_profile_url=profile_url,
            stats_file_path=stats_file_path,
            comments_file_path=comments_file_path,
            max_results=30
        )
    
    driver.quit()
