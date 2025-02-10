from selenium import webdriver
import time
from openpyxl import Workbook
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import warnings
import re

id = "QtgO9wUgj9E"

# 전처리 함수
def preprocess(driver):
    print("Preprocessing Start")
    # Scroll down to load comments
    time.sleep(1.5)
    driver.execute_script("window.scrollTo(0, 800)")
    time.sleep(3)

    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(1.5)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    time.sleep(1.5)

    # Close any pop-ups if present
    try:
        driver.find_element(By.CSS_SELECTOR, "#dismiss-button > a").click()
    except:
        pass

    # Click on 'more replies' buttons
    buttons = driver.find_elements(By.CSS_SELECTOR, "ytd-button-renderer#more-replies > yt-button-shape > button")
    print(buttons[1])
    for i in range(len(buttons)):
        buttons[i].send_keys(Keys.ENTER)
    print("Preprocessing complete")

# 댓글 수집 함수
def collect_comments(driver):
    print("Collecting start")

    # Get page source and parse with BeautifulSoup
    html_source = driver.page_source
    soup = BeautifulSoup(html_source, 'html.parser')
    
    # CSS Selectors for extracting data
    youtube_name = extract_str(str(soup.select_one("div#header-text > h3")))
    youtube_follower = extract_str(str(soup.select_one("div#header-text > div#subtitle")))
    print(youtube_name)
    print(youtube_follower)


    video_info = str(soup.select_one("div#watch7-content"))
    #print(video_info)
    
    id_list = soup.select("div#header-author > h3 > #author-text > span")
    comment_list = soup.select("div#content > yt-attributed-string#content-text > span")
    like_list = soup.select("span#vote-count-middle")
    
    print(extract_str(id_list[0]))
    print(remove_tag(comment_list[0]))
    print(extract_str(like_list[0]))
    
    id_final = []
    comment_final = []
    like_final = []

    # Loop through each comment and extract data
    for i in range(len(comment_list)):
        temp_id = extract_str(id_list[i]).strip() if i < len(id_list) else ""  # strip() 메서드를 사용하여 공백 제거
        id_final.append(temp_id)  # 댓글 작성자

        temp_comment = remove_tag(comment_list[i]).strip() if i < len(comment_list) else ""  # strip() 메서드를 사용하여 공백 제거
        comment_final.append(temp_comment)  # 댓글 내용

        temp_like = extract_str(like_list[i]).strip() if i < len(id_list) else ""  # strip() 메서드를 사용하여 공백 제거
        like_final.append(temp_like)  # 댓글 작성자

    # Create DataFrame
    pd_data = {"아이디": id_final, "댓글 내용": comment_final, "좋아요":like_final}
    youtube_pd = pd.DataFrame(pd_data)

    # Save DataFrame to Excel
    youtube_pd.to_excel('../xlsx/crawling_manu.xlsx', index=False)

    print("Comment extraction complete")


# 메인 실행 함수
def main():
    # Set up Chrome options and service
    chrome_options = Options()
    chrome_service = Service(executable_path="./chromedriver.exe")

    # Initialize driver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    driver.get("https://www.youtube.com/watch?v=" + id)
    driver.implicitly_wait(3)

    # 전처리 실행
    preprocess(driver)

    # 댓글 수집 실행
    collect_comments(driver)

    # Quit the driver
    driver.quit()

def extract_str(text):
    # 정규 표현식 패턴 정의: 임의의 태그 안의 숫자 추출
    pattern = r'<[^>]*>(.*?)<\/[^>]*>'
    # 정규 표현식 검색
    match = re.search(pattern, my_str(text))
    if match:
        return match.group(1)  # 첫 번째 캡처 그룹(숫자)을 반환
    return None

def remove_tag(content):
    while True:
        # 모든 태그를 제거하고 내용(content)만 남기기
        cleaned_content = re.sub(r'<[^>]*>', '', my_str(content))
        # 변경된 내용이 이전 내용과 같다면 반복 종료
        if cleaned_content == content:
            break
        content = cleaned_content  # 변경된 내용을 다시 처리할 대상으로 설정
    return cleaned_content


def my_str(value):
    # 개행 문자 제거
    value = str(value)
    value = value.replace("\n", "").replace("\r", "")
    return value

if __name__ == "__main__":
    main()
