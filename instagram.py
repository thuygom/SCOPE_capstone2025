from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

import pandas as pd
import time

# 웹드라이버 초기화 및 로그인
driver = webdriver.Chrome(service=Service())
driver.get("https://www.instagram.com/accounts/login/")
time.sleep(2)

# 로그인 정보 입력
id = ""
pw = ""
inputs = driver.find_elements(By.TAG_NAME, "input")
inputs[0].send_keys(id)
inputs[1].send_keys(pw)
inputs[1].send_keys("\n")
time.sleep(5)

# "Not now" 버튼 클릭
try:
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[text()='Not now' or text()='나중에 하기']"))
    ).click()
except Exception as e:
    print("알림 창을 찾지 못했습니다:", e)

# 계정 페이지로 이동
profile_url = "https://www.instagram.com/c____chae/"
driver.get(profile_url)
time.sleep(5)

# 최신 게시물 5개의 URL 가져오기
post_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/reel/"]')[:5]  
post_urls = [post.get_attribute("href") for post in post_links]

# data lists
# 결과를 저장할 리스트
all_comments_data = []
id_f=[]
rp_f=[]



# 각 게시물로 이동해 댓글 크롤링
for post_url in post_urls:
    driver.get(post_url)
    time.sleep(3)

    comments_section = WebDriverWait(driver, 10).until(
    EC.visibility_of_element_located((By.CSS_SELECTOR, "ul._a9z6._a9za"))
    )
    
    comment_more_btn = "button[aria-label='Load more comments']"
    while True:
        try:
            more_btn = driver.find_element(By.CSS_SELECTOR, comment_more_btn)
            more_btn.click()
            time.sleep(2)  # 페이지 로딩을 기다리기 위해 잠시 대기
            print("더보기 버튼 클릭")
        except:
            print("더보기 버튼이 더 이상 존재하지 않습니다.")
            break

    # 아이디와 댓글 내용 추출
    ids = driver.find_elements(By.CSS_SELECTOR, "ul._a9z6._a9za li a[role='link']")
    replies = driver.find_elements(By.CSS_SELECTOR, "ul._a9z6._a9za li span[dir='auto']._aaco")

    # zip으로 아이디와 댓글 매핑
    for id_element, reply_element in zip(ids, replies):
        id_f.append(id_element.text.strip())
        rp_f.append(reply_element.text.strip())

    # 각 게시물에 대해 추출된 데이터를 리스트에 저장
    for i in range(len(id_f)):
        all_comments_data.append({
            "게시물 URL": post_url,
            "작성자": id_f[i],
            "댓글": rp_f[i]
        })

# DataFrame 생성
df = pd.DataFrame(all_comments_data)

# 엑셀 파일로 저장
df.to_excel("instagram_comments.xlsx", index=False)
print("댓글 데이터가 엑셀 파일로 저장되었습니다.")

driver.quit()
