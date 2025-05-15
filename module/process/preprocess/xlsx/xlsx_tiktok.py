import os
import pandas as pd
import re

COLUMNS_FINAL = ["video_url", "comment", "date", "감정", "주제", "군집"]

def reorder_tiktok_excel_columns_with_clean_url(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            print(f"\n🔍 처리 중: {filename}")
            try:
                df = pd.read_excel(file_path)

                # ✅ video_url → 숫자 video_id만 남기기 (video_url 열 자체를 덮어씀)
                if "video_url" in df.columns:
                    def extract_video_id(url):
                        if isinstance(url, str):
                            match = re.search(r"/video/(\d+)", url.strip())
                            return match.group(1) if match else url
                        return url

                    df["video_url"] = df["video_url"].apply(extract_video_id)
                    print("✅ video_url에서 숫자만 추출 완료")

                # ▶ 필요한 컬럼만 추출 & 순서 정렬
                ordered_cols = [col for col in COLUMNS_FINAL if col in df.columns]
                df = df[ordered_cols]

                # ▶ 덮어쓰기 저장
                df.to_excel(file_path, index=False)
                print(f"✅ 정리 및 저장 완료: {file_path}")

            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

# ▶ 실행
if __name__ == "__main__":
    target_dir = "C:/Users/bandl/OneDrive/바탕 화면/tiktok_data"
    reorder_tiktok_excel_columns_with_clean_url(target_dir)
