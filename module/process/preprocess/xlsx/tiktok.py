import os
import pandas as pd

def merge_score_fss_columns(source_dir, target_dir):
    target_files = set(os.listdir(target_dir))  # 🔁 target_dir 캐싱

    for filename in os.listdir(source_dir):
        if filename.endswith(".xlsx") and filename in target_files:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            try:
                df_source = pd.read_excel(source_path)
                df_target = pd.read_excel(target_path)

                # 필요한 컬럼이 있는지 확인
                required_cols = ["score", "FSS"]
                if not all(col in df_source.columns for col in required_cols):
                    print(f"⚠️ 필수 컬럼 없음 → 건너뜀: {filename}")
                    continue

                # 행 수 체크
                if len(df_source) != len(df_target):
                    print(f"⚠️ 행 수 불일치 → 건너뜀: {filename}")
                    continue

                # 열 추가
                for col in required_cols:
                    df_target[col] = df_source[col].values

                # 저장
                df_target.to_excel(target_path, index=False)
                print(f"✅ 병합 완료: {filename}")

            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

# ▶ 실행 예시
if __name__ == "__main__":
    source_dir = "C:/Users/bandl/OneDrive/바탕 화면/docs/tiktok_data"
    target_dir = "C:/Users/bandl/OneDrive/바탕 화면/tiktok_data"
    merge_score_fss_columns(source_dir, target_dir)
