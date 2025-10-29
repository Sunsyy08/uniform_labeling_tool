import os
import re

# 삭제할 폴더 경로 (여기선 crops 폴더)
FOLDER = "crops"

# 정규식: 파일 이름이 "_숫자"로 끝나는 경우 (예: training1_0.jpg)
pattern = re.compile(r"_\d+\.(jpg|jpeg|png)$", re.IGNORECASE)

deleted_count = 0
for f in os.listdir(FOLDER):
    file_path = os.path.join(FOLDER, f)
    if os.path.isfile(file_path) and pattern.search(f):
        os.remove(file_path)
        deleted_count += 1

print(f"[DONE] '_숫자' 형식 파일 {deleted_count}개 삭제 완료 ✅")
