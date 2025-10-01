import os, cv2, numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 실행 경로
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")  # images/ 밑에 모든 하위 폴더 포함
OUTPUT_LABEL_FOLDER = os.path.join(BASE_DIR, "labels")
OUTPUT_ANN_FOLDER = os.path.join(BASE_DIR, "annotated")
CROP_FOLDER = os.path.join(BASE_DIR, "crops")

CLASSES = ["교복", "체육복", "생활복"]  # index 0,1,2
CONF_THRESH = 0.25

# --- YOLO 모델 로드 ---
model = YOLO("yolov8n.pt")
print("[INFO] YOLO 로드 완료")

# --- 사람 검출 함수 ---
def detect_persons(img):
    boxes = []
    results = model.predict(source=img, conf=CONF_THRESH, verbose=False)
    if len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for b, c in zip(xyxy, classes):
                if int(c) == 0:  # COCO person
                    x1, y1, x2, y2 = map(int, b[:4])
                    boxes.append((x1, y1, x2, y2))
    return boxes

# --- 메인 ---
def main():
    # ✅ 기존 결과 유지, 없으면 폴더 생성
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_ANN_FOLDER, exist_ok=True)
    os.makedirs(CROP_FOLDER, exist_ok=True)

    # images 폴더 확인
    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] images/ 폴더가 존재하지 않습니다: {IMAGE_FOLDER}")
        return

    # ✅ images/ 및 모든 하위 폴더 검색
    img_files = []
    for root, _, files in os.walk(IMAGE_FOLDER):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                img_files.append(os.path.join(root, f))
    img_files.sort()

    if len(img_files) == 0:
        print(f"[ERROR] {IMAGE_FOLDER}/ 폴더에 이미지가 없습니다.")
        return

    for img_path in tqdm(img_files, desc="[INFO] 이미지 처리 중"):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        boxes = detect_persons(img)
        label_lines = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 얼굴 제외 (상단 15%), 신발 제외 (하단 10%)
            y1_new = y1 + int(0.15*(y2-y1))
            y2_new = y2 - int(0.10*(y2-y1))

            # 경계 체크
            y1_new = max(0, y1_new)
            y2_new = min(h_img-1, y2_new)
            if y2_new <= y1_new:
                continue

            # 크롭 저장
            crop = img[y1_new:y2_new, x1:x2]
            crop_name = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            crop_path = os.path.join(CROP_FOLDER, crop_name)
            cv2.imwrite(crop_path, crop)

            # YOLO 포맷 라벨링 (목~발목)
            x_c = ((x1+x2)/2)/w_img
            y_c = ((y1_new+y2_new)/2)/h_img
            bw = (x2-x1)/w_img
            bh = (y2_new-y1_new)/h_img

            # 클래스 0 (기본: 교복, 필요시 mapping 수정)
            label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

            # annotated 이미지
            ann_path = os.path.join(OUTPUT_ANN_FOLDER, img_name)
            img_ann = img.copy()
            cv2.rectangle(img_ann, (x1, y1_new), (x2, y2_new), (0,255,0), 2)
            cv2.putText(img_ann, f"목~발목", (x1, max(15,y1_new-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.imwrite(ann_path, img_ann)

        # ✅ 기존 라벨 있으면 덮어쓰기, 없으면 새로 생성
        txt_path = os.path.join(OUTPUT_LABEL_FOLDER, os.path.splitext(img_name)[0]+".txt")
        with open(txt_path, "w") as f:
            f.writelines(label_lines)

    print("[DONE] 목~발목 라벨링 완료 (누적 모드)")

if __name__=="__main__":
    main()
