import os, cv2, numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")  # 모든 하위 폴더 포함
OUTPUT_LABEL_FOLDER = os.path.join(BASE_DIR, "labels")
OUTPUT_ANN_FOLDER = os.path.join(BASE_DIR, "annotated")
CROP_FOLDER = os.path.join(BASE_DIR, "crops")

CLASSES = ["교복", "체육복", "생활복"]  # index 0,1,2
CONF_THRESH = 0.25

# --- YOLO 모델 로드 ---
model = YOLO("yolov8n.pt")
print("[INFO] YOLO 로드 완료")

def detect_persons(img):
    boxes = []
    results = model.predict(source=img, conf=CONF_THRESH, verbose=False)
    if len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for b, c in zip(xyxy, classes):
                if int(c) == 0:  # COCO 'person'
                    x1, y1, x2, y2 = map(int, b[:4])
                    boxes.append((x1, y1, x2, y2))
    return boxes


def main():
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_ANN_FOLDER, exist_ok=True)
    os.makedirs(CROP_FOLDER, exist_ok=True)

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
        label_path = os.path.join(OUTPUT_LABEL_FOLDER, os.path.splitext(img_name)[0] + ".txt")

        # ✅ 이미 라벨이 있으면 건너뜀 (기존 유지)
        if os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        boxes = detect_persons(img)
        label_lines = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 얼굴 제외 (상단 15%), 신발 제외 (하단 10%)
            y1_new = y1 + int(0.15 * (y2 - y1))
            y2_new = y2 - int(0.10 * (y2 - y1))
            y1_new = max(0, y1_new)
            y2_new = min(h_img - 1, y2_new)
            if y2_new <= y1_new:
                continue

            crop = img[y1_new:y2_new, x1:x2]
            crop_name = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            crop_path = os.path.join(CROP_FOLDER, crop_name)
            cv2.imwrite(crop_path, crop)

            # 기본 교복(0)으로 라벨링 (필요시 폴더 이름 기준으로 클래스 매핑 가능)
            label_lines.append(f"0 {(x1+x2)/(2*w_img):.6f} {(y1_new+y2_new)/(2*h_img):.6f} {(x2-x1)/w_img:.6f} {(y2_new-y1_new)/h_img:.6f}\n")

            # Annotated 저장
            img_ann = img.copy()
            cv2.rectangle(img_ann, (x1, y1_new), (x2, y2_new), (0,255,0), 2)
            cv2.putText(img_ann, "목~발목", (x1, max(15,y1_new-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            ann_path = os.path.join(OUTPUT_ANN_FOLDER, img_name)
            cv2.imwrite(ann_path, img_ann)

        # 새로 생성된 라벨만 저장
        if label_lines:
            with open(label_path, "w") as f:
                f.writelines(label_lines)

    print("[DONE] 새 이미지만 라벨링 완료 (기존 라벨 유지)")


if __name__ == "__main__":
    main()