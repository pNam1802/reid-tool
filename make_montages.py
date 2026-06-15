# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
CROPS_DIR   = "crops"      # thư mục output từ bước 1 (extract_crops.py)
OUTPUT_DIR  = "montages"   # thư mục lưu ảnh lưới
MAX_IMAGES  = 10           # số ảnh đại diện tối đa mỗi tracklet trong lưới
THUMB_W     = 128          # chiều rộng mỗi thumbnail (px)
THUMB_H     = 256          # chiều cao mỗi thumbnail (px)
COLS        = 5            # số cột trong lưới
MIN_IMAGES  = 3            # bỏ qua tracklet có ít hơn N ảnh
# =================================================================

import sys
from pathlib import Path

# Ép stdout UTF-8 để in tiếng Việt không crash trên terminal cp1252 (Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np


def select_evenly(paths: list, n: int) -> list:
    """Chọn n ảnh phân bố đều từ danh sách để đại diện cho tracklet."""
    if len(paths) <= n:
        return list(paths)
    indices = [int(round(i * (len(paths) - 1) / (n - 1))) for i in range(n)]
    # Khử trùng lặp chỉ số nhưng giữ thứ tự
    seen = set()
    return [paths[idx] for idx in indices if not (idx in seen or seen.add(idx))]


def make_montage(image_paths: list, thumb_w: int, thumb_h: int, cols: int) -> np.ndarray:
    """Ghép ảnh thành lưới; ô không đủ ảnh để đen."""
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue  # bỏ qua ảnh không đọc được
        img = cv2.resize(img, (thumb_w, thumb_h))
        r, c = divmod(idx, cols)
        canvas[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = img

    return canvas


def main():
    crops_dir = Path(CROPS_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not crops_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục crops: {crops_dir}")
        sys.exit(1)

    # Tìm tất cả tracklet theo cấu trúc crops/cam{C}/track{TTTT}/
    tracklet_dirs = sorted(crops_dir.glob("cam*/track*"))
    if not tracklet_dirs:
        print(f"[CẢNH BÁO] Không tìm thấy tracklet nào trong {crops_dir}")
        sys.exit(0)

    print(f"Tìm thấy {len(tracklet_dirs)} tracklet trong {crops_dir}")

    saved = 0
    skipped = 0

    for track_dir in tracklet_dirs:
        cam_part   = track_dir.parent.name   # "cam1"
        track_part = track_dir.name          # "track0001"

        images = sorted(track_dir.glob("*.jpg"))

        # Bỏ qua tracklet không đủ ảnh
        if len(images) < MIN_IMAGES:
            skipped += 1
            continue

        # Chọn ảnh đại diện phân bố đều
        selected = select_evenly(images, MAX_IMAGES)

        # Tạo và lưu ảnh lưới
        montage = make_montage(selected, THUMB_W, THUMB_H, COLS)
        out_name = f"{cam_part}_{track_part}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), montage)
        saved += 1

    print(f"\n=== THỐNG KÊ ===")
    print(f"Tổng tracklet         : {len(tracklet_dirs)}")
    print(f"Montage đã tạo        : {saved}")
    print(f"Bỏ qua (< {MIN_IMAGES} ảnh)  : {skipped}")
    print(f"Thư mục output        : {out_dir}")


if __name__ == "__main__":
    main()
