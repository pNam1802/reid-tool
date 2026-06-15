# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
CROPS_DIR    = "crops"        # thư mục output từ bước 1
MONTAGE_DIR  = "montages"     # thư mục montage từ bước 2 (để điền cột montage)
OUTPUT_CSV   = "labels.csv"   # file CSV kết quả
THRESH       = 0.55           # ngưỡng cosine similarity cho cặp khác camera
WEIGHTS_PATH = ""             # đường dẫn weights OSNet tùy chỉnh (để trống = dùng pretrained)
BATCH_SIZE   = 32             # số ảnh xử lý mỗi lần (giảm nếu hết RAM/VRAM)
MIN_IMAGES   = 3              # bỏ qua tracklet có ít hơn N ảnh
# =================================================================

import csv
import sys
from pathlib import Path

import numpy as np


# ── Union-Find với path compression + union by rank ───────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def compute_tracklet_embedding(extractor, image_paths: list, batch_size: int):
    """
    Tính embedding đại diện cho 1 tracklet.
    = trung bình của các vector đặc trưng đã L2-normalize theo từng ảnh.
    """
    import torch

    feats_list = []
    str_paths = [str(p) for p in image_paths]

    for i in range(0, len(str_paths), batch_size):
        batch = str_paths[i:i + batch_size]
        try:
            with torch.no_grad():
                feats = extractor(batch)          # (B, D) torch.Tensor
            for feat in feats.cpu().numpy():
                feats_list.append(l2_norm(feat))  # L2-normalize từng ảnh
        except Exception as e:
            print(f"  [CẢNH BÁO] Lỗi batch {i}: {e}")

    if not feats_list:
        return None

    return np.mean(feats_list, axis=0)            # trung bình các vector đã normalize


def main():
    import torch

    try:
        import torchreid
    except Exception as _e:
        import traceback
        # ASCII-only messages to avoid UnicodeEncodeError on Windows CP1252 terminals
        print("[ERROR] Cannot import torchreid. Real error:")
        traceback.print_exc()
        print()
        print("Fix: install torchreid into the ACTIVE venv (not base conda):")
        print("  pip install torchreid")
        print()
        print("If already installed but still failing, check with:")
        print("  python -c \"import torchreid; print(torchreid.__file__)\"")
        sys.exit(1)

    crops_dir   = Path(CROPS_DIR)
    montage_dir = Path(MONTAGE_DIR)

    if not crops_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục crops: {crops_dir}")
        sys.exit(1)

    # --- Thu thập tracklet hợp lệ ---
    tracklet_dirs = sorted(crops_dir.glob("cam*/track*"))
    tracklets = []   # list of (cam_id, track_id, image_paths)

    for td in tracklet_dirs:
        images = sorted(td.glob("*.jpg"))
        if len(images) < MIN_IMAGES:
            continue
        try:
            cam_id   = int(td.parent.name.replace("cam", ""))
            track_id = int(td.name.replace("track", ""))
        except ValueError:
            continue
        tracklets.append((cam_id, track_id, images))

    if not tracklets:
        print("[LỖI] Không tìm thấy tracklet hợp lệ nào.")
        sys.exit(1)

    print(f"Tracklet hợp lệ (≥{MIN_IMAGES} ảnh): {len(tracklets)}")

    # --- Tải OSNet extractor ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")
    print("Đang tải model OSNet-x1.0 ...")

    extractor_kwargs: dict = {"model_name": "osnet_x1_0", "device": device}
    if WEIGHTS_PATH:
        extractor_kwargs["model_path"] = WEIGHTS_PATH

    extractor = torchreid.utils.FeatureExtractor(**extractor_kwargs)

    # --- Tính embedding cho từng tracklet ---
    print("Đang tính embedding tracklet ...")
    embeddings = []
    valid_tracklets = []

    for idx, (cam_id, track_id, images) in enumerate(tracklets):
        emb = compute_tracklet_embedding(extractor, images, BATCH_SIZE)
        if emb is None:
            print(f"  [BỎ QUA] cam{cam_id}/track{track_id:04d} — không tính được embedding")
            continue
        embeddings.append(emb)
        valid_tracklets.append((cam_id, track_id, images))
        if (idx + 1) % 20 == 0 or (idx + 1) == len(tracklets):
            print(f"  {idx + 1}/{len(tracklets)} tracklet xử lý xong")

    if not embeddings:
        print("[LỖI] Không có embedding nào được tính.")
        sys.exit(1)

    emb_matrix = np.array(embeddings)                               # (N, D)

    # --- Tính cosine similarity hiệu quả bằng phép nhân ma trận ---
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.maximum(norms, 1e-8)              # (N, D) đã normalize
    sim_matrix = emb_normed @ emb_normed.T                          # (N, N) cosine sim

    # --- Union-Find: gộp cặp vượt ngưỡng ---
    # Cùng camera: ngưỡng cao hơn +0.15 vì ByteTrack đã tách tốt trong 1 cam
    same_cam_thresh  = THRESH + 0.15
    cross_cam_thresh = THRESH
    N = len(valid_tracklets)
    uf = UnionFind(N)

    for i in range(N):
        for j in range(i + 1, N):
            cam_i = valid_tracklets[i][0]
            cam_j = valid_tracklets[j][0]
            thresh = same_cam_thresh if cam_i == cam_j else cross_cam_thresh
            if sim_matrix[i, j] >= thresh:
                uf.union(i, j)

    # Đánh số lại auto_group liên tiếp từ 0 theo thứ tự root
    roots = [uf.find(i) for i in range(N)]
    unique_roots = sorted(set(roots))
    root_to_group = {r: g for g, r in enumerate(unique_roots)}
    auto_groups = [root_to_group[r] for r in roots]

    # --- Tạo rows CSV ---
    rows = []
    for idx, (cam_id, track_id, images) in enumerate(valid_tracklets):
        montage_path = montage_dir / f"cam{cam_id}_track{track_id:04d}.jpg"
        montage_str  = str(montage_path).replace("\\", "/") if montage_path.exists() else ""

        rows.append({
            "cam":        cam_id,
            "track_id":   track_id,
            "num_images": len(images),
            "montage":    montage_str,
            "auto_group": auto_groups[idx],
            "global_pid": auto_groups[idx],   # người dùng sẽ chỉnh cột này
        })

    # Sắp xếp theo auto_group để các tracklet cùng cụm nằm kề nhau
    rows.sort(key=lambda r: (r["auto_group"], r["cam"], r["track_id"]))

    # --- Xuất labels.csv ---
    csv_path = Path(OUTPUT_CSV)
    fieldnames = ["cam", "track_id", "num_images", "montage", "auto_group", "global_pid"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== THỐNG KÊ ===")
    print(f"Tracklet đã xử lý    : {N}")
    print(f"Số nhóm auto-suggest : {len(unique_roots)}")
    print(f"File CSV             : {csv_path}")
    print("""
=== HƯỚNG DẪN GÁN NHÃN ===
1. Mở file labels.csv bằng Excel / LibreOffice Calc / Google Sheets.
2. Xem cột 'montage' để kiểm tra từng tracklet bằng mắt
   (mở ảnh lưới trong montage/ để xem hình thực tế).
3. Chỉnh sửa cột 'global_pid':
   - Các tracklet cùng người → đặt cùng số nguyên (0, 1, 2, ...)
   - Tracklet nhiễu / không rõ / không muốn dùng → đặt -1
4. Cột 'auto_group' chỉ để tham khảo — KHÔNG cần sửa.
5. Lưu file rồi chạy: python build_dataset.py
""")


if __name__ == "__main__":
    main()
