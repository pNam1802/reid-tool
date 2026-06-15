# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
VIDEO_DIR    = "video"        # nơi bỏ 3 file mp4 (1 file/camera)
CROPS_DIR    = "crops"        # ảnh crop theo track (TẠM)
MONTAGE_DIR  = "montages"     # ảnh lưới để gán nhãn (TẠM)
LABELS_CSV   = "labels.csv"   # nhãn đợt hiện tại (TẠM)
DATASET_DIR  = "myreid"       # DATASET THẬT, Market-1501, tích lũy vĩnh viễn ★

# Tham số extract (ghi vào extract_crops.py khi chạy)
SAVE_EVERY    = 8
MIN_HEIGHT    = 80
MAX_PER_TRACK = 25
MIN_GAP       = 24
# Tham số montage
MONTAGE_MIN_IMAGES = 3
# Tỉ lệ identity (≥2 camera) đưa vào test khi gộp vào dataset
TEST_RATIO    = 0.3
# Người chỉ xuất hiện ở 1 camera (chống camera bias):
#   "distractor" = thả vào gallery làm người nhiễu (đánh giá sát thực tế)
#   "drop"       = bỏ hẳn, không đưa vào dataset
SINGLE_CAM_MODE = "distractor"
DISTRACTOR_BASE = 100000      # pid >= giá trị này là distractor, KHÔNG phải identity train/reuse

HOST         = "127.0.0.1"
PORT         = 8000
OPEN_BROWSER = True
# =================================================================
#
# Cách chạy:   python label_server.py
# Chỉ dùng thư viện chuẩn Python cho server (extract/montage gọi script sẵn có).
#
# LUỒNG:  bỏ video vào video/  →  web: Crop → Montage → Gán ID → Gộp vào dataset
#         Gộp xong: xóa sạch crops/ montages/ labels.csv + video đã xử lý,
#         chỉ giữ lại dataset/ lớn dần.
#

import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
UI_FILE = HERE / "label_ui.html"
PYTHON = sys.executable

# Trạng thái gán nhãn: -2 = chưa xem, -1 = bỏ qua, >=0 = đã gán PID
UNSEEN = -2
SKIP = -1

FIELDS = ["cam", "track_id", "num_images", "montage", "global_pid", "track_dir"]

# ── State trong bộ nhớ ────────────────────────────────────────────────────────
TRACKS: list[dict] = []
BY_DIR: dict[str, dict] = {}
_LOCK = threading.Lock()

# Tiến trình xử lý nền (extract + montage)
PROGRESS = {"running": False, "phase": "", "done": False, "error": ""}
LOG = deque(maxlen=40)


# ══════════════════════════════════════════════════════════════════════════════
# Ghi CONFIG block vào script con (không xâm phạm logic, giống pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════
def write_config(script: str, updates: dict) -> None:
    p = HERE / script
    content = p.read_text(encoding="utf-8")
    for key, value in updates.items():
        if isinstance(value, bool):
            pattern = rf'^({re.escape(key)}\s*=\s*)(?:True|False)'
            new_val = "True" if value else "False"
        elif isinstance(value, str):
            pattern = rf'^({re.escape(key)}\s*=\s*)"[^"]*"'
            new_val = f'"{value}"'
        else:
            pattern = rf'^({re.escape(key)}\s*=\s*)[\d.]+'
            new_val = str(value)
        content = re.sub(pattern, lambda m, v=new_val: m.group(1) + v,
                         content, flags=re.MULTILINE)
    p.write_text(content, encoding="utf-8")


def run_step(script: str) -> bool:
    """Chạy 1 script con, đẩy stdout vào LOG. Trả về True nếu thành công."""
    # Ép tiến trình con in UTF-8 (tránh UnicodeEncodeError do terminal cp1252
    # khi script in tiếng Việt) — sửa 1 lần cho mọi script con.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        [PYTHON, "-u", str(HERE / script)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(HERE),   # crops/ montages/ luôn nằm cạnh script
        env=env,
    )
    for line in proc.stdout:
        LOG.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        LOG.append(f"[LỖI] {script} kết thúc với mã {proc.returncode}")
    return proc.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# Quét video/ → danh sách camera
# ══════════════════════════════════════════════════════════════════════════════
def _mp4_files(folder: Path) -> list[Path]:
    """Liệt kê file .mp4 (không phân biệt hoa/thường, không trùng lặp)."""
    if not folder.exists():
        return []
    seen, out = set(), []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() == ".mp4":
            key = str(f).lower()
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


def scan_videos() -> list[dict]:
    p = Path(VIDEO_DIR)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return []
    return [{"video": str(f).replace("\\", "/"), "cam_id": i + 1, "name": f.name}
            for i, f in enumerate(_mp4_files(p))]


# ══════════════════════════════════════════════════════════════════════════════
# Quét crops + montages → labels.csv (mỗi track 1 dòng, pid mặc định = -2)
# ══════════════════════════════════════════════════════════════════════════════
def scan_to_labels() -> None:
    crops_dir   = Path(CROPS_DIR)
    montage_dir = Path(MONTAGE_DIR)
    csv_path    = Path(LABELS_CSV)

    existing: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row.get("track_dir", "")] = row

    rows: list[dict] = []
    if crops_dir.exists():
        for track_dir in sorted(crops_dir.glob("cam*/track*")):
            images = sorted(track_dir.glob("*.jpg"))
            if not images:
                continue
            m = re.search(r"cam(\d+)", track_dir.parent.name)
            if not m:
                continue
            cam_id   = int(m.group(1))
            track_id = int(track_dir.name.replace("track", ""))
            track_dir_rel = str(track_dir.relative_to(crops_dir)).replace("\\", "/")
            mp = montage_dir / f"{track_dir.parent.name}_{track_dir.name}.jpg"
            montage_rel = str(mp).replace("\\", "/") if mp.exists() else ""
            pid = int(existing[track_dir_rel]["global_pid"]) if track_dir_rel in existing else UNSEEN
            rows.append({
                "cam": cam_id, "track_id": track_id, "num_images": len(images),
                "montage": montage_rel, "global_pid": pid, "track_dir": track_dir_rel,
            })

    rows.sort(key=lambda r: (r["cam"], r["track_dir"]))
    _load_rows(rows)
    save_csv()


def _load_rows(rows: list[dict]) -> None:
    TRACKS.clear(); BY_DIR.clear()
    TRACKS.extend(rows)
    for r in rows:
        BY_DIR[r["track_dir"]] = r


def load_labels_from_disk() -> None:
    """Nạp labels.csv vào bộ nhớ (khi server khởi động giữa chừng)."""
    csv_path = Path(LABELS_CSV)
    if not csv_path.exists():
        _load_rows([])
        return
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "cam": int(row["cam"]), "track_id": int(row["track_id"]),
                "num_images": int(row["num_images"]), "montage": row.get("montage", ""),
                "global_pid": int(row["global_pid"]), "track_dir": row["track_dir"],
            })
    _load_rows(rows)


def save_csv() -> None:
    with _LOCK:
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(TRACKS)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset (Market-1501) — đọc identity đã có, gộp đợt mới, dọn tạm
# ══════════════════════════════════════════════════════════════════════════════
def ds_dirs():
    base = Path(DATASET_DIR)
    return (base / "bounding_box_train", base / "query",
            base / "bounding_box_test", base / "_identities")


def existing_identities() -> list[dict]:
    """Danh sách identity đã có trong dataset (để đối chiếu khi gán đợt mới)."""
    _, _, _, idd = ds_dirs()
    if not idd.exists():
        return []
    out = []
    for f in sorted(idd.glob("pid_*.jpg")):
        m = re.match(r"pid_(\d+)\.jpg", f.name)
        if m:
            out.append({"pid": int(m.group(1))})
    return out


def _all_dataset_pids() -> set[int]:
    train, query, test, idd = ds_dirs()
    pids = set()
    for d in (train, query, test):
        if d.exists():
            for f in d.glob("*.jpg"):
                mm = re.match(r"(\d+)_c", f.name)
                if mm:
                    pid = int(mm.group(1))
                    if pid < DISTRACTOR_BASE:          # bỏ qua distractor
                        pids.add(pid)
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            mm = re.match(r"pid_(\d+)\.jpg", f.name)
            if mm:
                pids.add(int(mm.group(1)))
    return pids


def _max_distractor_pid() -> int:
    """pid distractor lớn nhất đang có trong gallery (để đánh số tiếp)."""
    _, _, test, _ = ds_dirs()
    mx = DISTRACTOR_BASE - 1
    if test.exists():
        for f in test.glob("*.jpg"):
            mm = re.match(r"(\d+)_c", f.name)
            if mm:
                p = int(mm.group(1))
                if p >= DISTRACTOR_BASE:
                    mx = max(mx, p)
    return mx


def camera_coverage() -> dict:
    """Thống kê 'sức khỏe' dataset: mỗi identity xuất hiện ở mấy camera."""
    train, query, test, _ = ds_dirs()
    pid_cams: dict[int, set] = defaultdict(set)
    distractor_pids: set = set()
    distractor_imgs = 0
    for d in (train, query, test):
        if not d.exists():
            continue
        for f in d.glob("*.jpg"):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm:
                continue
            pid, cam = int(mm.group(1)), int(mm.group(2))
            if pid >= DISTRACTOR_BASE:
                distractor_pids.add(pid)
                distractor_imgs += 1
            else:
                pid_cams[pid].add(cam)
    by_cams = {1: 0, 2: 0, 3: 0, "4+": 0}
    for cams in pid_cams.values():
        n = len(cams)
        by_cams["4+" if n >= 4 else n] = by_cams.get("4+" if n >= 4 else n, 0) + 1
    return {
        "identities": len(pid_cams),
        "by_cams": by_cams,
        "ge2": sum(1 for c in pid_cams.values() if len(c) >= 2),
        "distractor_people": len(distractor_pids),
        "distractor_images": distractor_imgs,
    }


def next_pid() -> int:
    used = _all_dataset_pids()
    batch = {t["global_pid"] for t in TRACKS if t["global_pid"] >= 0}
    allp = used | batch
    return (max(allp) + 1) if allp else 0


def _identity_split(pid: int) -> str | None:
    """Xác định identity đã thuộc train hay test (None nếu là người mới)."""
    train, query, test, _ = ds_dirs()
    pref = f"{pid:04d}_c"
    for d in (test, query):
        if d.exists() and any(f.name.startswith(pref) for f in d.glob("*.jpg")):
            return "test"
    if train.exists() and any(f.name.startswith(pref) for f in train.glob("*.jpg")):
        return "train"
    return None


def _cams_with_query(pid: int) -> set[int]:
    _, query, _, _ = ds_dirs()
    cams = set()
    if query.exists():
        for f in query.glob(f"{pid:04d}_c*.jpg"):
            mm = re.search(r"_c(\d+)s", f.name)
            if mm:
                cams.add(int(mm.group(1)))
    return cams


def _market_copy(src: Path, dst_dir: Path, pid: int, cam: int, frame: int) -> None:
    k = 1
    while True:
        name = f"{pid:04d}_c{cam}s1_{frame:06d}_{k:02d}.jpg"
        dst = dst_dir / name
        if not dst.exists():
            break
        k += 1
    shutil.copy2(str(src), str(dst))


def _frame_of(img: Path) -> int:
    m = re.search(r"f(\d+)", img.stem)
    return int(m.group(1)) if m else 0


def commit_to_dataset() -> dict:
    """Gộp các track đã gán PID vào dataset/, rồi xóa sạch dữ liệu tạm."""
    random.seed()
    train, query, test, idd = ds_dirs()
    for d in (train, query, test, idd):
        d.mkdir(parents=True, exist_ok=True)

    crops = Path(CROPS_DIR)
    montages = Path(MONTAGE_DIR)

    # Gom track theo PID (chỉ lấy pid >= 0)
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for t in TRACKS:
        if t["global_pid"] >= 0:
            by_pid[t["global_pid"]].append(t)

    stats = {"new_ids": 0, "updated_ids": 0, "train": 0, "query": 0, "gallery": 0,
             "new_distractor": 0, "distractor_images": 0, "dropped": 0}
    next_distractor = _max_distractor_pid() + 1

    for pid, group in sorted(by_pid.items()):
        split = _identity_split(pid)
        is_new = split is None
        cams = {t["cam"] for t in group}

        # CÁCH B — người MỚI chỉ 1 camera → distractor (chống camera bias, lỗi #2/#7)
        if is_new and len(cams) == 1:
            if SINGLE_CAM_MODE == "drop":
                stats["dropped"] += 1
                continue
            dpid = next_distractor
            next_distractor += 1
            stats["new_distractor"] += 1
            for t in group:
                for img in sorted((crops / t["track_dir"]).glob("*.jpg")):
                    _market_copy(img, test, dpid, t["cam"], _frame_of(img))   # vào gallery
                    stats["distractor_images"] += 1
            continue   # không lưu avatar, không vào train, không phải identity

        if is_new:
            split = "test" if (len(cams) >= 2 and random.random() < TEST_RATIO) else "train"
            stats["new_ids"] += 1
            # Lưu ảnh đại diện = 1 ẢNH CROP ĐƠN (ảnh giữa của track đầu) cho dễ nhận mặt
            rep_imgs = sorted((crops / group[0]["track_dir"]).glob("*.jpg"))
            if rep_imgs:
                shutil.copy2(str(rep_imgs[len(rep_imgs) // 2]), str(idd / f"pid_{pid:04d}.jpg"))
        else:
            stats["updated_ids"] += 1

        if split == "train":
            for t in group:
                folder = crops / t["track_dir"]
                for img in sorted(folder.glob("*.jpg")):
                    _market_copy(img, train, pid, t["cam"], _frame_of(img))
                    stats["train"] += 1
        else:  # test
            qcams = _cams_with_query(pid)
            # Gom ảnh theo camera
            cam_imgs: dict[int, list[Path]] = defaultdict(list)
            for t in group:
                folder = crops / t["track_dir"]
                for img in sorted(folder.glob("*.jpg")):
                    cam_imgs[t["cam"]].append(img)
            for cam, imgs in cam_imgs.items():
                imgs.sort()
                start = 0
                if cam not in qcams and imgs:
                    _market_copy(imgs[0], query, pid, cam, _frame_of(imgs[0]))
                    stats["query"] += 1
                    start = 1
                for img in imgs[start:]:
                    _market_copy(img, test, pid, cam, _frame_of(img))
                    stats["gallery"] += 1

    # ── Dọn dữ liệu tạm ──
    if crops.exists():
        shutil.rmtree(str(crops))
    if montages.exists():
        shutil.rmtree(str(montages))
    if Path(LABELS_CSV).exists():
        Path(LABELS_CSV).unlink()
    # Xóa video đã xử lý
    for f in _mp4_files(Path(VIDEO_DIR)):
        f.unlink()

    _load_rows([])
    stats["coverage"] = camera_coverage()
    stats["dataset"] = dataset_stats()

    # ── In thống kê đợt này ra console ──
    cov = stats["coverage"]
    bc = cov["by_cams"]
    print("\n========== THỐNG KÊ ĐỢT VỪA GỘP ==========")
    print(f"Người mới (≥2 camera, làm identity) : {stats['new_ids']}")
    print(f"Người mới 1 camera → distractor     : {stats['new_distractor']} "
          f"({stats['distractor_images']} ảnh vào gallery)")
    if stats["dropped"]:
        print(f"Người mới 1 camera → bỏ             : {stats['dropped']}")
    print(f"Identity cũ được bổ sung            : {stats['updated_ids']}")
    print(f"Ảnh thêm: {stats['train']} train · {stats['query']} query · {stats['gallery']} gallery")
    print("---------- SỨC KHỎE DATASET ----------")
    print(f"Tổng identity (train+test) : {cov['identities']}")
    print(f"  ở 1 camera : {bc.get(1,0)}   ← càng nhiều càng dễ camera-bias")
    print(f"  ở 2 camera : {bc.get(2,0)}")
    print(f"  ở 3 camera : {bc.get(3,0)}")
    print(f"  ở ≥4 camera: {bc.get('4+',0)}")
    print(f"Identity ≥2 camera (phần 'khỏe' cho ReID): {cov['ge2']} / {cov['identities']}")
    print(f"Distractor trong gallery : {cov['distractor_people']} người, {cov['distractor_images']} ảnh")
    print("==========================================\n")
    return stats


def dataset_stats() -> dict:
    train, query, test, idd = ds_dirs()
    def n(d): return len(list(d.glob("*.jpg"))) if d.exists() else 0
    return {
        "identities": len(existing_identities()),
        "train": n(train), "query": n(query), "gallery": n(test),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trình duyệt dataset — liệt kê identity, xem/sửa/xóa ảnh trong myreid
# ══════════════════════════════════════════════════════════════════════════════
def _split_dir(split: str) -> Path | None:
    """Map tên split của UI ('train'/'query'/'gallery') → thư mục dataset."""
    train, query, test, _ = ds_dirs()
    return {"train": train, "query": query, "gallery": test}.get(split)


def dataset_identities() -> list[dict]:
    """Tất cả identity trong dataset kèm thống kê: số ảnh mỗi split, số camera."""
    train, query, test, idd = ds_dirs()
    splits = {"train": train, "query": query, "gallery": test}
    info: dict[int, dict] = {}
    for split, d in splits.items():
        if not d.exists():
            continue
        for f in d.glob("*.jpg"):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm:
                continue
            pid, cam = int(mm.group(1)), int(mm.group(2))
            rec = info.setdefault(pid, {"train": 0, "query": 0, "gallery": 0,
                                        "cams": set(), "thumb": None})
            rec[split] += 1
            rec["cams"].add(cam)
            if rec["thumb"] is None:
                rec["thumb"] = {"split": split, "name": f.name}

    avatars: set[int] = set()
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            mm = re.match(r"pid_(\d+)\.jpg", f.name)
            if mm:
                pid = int(mm.group(1))
                avatars.add(pid)
                info.setdefault(pid, {"train": 0, "query": 0, "gallery": 0,
                                      "cams": set(), "thumb": None})

    out = []
    for pid, rec in info.items():
        out.append({
            "pid": pid,
            "train": rec["train"], "query": rec["query"], "gallery": rec["gallery"],
            "total": rec["train"] + rec["query"] + rec["gallery"],
            "cams": sorted(rec["cams"]), "ncams": len(rec["cams"]),
            "has_avatar": pid in avatars, "thumb": rec["thumb"],
            "distractor": pid >= DISTRACTOR_BASE,
        })
    out.sort(key=lambda r: r["pid"])
    return out


def identity_images(pid: int) -> list[dict]:
    """Tất cả ảnh của 1 pid trong dataset, kèm split + cam (để hiện và chọn xóa)."""
    splits = {"train": _split_dir("train"), "query": _split_dir("query"),
              "gallery": _split_dir("gallery")}
    out = []
    for split, d in splits.items():
        if not d or not d.exists():
            continue
        for f in sorted(d.glob(f"{pid:04d}_c*.jpg")):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm or int(mm.group(1)) != pid:
                continue
            out.append({"split": split, "name": f.name, "cam": int(mm.group(2))})
    return out


def ds_delete_images(images: list[dict]) -> int:
    """Xóa các ảnh dataset đã chọn. Mỗi phần tử {split, name}. Trả về số ảnh đã xóa."""
    deleted = 0
    for item in images:
        d = _split_dir(item.get("split", ""))
        name = Path(item.get("name", "")).name
        if not d or not name.endswith(".jpg"):
            continue
        f = d / name
        if f.exists():
            f.unlink()
            deleted += 1
    return deleted


def ds_delete_identity(pid: int) -> int:
    """Xóa hẳn 1 identity: mọi ảnh train/query/gallery + ảnh đại diện."""
    _, _, _, idd = ds_dirs()
    deleted = 0
    for split in ("train", "query", "gallery"):
        d = _split_dir(split)
        if d and d.exists():
            for f in d.glob(f"{pid:04d}_c*.jpg"):
                mm = re.match(r"(\d+)_c", f.name)
                if mm and int(mm.group(1)) == pid:
                    f.unlink()
                    deleted += 1
    avatar = idd / f"pid_{pid:04d}.jpg"
    if avatar.exists():
        avatar.unlink()
    return deleted


def ds_set_avatar(pid: int, split: str, name: str) -> bool:
    """Đặt 1 ảnh trong dataset làm ảnh đại diện của identity."""
    d = _split_dir(split)
    name = Path(name).name
    _, _, _, idd = ds_dirs()
    if not d or not name.endswith(".jpg") or not (d / name).exists():
        return False
    idd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(d / name), str(idd / f"pid_{pid:04d}.jpg"))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Máy trạng thái
# ══════════════════════════════════════════════════════════════════════════════
def detect_state() -> str:
    if PROGRESS["running"]:
        return "processing"
    crops = Path(CROPS_DIR)
    has_crops = crops.exists() and any(crops.glob("cam*/track*"))
    has_labels = Path(LABELS_CSV).exists()

    if has_labels:
        load_labels_from_disk()
        if any(t["global_pid"] == UNSEEN for t in TRACKS):
            return "label"
        if any(t["global_pid"] >= 0 for t in TRACKS):
            return "commit"
        return "commit"  # tất cả bỏ qua → vẫn cho commit (sẽ không thêm gì)
    if has_crops:
        return "montage_pending"   # có crop nhưng chưa montage (server tắt giữa chừng)
    # Sạch → cho thêm video
    return "idle"


def label_stats() -> dict:
    total   = len(TRACKS)
    labeled = sum(1 for t in TRACKS if t["global_pid"] >= 0)
    skipped = sum(1 for t in TRACKS if t["global_pid"] == SKIP)
    unseen  = sum(1 for t in TRACKS if t["global_pid"] == UNSEEN)
    n_pids  = len({t["global_pid"] for t in TRACKS if t["global_pid"] >= 0})
    return {"total": total, "labeled": labeled, "skipped": skipped,
            "unseen": unseen, "pids": n_pids}


# ══════════════════════════════════════════════════════════════════════════════
# Chạy pipeline nền: extract (mỗi camera) → montage → scan_to_labels
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(cameras: list[dict]) -> None:
    PROGRESS.update(running=True, phase="extract", done=False, error="")
    LOG.clear()
    try:
        for cam in cameras:
            LOG.append(f"=== CROP cam {cam['cam_id']}: {cam['video']} ===")
            write_config("extract_crops.py", {
                "VIDEO_PATH": cam["video"], "CAM_ID": int(cam["cam_id"]),
                "OUTPUT_DIR": CROPS_DIR, "RESET_CAM": False,
                "SAVE_EVERY": SAVE_EVERY, "MIN_HEIGHT": MIN_HEIGHT,
                "MAX_PER_TRACK": MAX_PER_TRACK, "MIN_GAP": MIN_GAP,
            })
            if not run_step("extract_crops.py"):
                raise RuntimeError(f"Extract cam {cam['cam_id']} lỗi")

        PROGRESS["phase"] = "montage"
        LOG.append("=== MONTAGE ===")
        write_config("make_montages.py", {
            "CROPS_DIR": CROPS_DIR, "OUTPUT_DIR": MONTAGE_DIR,
            "MIN_IMAGES": MONTAGE_MIN_IMAGES,
        })
        if not run_step("make_montages.py"):
            raise RuntimeError("Montage lỗi")

        LOG.append("=== TẠO DANH SÁCH GÁN NHÃN ===")
        scan_to_labels()
        LOG.append(f"Xong! {len(TRACKS)} track sẵn sàng để gán nhãn.")
        PROGRESS.update(phase="done", done=True)
    except Exception as e:
        PROGRESS["error"] = str(e)
        LOG.append(f"[LỖI] {e}")
    finally:
        PROGRESS["running"] = False


# ══════════════════════════════════════════════════════════════════════════════
# HTTP handler
# ══════════════════════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data: bytes, ctype: str, code=200, no_cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if no_cache:   # buộc trình duyệt tải lại HTML mới, tránh kẹt bản cũ trong cache
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # --- GET ---
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)

        if path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return

        if path == "/":
            if not UI_FILE.exists():
                self._json({"error": f"Thiếu {UI_FILE.name}"}, 500); return
            self._bytes(UI_FILE.read_bytes(), "text/html; charset=utf-8", no_cache=True); return

        if path == "/api/state":
            self._state_payload(); return

        if path == "/api/progress":
            self._json({**PROGRESS, "log": list(LOG)}); return

        if path == "/montage":
            td = qs.get("track_dir", [""])[0]
            rec = BY_DIR.get(td)
            if rec and rec["montage"] and Path(rec["montage"]).exists():
                self._bytes(Path(rec["montage"]).read_bytes(), "image/jpeg"); return
            self._json({"error": "no montage"}, 404); return

        if path == "/crop":
            # Ảnh dự phòng: track chưa có montage (ít ảnh) vẫn xem được 1 crop
            td = qs.get("track_dir", [""])[0]
            folder = Path(CROPS_DIR) / td
            imgs = sorted(folder.glob("*.jpg")) if folder.exists() else []
            if imgs:
                self._bytes(imgs[len(imgs) // 2].read_bytes(), "image/jpeg"); return
            self._json({"error": "no crop"}, 404); return

        if path == "/track_images":
            # Danh sách tên ảnh của 1 track (để hiện từng ảnh + cho xóa lẻ)
            td = qs.get("track_dir", [""])[0]
            folder = Path(CROPS_DIR) / td
            names = [f.name for f in sorted(folder.glob("*.jpg"))] if folder.exists() else []
            self._json({"images": names}); return

        if path == "/img":
            # Phục vụ 1 ảnh crop cụ thể (name chỉ lấy phần tên file, chống path traversal)
            td = qs.get("track_dir", [""])[0]
            name = Path(qs.get("name", [""])[0]).name
            f = Path(CROPS_DIR) / td / name
            if name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        if path == "/identity":
            pid = qs.get("pid", [""])[0]
            _, _, _, idd = ds_dirs()
            f = idd / f"pid_{int(pid):04d}.jpg" if pid.isdigit() else None
            if f and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no identity"}, 404); return

        if path == "/api/dataset":
            # Danh sách toàn bộ identity trong myreid + thống kê tổng
            self._json({"identities": dataset_identities(), "stats": dataset_stats()}); return

        if path == "/api/identity_images":
            pid = qs.get("pid", [""])[0]
            if not pid.isdigit():
                self._json({"error": "bad pid"}, 400); return
            self._json({"pid": int(pid), "images": identity_images(int(pid))}); return

        if path == "/ds_img":
            # Phục vụ 1 ảnh dataset cụ thể (chống path traversal: chỉ lấy basename)
            split = qs.get("split", [""])[0]
            name = Path(qs.get("name", [""])[0]).name
            d = _split_dir(split)
            f = (d / name) if d else None
            if f and name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        self._json({"error": "not found"}, 404)

    def _state_payload(self):
        state = detect_state()
        payload = {"state": state, "dataset": dataset_stats()}
        if state == "idle":
            payload["videos"] = scan_videos()
        elif state == "processing":
            payload["progress"] = {**PROGRESS, "log": list(LOG)}
        elif state == "montage_pending":
            pass
        elif state in ("label", "commit"):
            payload["tracks"] = TRACKS
            payload["stats"] = label_stats()
            payload["cameras"] = sorted({t["cam"] for t in TRACKS})
            payload["ds_pids"] = [d["pid"] for d in existing_identities()]  # ID đã có trong dataset
            payload["next_pid"] = next_pid()
        self._json(payload)

    # --- POST ---
    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/start":
            cameras = body.get("cameras", [])
            if not cameras:
                self._json({"error": "Không có camera"}, 400); return
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            # Đặt running=True NGAY (đồng bộ) để /api/state gọi liền sau đó
            # nhận đúng trạng thái "processing" — tránh race khiến nút như bị "đơ".
            PROGRESS.update(running=True, phase="extract", done=False, error="")
            LOG.clear(); LOG.append("Đang khởi động…")
            threading.Thread(target=run_pipeline, args=(cameras,), daemon=True).start()
            self._json({"ok": True}); return

        if path == "/api/resume_montage":
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            def _resume():
                PROGRESS.update(running=True, phase="montage", done=False, error="")
                LOG.clear(); LOG.append("=== MONTAGE (tiếp tục) ===")
                try:
                    write_config("make_montages.py", {
                        "CROPS_DIR": CROPS_DIR, "OUTPUT_DIR": MONTAGE_DIR,
                        "MIN_IMAGES": MONTAGE_MIN_IMAGES})
                    if not run_step("make_montages.py"):
                        raise RuntimeError("Montage lỗi")
                    scan_to_labels()
                    PROGRESS.update(phase="done", done=True)
                except Exception as e:
                    PROGRESS["error"] = str(e); LOG.append(f"[LỖI] {e}")
                finally:
                    PROGRESS["running"] = False
            threading.Thread(target=_resume, daemon=True).start()
            self._json({"ok": True}); return

        # Các thao tác gán nhãn
        td = body.get("track_dir", "")
        rec = BY_DIR.get(td)

        if path == "/api/label":
            if not rec: self._json({"error": "not found"}, 404); return
            with _LOCK: rec["global_pid"] = int(body.get("global_pid", UNSEEN))
            save_csv(); self._json({"ok": True, "stats": label_stats(), "next_pid": next_pid()}); return

        if path == "/api/skip":
            if not rec: self._json({"error": "not found"}, 404); return
            with _LOCK: rec["global_pid"] = SKIP
            save_csv(); self._json({"ok": True, "stats": label_stats()}); return

        if path == "/api/delete":
            if not rec: self._json({"error": "not found"}, 404); return
            folder = Path(CROPS_DIR) / td
            if folder.exists(): shutil.rmtree(str(folder))
            if rec["montage"] and Path(rec["montage"]).exists(): Path(rec["montage"]).unlink()
            with _LOCK:
                TRACKS.remove(rec); BY_DIR.pop(td, None)
            save_csv(); self._json({"ok": True, "stats": label_stats()}); return

        if path == "/api/delete_image":
            # Xóa 1 ảnh lẻ trong track (xử lý ca nhảy ID: 1 track lẫn 2 người)
            name = Path(body.get("name", "")).name
            f = Path(CROPS_DIR) / td / name
            if not (rec and name.endswith(".jpg") and f.exists()):
                self._json({"error": "not found"}, 404); return
            f.unlink()
            with _LOCK:
                remain = len(list((Path(CROPS_DIR) / td).glob("*.jpg")))
                rec["num_images"] = remain
            save_csv()
            self._json({"ok": True, "num_images": rec["num_images"]}); return

        if path == "/api/reset_batch":
            # Xóa TẤT CẢ dữ liệu đợt đang làm (crops, montages, labels, video) — về idle.
            # KHÔNG đụng dataset myreid.
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            for d in (Path(CROPS_DIR), Path(MONTAGE_DIR)):
                if d.exists():
                    shutil.rmtree(str(d))
            if Path(LABELS_CSV).exists():
                Path(LABELS_CSV).unlink()
            for f in _mp4_files(Path(VIDEO_DIR)):
                f.unlink()
            with _LOCK:
                _load_rows([])
            self._json({"ok": True}); return

        # ── Thao tác trên dataset myreid (trình duyệt dataset) ──
        if path == "/api/ds_delete_images":
            pid = body.get("pid")
            images = body.get("images", [])
            if not isinstance(images, list) or not images:
                self._json({"error": "no images"}, 400); return
            deleted = ds_delete_images(images)
            # Nếu identity không còn ảnh nào → xóa luôn avatar mồ côi
            remaining = identity_images(int(pid)) if isinstance(pid, int) else []
            if isinstance(pid, int) and not remaining:
                _, _, _, idd = ds_dirs()
                av = idd / f"pid_{int(pid):04d}.jpg"
                if av.exists():
                    av.unlink()
            self._json({"ok": True, "deleted": deleted,
                        "remaining": len(remaining), "dataset": dataset_stats()}); return

        if path == "/api/ds_delete_identity":
            pid = body.get("pid")
            if not isinstance(pid, int):
                self._json({"error": "bad pid"}, 400); return
            deleted = ds_delete_identity(pid)
            self._json({"ok": True, "deleted": deleted, "dataset": dataset_stats()}); return

        if path == "/api/ds_set_avatar":
            pid = body.get("pid")
            ok = isinstance(pid, int) and ds_set_avatar(pid, body.get("split", ""),
                                                         body.get("name", ""))
            self._json({"ok": bool(ok)}, 200 if ok else 400); return

        if path == "/api/commit":
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            result = commit_to_dataset()
            self._json({"ok": True, "result": result}); return

        self._json({"error": "not found"}, 404)


# ── Khởi động ─────────────────────────────────────────────────────────────────
def main():
    import os
    os.chdir(HERE)   # mọi đường dẫn tương đối (crops/, dataset/, video/...) bám theo script

    # Nạp trạng thái hiện có (nếu đang dở)
    if Path(LABELS_CSV).exists():
        load_labels_from_disk()

    url = f"http://{HOST}:{PORT}"
    st = detect_state()
    print(f"\nRe-ID Pipeline server: {url}")
    print(f"Trạng thái hiện tại: {st}")
    print("Nhấn Ctrl+C để dừng.\n")

    if OPEN_BROWSER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
        server.shutdown()


if __name__ == "__main__":
    main()
