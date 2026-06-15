# Re-ID Pipeline Dashboard
# Chạy: streamlit run pipeline.py

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Re-ID Pipeline", layout="wide")

PYTHON = sys.executable
SCRIPTS = {
    "extract":  "extract_crops.py",
    "montage":  "make_montages.py",
    "label":    "label_manual.py",
    "build":    "build_dataset.py",
}


# ── Đọc / ghi CONFIG block trong script ───────────────────────────────────────

def read_config(script: str) -> dict:
    config = {}
    with open(script, encoding="utf-8") as f:
        in_block = False
        for line in f:
            if "CẤU HÌNH" in line:
                in_block = True; continue
            if in_block and line.strip().startswith("# ==="):
                break
            if in_block and "=" in line and not line.strip().startswith("#"):
                key, _, rest = line.partition("=")
                val_str = rest.split("#")[0].strip().strip('"').strip("'")
                try:    config[key.strip()] = int(val_str)
                except ValueError:
                    try: config[key.strip()] = float(val_str)
                    except ValueError:
                         config[key.strip()] = val_str
    return config


def write_config(script: str, updates: dict) -> None:
    with open(script, encoding="utf-8") as f:
        content = f.read()
    for key, value in updates.items():
        if isinstance(value, bool):
            # Phải kiểm tra bool TRƯỚC int (bool là con của int trong Python)
            pattern = rf'^({re.escape(key)}\s*=\s*)(?:True|False)'
            new_val = "True" if value else "False"
        elif isinstance(value, str):
            pattern = rf'^({re.escape(key)}\s*=\s*)"[^"]*"'
            new_val = f'"{value}"'
        else:
            pattern = rf'^({re.escape(key)}\s*=\s*)[\d.]+'
            new_val = str(value)
        # Dùng lambda để giá trị không bị regex diễn giải (\1 + "1" → \11, hay
        # backslash trong đường dẫn Windows làm hỏng chuỗi thay thế)
        content = re.sub(pattern, lambda m, v=new_val: m.group(1) + v,
                         content, flags=re.MULTILINE)
    with open(script, "w", encoding="utf-8") as f:
        f.write(content)


# ── Chạy script, stream log trực tiếp lên UI ──────────────────────────────────

def run_script(script: str, extra_input: str = "",
               log_box=None, tail: int = 30) -> tuple[bool, str]:
    """Chạy script con, đọc stdout từng dòng và cập nhật log_box (st.empty())
    theo thời gian thực. Trả về (ok, toàn bộ output)."""
    proc = subprocess.Popen(
        [PYTHON, "-u", script],          # -u: không buffer stdout để log hiện ngay
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,        # gộp stderr vào stdout theo đúng thứ tự
        text=True, encoding="utf-8", errors="replace",
    )
    # Trả lời sẵn prompt [y/N] nếu script hỏi
    if extra_input:
        proc.stdin.write(extra_input)
        proc.stdin.flush()
    proc.stdin.close()

    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        if log_box is not None:
            # Chỉ hiện đuôi log để UI nhẹ, toàn bộ log trả về sau khi xong
            log_box.code("\n".join(lines[-tail:]) or "(đang khởi động...)")
    proc.wait()
    return proc.returncode == 0, "\n".join(lines)


# ── Kiểm tra trạng thái từng bước ─────────────────────────────────────────────

def status_extract(crops_dir="crops"):
    p = Path(crops_dir)
    if not p.exists(): return {}
    return {
        cam.name: len(list(cam.glob("track*")))
        for cam in sorted(p.glob("cam*"))
    }

def status_montage(montage_dir="montages"):
    p = Path(montage_dir)
    return len(list(p.glob("*.jpg"))) if p.exists() else 0

def status_label(labels_csv="labels.csv"):
    if not Path(labels_csv).exists(): return None
    df = pd.read_csv(labels_csv)
    return df

def status_build(out_dir="myreid"):
    p = Path(out_dir)
    if not p.exists(): return None
    return {
        "train":   len(list((p / "bounding_box_train").glob("*.jpg"))) if (p / "bounding_box_train").exists() else 0,
        "query":   len(list((p / "query").glob("*.jpg"))) if (p / "query").exists() else 0,
        "gallery": len(list((p / "bounding_box_test").glob("*.jpg"))) if (p / "bounding_box_test").exists() else 0,
    }


# ── Badge trạng thái ──────────────────────────────────────────────────────────

def badge(ok: bool, label: str) -> str:
    icon = "✅" if ok else "⬜"
    return f"{icon} {label}"


# ══════════════════════════════════════════════════════════════════════════════
st.title("🔄 Re-ID Pipeline")
st.caption("Chạy lần lượt từ Bước 1 → Bước 4. Mỗi tab là một bước trong pipeline.")

# ── Tổng quan trạng thái ──────────────────────────────────────────────────────
ext_st  = status_extract()
mon_st  = status_montage()
lbl_df  = status_label()
bld_st  = status_build()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bước 1  Extract", f"{sum(ext_st.values())} track" if ext_st else "Chưa có",
          f"{len(ext_st)} camera" if ext_st else None)
c2.metric("Bước 2  Montage", f"{mon_st} file" if mon_st else "Chưa có")
if lbl_df is not None:
    labeled = int((lbl_df["global_pid"] >= 0).sum())
    c3.metric("Bước 3  Label", f"{labeled} / {len(lbl_df)} track",
              "labels.csv tồn tại")
else:
    c3.metric("Bước 3  Label", "Chưa có labels.csv")
if bld_st:
    c4.metric("Bước 4  Dataset", f"{bld_st['train']} train",
              f"{bld_st['query']} query · {bld_st['gallery']} gallery")
else:
    c4.metric("Bước 4  Dataset", "Chưa build")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    badge(bool(ext_st),  " Bước 1 — Extract crops"),
    badge(mon_st > 0,    " Bước 2 — Montages"),
    badge(lbl_df is not None and int((lbl_df["global_pid"] >= 0).sum()) == len(lbl_df),
          " Bước 3 — Label"),
    badge(bool(bld_st),  " Bước 4 — Build dataset"),
])


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Extract crops
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Bước 1 — Extract crops từ video")

    # ── Tự động quét thư mục video ────────────────────────────────
    VIDEO_DIR = st.text_input("Thư mục chứa video:", value="video",
                               help="Đặt tất cả file .mp4 vào đây, pipeline tự nhận.")

    def scan_videos(folder: str) -> list[dict]:
        """Quét folder, sắp xếp theo tên, gán cam_id 1,2,3..."""
        p = Path(folder)
        if not p.exists():
            return []
        files = sorted(p.glob("*.mp4")) + sorted(p.glob("*.MP4"))
        return [{"video": str(f).replace("\\", "/"), "cam_id": i + 1}
                for i, f in enumerate(files)]

    # Tự động load khi folder thay đổi hoặc chưa có cameras
    detected = scan_videos(VIDEO_DIR)
    if "cameras" not in st.session_state or "video_dir" not in st.session_state:
        st.session_state.cameras   = detected or [{"video": "", "cam_id": 1}]
        st.session_state.video_dir = VIDEO_DIR

    if st.session_state.video_dir != VIDEO_DIR:
        # Folder đổi → quét lại
        st.session_state.cameras   = detected or [{"video": "", "cam_id": 1}]
        st.session_state.video_dir = VIDEO_DIR

    col_scan, _ = st.columns([2, 5])
    if col_scan.button("🔍  Quét lại thư mục"):
        st.session_state.cameras   = detected or [{"video": "", "cam_id": 1}]
        st.session_state.video_dir = VIDEO_DIR
        st.rerun()

    if detected:
        st.caption(f"Tìm thấy **{len(detected)}** file .mp4 trong `{VIDEO_DIR}/` — "
                   f"tự động gán cam_id theo thứ tự tên file.")
    else:
        st.warning(f"Không tìm thấy file .mp4 nào trong `{VIDEO_DIR}/`. "
                   f"Hãy đặt video vào đúng thư mục hoặc điền thủ công bên dưới.")

    st.divider()

    # ── Danh sách camera (có thể sửa thủ công) ────────────────────
    st.caption("Kiểm tra và chỉnh lại nếu cần:")

    for i, cam in enumerate(st.session_state.cameras):
        with st.container(border=True):
            col_v, col_c, col_del = st.columns([5, 1, 1])
            with col_v:
                st.session_state.cameras[i]["video"] = st.text_input(
                    "Đường dẫn video", value=cam["video"],
                    key=f"vid_{i}", placeholder="video/cam1.mp4"
                )
            with col_c:
                st.session_state.cameras[i]["cam_id"] = st.number_input(
                    "Cam ID", value=int(cam["cam_id"]),
                    min_value=1, step=1, key=f"cid_{i}"
                )
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{i}",
                             disabled=(len(st.session_state.cameras) == 1)):
                    st.session_state.cameras.pop(i)
                    st.rerun()

            cid = st.session_state.cameras[i]["cam_id"]
            # Cộng dồn track của cam này qua mọi batch (cam1, cam1_b20260612, ...)
            n_tracks = sum(
                n for name, n in ext_st.items()
                if re.fullmatch(rf"cam{cid}(_b.+)?", name)
            )
            if n_tracks:
                st.success(f"✅  cam{cid}: {n_tracks} track đã extract (mọi batch)")
            else:
                st.warning(f"⬜  cam{cid}: chưa có dữ liệu")

    if st.button("➕  Thêm camera thủ công"):
        st.session_state.cameras.append({"video": "", "cam_id": len(st.session_state.cameras) + 1})
        st.rerun()

    st.divider()

    # Cấu hình chung
    cfg_ex = read_config(SCRIPTS["extract"])
    col_a, col_b, col_c, col_d = st.columns(4)
    se  = col_a.number_input("Save every (frame)", value=int(cfg_ex.get("SAVE_EVERY", 8)),    min_value=1, step=1)
    mh  = col_b.number_input("Min height (px)",    value=int(cfg_ex.get("MIN_HEIGHT", 80)),    min_value=1, step=1)
    mpt = col_c.number_input("Max per track",       value=int(cfg_ex.get("MAX_PER_TRACK", 25)), min_value=1, step=1)
    mgap = col_d.number_input("Min gap (frame)",    value=int(cfg_ex.get("MIN_GAP", 24)),       min_value=1, step=1,
                              help="Khoảng cách frame tối thiểu giữa 2 ảnh cùng track — chống ảnh trùng lặp khi người đứng yên")
    out_ex = st.text_input("Output dir", value=cfg_ex.get("OUTPUT_DIR", "crops"),
                           key="out_dir_extract")

    force_clean = st.checkbox(
        "Xóa dữ liệu cũ của camera trước khi chạy (làm lại từ đầu)", value=False,
        help="Bỏ tick = đánh số track NỐI TIẾP dữ liệu cũ (tích lũy nhiều lần quay)."
    )

    if st.button("▶  Chạy extract cho tất cả camera", type="primary"):
        for cam in st.session_state.cameras:
            if not cam["video"]:
                st.warning(f"Cam {cam['cam_id']}: chưa điền đường dẫn video — bỏ qua.")
                continue

            st.info(f"Đang chạy cam {cam['cam_id']}: {cam['video']} ...")
            write_config(SCRIPTS["extract"], {
                "VIDEO_PATH":    cam["video"],
                "CAM_ID":        cam["cam_id"],
                "OUTPUT_DIR":    out_ex,
                "RESET_CAM":     bool(force_clean),
                "SAVE_EVERY":    se,
                "MIN_HEIGHT":    mh,
                "MAX_PER_TRACK": mpt,
                "MIN_GAP":       mgap,
            })

            log_box = st.empty()   # log cập nhật trực tiếp tại đây
            ok, output = run_script(SCRIPTS["extract"], log_box=log_box)
            log_box.empty()

            with st.expander(f"Log đầy đủ cam {cam['cam_id']}", expanded=not ok):
                st.code(output)
            if ok:
                st.success(f"✅  cam {cam['cam_id']} xong!")
            else:
                st.error(f"❌  cam {cam['cam_id']} lỗi — xem log bên trên.")
        st.button("🔄  Cập nhật trạng thái", key="refresh_extract")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Montages
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bước 2 — Tạo ảnh lưới (montage)")
    st.caption("Tạo ảnh đại diện cho từng tracklet. Dùng để xem khi gán nhãn.")

    cfg_m = read_config(SCRIPTS["montage"])
    col1, col2 = st.columns(2)
    crops_in  = col1.text_input("Crops dir",   value=cfg_m.get("CROPS_DIR", "crops"),
                                key="crops_dir_montage")
    mon_out   = col2.text_input("Montage dir", value=cfg_m.get("OUTPUT_DIR", "montages"),
                                key="montage_dir_montage")

    col3, col4, col5 = st.columns(3)
    max_img = col3.number_input("Max ảnh/track", value=int(cfg_m.get("MAX_IMAGES", 10)),  min_value=1, step=1)
    cols_m  = col4.number_input("Số cột",        value=int(cfg_m.get("COLS", 5)),          min_value=1, step=1)
    min_img = col5.number_input("Min ảnh",        value=int(cfg_m.get("MIN_IMAGES", 3)),    min_value=1, step=1)

    if mon_st:
        st.success(f"✅  {mon_st} file montage hiện có trong {mon_out}/")

    if st.button("▶  Chạy make_montages", type="primary"):
        write_config(SCRIPTS["montage"], {
            "CROPS_DIR":  crops_in,
            "OUTPUT_DIR": mon_out,
            "MAX_IMAGES": max_img,
            "COLS":       cols_m,
            "MIN_IMAGES": min_img,
        })
        log_box = st.empty()
        ok, output = run_script(SCRIPTS["montage"], log_box=log_box)
        log_box.empty()
        with st.expander("Log đầy đủ", expanded=not ok):
            st.code(output)
        if ok: st.success("✅  Xong!")
        else:  st.error("❌  Lỗi — xem log bên trên.")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Label
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Bước 3 — Gán nhãn thủ công")
    st.caption("Mở Label Tool trong một terminal riêng.")

    if lbl_df is not None:
        total   = len(lbl_df)
        labeled = int((lbl_df["global_pid"] >= 0).sum())
        skipped = int((lbl_df["global_pid"] == -1).sum())
        not_yet = total - labeled

        st.progress(labeled / total if total else 0,
                    text=f"Đã gán PID: {labeled}/{total} track")
        c1, c2, c3 = st.columns(3)
        c1.metric("Đã gán PID", labeled)
        c2.metric("Bỏ qua (-1)", skipped)
        c3.metric("Chưa gán", not_yet)

        if not_yet == 0:
            st.success("✅  Tất cả track đã được gán nhãn!")
    else:
        st.info("labels.csv chưa có. Chạy bước 1 và 2 trước.")

    st.divider()
    st.markdown("**Mở Label Tool** (web app — chạy lệnh này trong terminal mới):")
    st.code("python label_server.py")
    st.caption("Tự mở trình duyệt tại http://127.0.0.1:8000 — gán nhãn bằng bàn phím "
               "(số = PID, Enter = gán + sang kế, Space = cùng PID track trước, "
               "←/→ di chuyển, S = bỏ qua, Del = xóa). "
               "Lần đầu cần: pip install fastapi uvicorn")

    if st.button("🔄  Làm mới trạng thái"):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: Build dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Bước 4 — Build dataset (Market-1501)")
    st.caption("Tạo dataset chuẩn để train TransReID. Chạy sau khi gán nhãn xong.")

    # Cảnh báo nếu còn track chưa gán nhãn
    if lbl_df is not None:
        not_yet = len(lbl_df) - int((lbl_df["global_pid"] >= 0).sum()) - int((lbl_df["global_pid"] == -1).sum())
        # Actually all start at -1, so "not_yet" = tracks with pid == -1 that user hasn't reviewed
        # Simpler: just show labeled count
        labeled = int((lbl_df["global_pid"] >= 0).sum())
        if labeled == 0:
            st.warning("⚠️  Chưa có track nào được gán PID. Hãy hoàn thành Bước 3 trước.")

    cfg_b = read_config(SCRIPTS["build"])
    col1, col2 = st.columns(2)
    crops_b  = col1.text_input("Crops dir",   value=cfg_b.get("CROPS_DIR", "crops"),
                               key="crops_dir_build")
    labels_b = col1.text_input("Labels CSV",  value=cfg_b.get("LABELS_CSV", "labels.csv"),
                               key="labels_csv_build")
    out_b    = col2.text_input("Output dir",  value=cfg_b.get("OUTPUT_DIR", "myreid"),
                               key="out_dir_build")
    ratio_b  = col2.slider("Test ratio", 0.0, 0.5,
                            value=float(cfg_b.get("TEST_RATIO", 0.3)), step=0.05)

    if bld_st:
        st.success(
            f"✅  Dataset hiện có: "
            f"train={bld_st['train']}  query={bld_st['query']}  gallery={bld_st['gallery']}"
        )

    if st.button("▶  Build dataset", type="primary"):
        write_config(SCRIPTS["build"], {
            "CROPS_DIR":  crops_b,
            "LABELS_CSV": labels_b,
            "OUTPUT_DIR": out_b,
            "TEST_RATIO": ratio_b,
        })
        log_box = st.empty()
        ok, output = run_script(SCRIPTS["build"], log_box=log_box)
        log_box.empty()
        with st.expander("Log đầy đủ", expanded=True):
            st.code(output)
        if ok: st.success("✅  Dataset build xong!")
        else:  st.error("❌  Lỗi — xem log bên trên.")
