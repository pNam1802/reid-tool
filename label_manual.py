# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
CROPS_DIR   = "crops"        # thư mục output của extract_crops.py
MONTAGE_DIR = "montages"     # thư mục output của make_montages.py
LABELS_CSV  = "labels.csv"   # file CSV sẽ được tạo/cập nhật
# =================================================================
# Cách chạy:   streamlit run label_manual.py
#
# KHÔNG cần chạy embed_and_suggest.py.
# Script này tự tạo labels.csv từ thư mục crops nếu chưa có.
#

import re

import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Re-ID Manual Label",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tạo / cập nhật labels.csv từ thư mục crops ───────────────────────────────
def init_labels_csv(crops_dir: Path, montage_dir: Path) -> pd.DataFrame:
    """
    Quét crops_dir tìm tất cả tracklet.
    - Nếu labels.csv chưa có  → tạo mới, global_pid = -1.
    - Nếu labels.csv đã có    → CHỈ THÊM track mới, giữ nguyên nhãn cũ.
    Hỗ trợ cả thư mục cũ (cam1/) và mới (cam1_b20260612/).
    """
    csv_path = Path(LABELS_CSV)

    # Đọc track đã có để không ghi đè nhãn cũ
    existing_dirs: set[str] = set()
    df_existing = pd.DataFrame()
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        if "track_dir" in df_existing.columns:
            existing_dirs = set(df_existing["track_dir"].dropna().tolist())

    new_rows = []
    for track_dir in sorted(crops_dir.glob("cam*/track*")):
        images = sorted(track_dir.glob("*.jpg"))
        if not images:
            continue

        # Parse cam_id: hoạt động với cả "cam1" lẫn "cam1_b20260612"
        m = re.search(r"cam(\d+)", track_dir.parent.name)
        if not m:
            continue
        cam_id   = int(m.group(1))
        track_id = int(track_dir.name.replace("track", ""))

        # Đường dẫn tương đối để nhận diện track duy nhất qua các batch
        track_dir_rel = str(track_dir.relative_to(crops_dir)).replace("\\", "/")
        if track_dir_rel in existing_dirs:
            continue   # track này đã có trong CSV, bỏ qua

        # Tìm montage tương ứng (tên gồm đầy đủ folder để tránh trùng batch)
        mp = montage_dir / f"{track_dir.parent.name}_{track_dir.name}.jpg"
        new_rows.append({
            "cam":       cam_id,
            "track_id":  track_id,
            "num_images": len(images),
            "montage":   str(mp).replace("\\", "/") if mp.exists() else "",
            "global_pid": -1,
            "track_dir": track_dir_rel,   # ví dụ: cam1_b20260612/track0001
        })

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_out = pd.concat([df_existing, df_new], ignore_index=True) if not df_existing.empty else df_new
        df_out.to_csv(csv_path, index=False)
        return df_out

    # Không có track mới — trả về CSV hiện tại hoặc DataFrame rỗng
    return df_existing if not df_existing.empty else pd.DataFrame()


# ── Load dữ liệu ──────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    csv_path   = Path(LABELS_CSV)
    crops_path = Path(CROPS_DIR)

    if crops_path.exists():
        # Luôn quét crops: tạo mới nếu chưa có CSV, hoặc THÊM track mới
        # của batch mới vào CSV cũ (nhãn đã gán được giữ nguyên).
        n_before = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
        df_loaded = init_labels_csv(crops_path, Path(MONTAGE_DIR))
        if df_loaded.empty:
            st.error(f"Không tìm thấy track nào trong {CROPS_DIR}.")
            st.stop()
        if "global_pid" not in df_loaded.columns:
            df_loaded["global_pid"] = -1
        st.session_state.df = df_loaded
        n_new = len(df_loaded) - n_before
        if n_new > 0 and n_before > 0:
            st.toast(f"Đã thêm {n_new} track mới vào labels.csv", icon="🆕")
    elif csv_path.exists():
        st.session_state.df = pd.read_csv(csv_path)
        if "global_pid" not in st.session_state.df.columns:
            st.session_state.df["global_pid"] = -1
    else:
        st.error(f"Không tìm thấy {LABELS_CSV} lẫn {CROPS_DIR}.")
        st.stop()

if "label_idx"   not in st.session_state: st.session_state.label_idx   = 0
if "label_cam"   not in st.session_state: st.session_state.label_cam   = None
if "ref_cam"     not in st.session_state: st.session_state.ref_cam     = None
if "_prev_lcam"  not in st.session_state: st.session_state._prev_lcam  = "__init__"

df      = st.session_state.df
cameras = sorted(df["cam"].unique().tolist())

if st.session_state.label_cam not in cameras:
    st.session_state.label_cam = cameras[0]
if st.session_state.ref_cam not in cameras:
    st.session_state.ref_cam = cameras[0]


# ── SIDEBAR: Gallery tham chiếu ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖼 Gallery tham chiếu")
    st.caption("Nhìn bảng này để tìm PID phù hợp khi label camera khác.")

    ref_cam = st.radio(
        "Hiện camera:", [f"Cam {c}" for c in cameras],
        index=cameras.index(st.session_state.ref_cam),
        horizontal=True,
        key="ref_cam_radio",
    )
    ref_cam_num = int(ref_cam.split()[-1])
    st.session_state.ref_cam = ref_cam_num

    ref_df = (df[df["cam"] == ref_cam_num]
              .sort_values(["global_pid", "track_id"])
              .copy())

    if ref_df.empty:
        st.info("Không có track nào.")
    else:
        st.caption(f"{len(ref_df)} track  ·  "
                   f"{int((ref_df['global_pid'] >= 0).sum())} đã gán PID")

        gcols = st.columns(2)
        for i, (_, row) in enumerate(ref_df.iterrows()):
            pid = int(row["global_pid"])
            mp  = Path(str(row["montage"])) if row["montage"] else None

            with gcols[i % 2]:
                if mp and mp.exists():
                    st.image(str(mp), use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#1e293b;height:70px;border-radius:4px;"
                        "display:flex;align-items:center;justify-content:center'>"
                        "<span style='color:#475569;font-size:10px'>no img</span></div>",
                        unsafe_allow_html=True,
                    )

                if pid == -1:
                    badge = "<span style='color:#94a3b8'>chưa gán</span>"
                else:
                    badge = f"<span style='color:#facc15;font-weight:bold'>PID {pid}</span>"

                st.markdown(
                    f"<div style='text-align:center;font-size:10px;line-height:1.5;"
                    f"margin-bottom:8px'>t{int(row['track_id']):04d}<br>{badge}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # Thống kê tổng
    total    = len(df)
    assigned = int((df["global_pid"] >= 0).sum())
    skipped  = int((df["global_pid"] == -1).sum())
    st.markdown(f"""
**Tiến trình tổng:**
🟡 Chưa gán: **{total - assigned - skipped + skipped}** → thực ra ban đầu tất cả là -1
🟢 Đã gán PID: **{assigned}**
🔴 Bỏ qua: **{skipped}**
⬜ Tổng track: **{total}**
    """)
    # Simpler stats
    st.progress(assigned / total if total > 0 else 0,
                text=f"{assigned}/{total} track đã gán PID")

    st.divider()

    # ── Panel "PID đã gán" — 1 ảnh đại diện mỗi PID ──────────────
    st.markdown("### 📌 PID đã gán")
    st.caption("Ảnh đại diện của từng PID — dùng để đối chiếu khi label camera khác.")

    assigned_pids = sorted(df[df["global_pid"] >= 0]["global_pid"].unique().tolist())

    if not assigned_pids:
        st.caption("_(Chưa có PID nào được gán)_")
    else:
        pcols = st.columns(2)
        for i, pid in enumerate(assigned_pids):
            # Lấy track đầu tiên được gán PID này làm đại diện
            rep = df[df["global_pid"] == pid].iloc[0]
            mp  = Path(str(rep["montage"])) if rep["montage"] else None

            with pcols[i % 2]:
                if mp and mp.exists():
                    st.image(str(mp), use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#1e293b;height:70px;border-radius:4px;"
                        "display:flex;align-items:center;justify-content:center'>"
                        "<span style='color:#475569;font-size:10px'>no img</span></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div style='text-align:center;font-size:12px;font-weight:bold;"
                    f"color:#facc15;margin-bottom:10px'>PID {pid}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    if st.button("💾  Lưu CSV", type="primary", use_container_width=True):
        df.to_csv(LABELS_CSV, index=False)
        st.success(f"✓  Đã lưu → {LABELS_CSV}")


# ── MAIN: Track đang label ────────────────────────────────────────────────────
st.markdown("## ✏️ Label track")

# Chọn camera để label
cam_options = [f"Cam {c}" for c in cameras]
cur_sel = f"Cam {st.session_state.label_cam}"
if cur_sel not in cam_options:
    cur_sel = cam_options[0]

label_cam_sel = st.radio(
    "Đang label camera:", cam_options,
    index=cam_options.index(cur_sel),
    horizontal=True,
    key="label_cam_radio",
)
cam_num = int(label_cam_sel.split()[-1])
st.session_state.label_cam = cam_num

# Reset index khi đổi camera
if label_cam_sel != st.session_state._prev_lcam:
    st.session_state.label_idx = 0
    st.session_state._prev_lcam = label_cam_sel

# Lấy danh sách track của camera đang label, giữ original index
filtered = (df[df["cam"] == cam_num]
            .reset_index()           # cột 'index' = index gốc trong df
            .rename(columns={"index": "_orig_idx"}))
n_tracks = len(filtered)

if n_tracks == 0:
    st.info("Không có track nào.")
    st.stop()

label_idx = min(max(st.session_state.label_idx, 0), n_tracks - 1)
current   = filtered.iloc[label_idx]
orig_idx  = int(current["_orig_idx"])

# ── Điều hướng ────────────────────────────────────────────────────
nav_l, nav_mid, nav_r = st.columns([1, 5, 1])
with nav_l:
    if st.button("◀", use_container_width=True, disabled=(label_idx == 0)):
        st.session_state.label_idx -= 1
        st.rerun()
with nav_mid:
    done_in_cam = int((filtered["global_pid"] >= 0).sum())
    st.markdown(
        f"<p style='text-align:center;font-size:17px;margin:6px 0'>"
        f"Track  <b>{label_idx + 1}</b>  /  {n_tracks}"
        f"  &nbsp;·&nbsp;  đã gán: <b>{done_in_cam}</b> / {n_tracks}</p>",
        unsafe_allow_html=True,
    )
with nav_r:
    if st.button("▶", use_container_width=True, disabled=(label_idx == n_tracks - 1)):
        st.session_state.label_idx += 1
        st.rerun()

st.divider()

# ── Hiển thị track hiện tại ───────────────────────────────────────
img_col, ctrl_col = st.columns([3, 2])

with img_col:
    mp = Path(str(current["montage"])) if current["montage"] else None
    if mp and mp.exists():
        st.image(str(mp), use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#1e293b;height:280px;border-radius:8px;"
            "display:flex;align-items:center;justify-content:center'>"
            "<span style='color:#475569'>Không có ảnh montage</span></div>",
            unsafe_allow_html=True,
        )

with ctrl_col:
    cur_pid = int(current["global_pid"])

    st.markdown(
        f"### cam {int(current['cam'])}  &nbsp;/&nbsp;  track `{int(current['track_id']):04d}`  \n"
        f"**{int(current['num_images'])} ảnh**"
    )

    if cur_pid == -1:
        st.warning("Trạng thái: chưa gán / bỏ qua")
    else:
        st.success(f"Trạng thái: **PID = {cur_pid}**")

    st.markdown("---")

    # Gợi ý PID tiếp theo nếu người này chưa có trong gallery
    max_pid     = int(df["global_pid"].max())
    next_new    = max_pid + 1 if max_pid >= 0 else 0
    default_val = cur_pid if cur_pid >= 0 else next_new

    st.caption(
        f"💡 Nếu người này chưa có trong gallery → dùng PID mới: **{next_new}**"
    )

    new_pid = st.number_input(
        "Nhập PID (nhìn gallery bên trái):",
        min_value=0,
        value=default_val,
        step=1,
        key=f"pid_{int(current['cam'])}_{int(current['track_id'])}",
        help="Gõ số PID của người trong gallery bên trái. "
             "Nếu người mới chưa có thì dùng số gợi ý ở trên.",
    )

    ca, cb = st.columns(2)
    with ca:
        if st.button("✅  Gán + Tiếp", use_container_width=True, type="primary"):
            df.at[orig_idx, "global_pid"] = int(new_pid)
            df.to_csv(LABELS_CSV, index=False)
            if label_idx < n_tracks - 1:
                st.session_state.label_idx += 1
            st.rerun()
    with cb:
        if st.button("🗑  Bỏ qua", use_container_width=True):
            df.at[orig_idx, "global_pid"] = -1
            df.to_csv(LABELS_CSV, index=False)
            if label_idx < n_tracks - 1:
                st.session_state.label_idx += 1
            st.rerun()

    st.markdown("---")

    # Nhảy tới track bất kỳ
    jump = st.number_input(
        "Nhảy tới track:", min_value=1, max_value=n_tracks,
        value=label_idx + 1, step=1, key="jump",
    )
    if st.button("Đi", use_container_width=True):
        st.session_state.label_idx = int(jump) - 1
        st.rerun()

    st.markdown("---")

    # ── Xóa track (xóa cả ảnh lẫn dòng CSV) ─────────────────────
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    if not st.session_state.confirm_delete:
        if st.button("🗑  Xóa track này", use_container_width=True):
            st.session_state.confirm_delete = True
            st.rerun()
    else:
        st.error(
            f"Xóa **cam{int(current['cam'])}/track{int(current['track_id']):04d}** "
            f"và toàn bộ **{int(current['num_images'])} ảnh**?"
        )
        cd1, cd2 = st.columns(2)
        with cd1:
            if st.button("✓  Xác nhận xóa", use_container_width=True, type="primary"):
                import shutil

                # Xóa thư mục ảnh (ưu tiên track_dir batch-aware)
                tdr = str(current.get("track_dir", "") or "")
                if tdr and tdr != "nan":
                    track_dir = Path(CROPS_DIR) / tdr
                else:
                    track_dir = (Path(CROPS_DIR)
                                 / f"cam{int(current['cam'])}"
                                 / f"track{int(current['track_id']):04d}")
                if track_dir.exists():
                    shutil.rmtree(str(track_dir))

                # Xóa file montage nếu có
                mp_del = Path(str(current["montage"])) if current["montage"] else None
                if mp_del and mp_del.exists():
                    mp_del.unlink()

                # Xóa dòng khỏi DataFrame và lưu CSV
                df.drop(index=orig_idx, inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.to_csv(LABELS_CSV, index=False)

                # Cập nhật session_state
                st.session_state.df = df
                st.session_state.confirm_delete = False
                st.session_state.label_idx = min(
                    st.session_state.label_idx, len(df) - 1
                )
                st.rerun()
        with cd2:
            if st.button("✗  Hủy", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()
