# Hệ thống thu thập dữ liệu Person Re-Identification (Re-ID)

Pipeline thu thập ảnh **upbody** từ hệ thống 3 camera để xây dựng dataset
chuẩn **Market-1501**, phục vụ train model **TransReID**. Toàn bộ chạy qua
**một web app duy nhất**: `label_server.py`.

```
        Bỏ 3 video vào folder video/
                   │
        python label_server.py   (mở web, tự dẫn theo trạng thái)
                   │
   ┌───────────────┴───────────────────────────────────────────┐
   │ 1. Crop      → cắt ảnh upbody theo track (ByteTrack)  → crops/      (TẠM) │
   │ 2. Montage   → ảnh lưới mỗi track để gán nhãn         → montages/   (TẠM) │
   │ 3. Gán ID    → BẠN gán ID thủ công bằng bàn phím      → labels.csv  (TẠM) │
   │ 4. Gộp       → đưa ảnh đã gán vào dataset Market-1501 → dataset/    (GIỮ) │
   └───────────────┬───────────────────────────────────────────┘
                   │  Gộp xong: XÓA sạch crops/ montages/ labels.csv + video,
                   ▼  chỉ giữ dataset/ lớn dần qua từng đợt.
            dataset/  →  train TransReID
```

**Thu thập liên tục:** mỗi lần có video mới, lặp lại 4 bước; dataset tích lũy
thêm. Người đã có trong dataset được hiện lại để bạn **tái dùng ID cũ** (giữ
nhất quán identity qua các đợt).

> Các script `extract_crops.py`, `make_montages.py` vẫn là từng bước riêng,
> được `label_server.py` gọi tự động. `pipeline.py` (Streamlit) và
> `build_dataset.py` là công cụ cũ/độc lập, không bắt buộc dùng nữa.

---

## 1. Cài đặt

```bash
pip install ultralytics supervision opencv-python
```

`label_server.py` (web + điều phối) chỉ dùng **thư viện chuẩn Python** —
không cần fastapi/streamlit/torchreid. Các thư viện trên là để
`extract_crops.py` chạy (YOLO + ByteTrack + OpenCV).

Yêu cầu thêm:
- File model **`head_detect.pt`** đặt cùng thư mục với `head_detect.py`.
- Python 3.10+.

---

## Chạy nhanh

```bash
python label_server.py
```

Tự mở `http://127.0.0.1:8000`. Web tự biết đang ở bước nào:
- **Sạch** → cho bỏ video vào `video/` và bấm *Bắt đầu Crop + Montage*.
- **Đang dở** (ví dụ đã crop nhưng chưa montage/gán) → **bắt làm nốt** rồi
  mới cho thêm video mới.
- **Gán xong** → nút *Gộp vào dataset* → dọn tạm, dataset lớn lên.

**Phím tắt khi gán ID:** `0-9` gõ ID · `Enter` gán + sang kế ·
`Space` cùng ID track trước · `←`/`→` di chuyển · `S` bỏ qua · `Del` xóa track.
Quy tắc: **cùng người = cùng ID** (kể cả khác camera/khác đợt). Người đã có
trong dataset hiện ở cột phải (thẻ tím) — bấm để tái dùng ID cũ.

---

## 2. Cấu trúc thư mục

```
ReID/
├── head_detect.py        # wrapper model detect đầu (KHÔNG SỬA)
├── head_detect.pt        # weights model
├── scale_box.py          # mở rộng box đầu → vùng upbody (KHÔNG SỬA)
│
├── pipeline.py           # ⭐ DASHBOARD điều khiển toàn bộ pipeline
├── extract_crops.py      # Bước 1
├── make_montages.py      # Bước 2
├── label_manual.py       # Bước 3 (giao diện gán nhãn)
├── build_dataset.py      # Bước 4
│
├── video/                # 📂 ĐẶT VIDEO VÀO ĐÂY
│   ├── cam1.mp4
│   ├── cam2.mp4
│   └── cam3.mp4
│
├── crops/                # output bước 1 (tự tạo)
│   ├── cam1_b20260612/   #   mỗi camera × mỗi batch (ngày) 1 thư mục
│   │   ├── track0001/    #   mỗi track = 1 người được theo dõi liên tục
│   │   │   ├── f000008.jpg
│   │   │   └── ...
│   │   └── track0002/
│   └── meta_cam1_b20260612.csv
│
├── montages/             # output bước 2 (tự tạo)
│   └── cam1_b20260612_track0001.jpg
│
├── labels.csv            # output bước 3 — file nhãn trung tâm
│
└── myreid/               # output bước 4 — dataset Market-1501
    ├── bounding_box_train/
    ├── query/
    └── bounding_box_test/
```

---

## 3. Cách chạy

### Cách 1 — Dùng dashboard (khuyến nghị)

```bash
streamlit run pipeline.py
```

Mở trình duyệt tại `http://localhost:8501`. Dashboard có 4 tab tương ứng
4 bước, kèm trạng thái từng bước (✅ xong / ⬜ chưa). Chạy lần lượt:

1. **Tab Bước 1** — đặt video vào thư mục `video/`, dashboard tự quét và gán
   cam ID theo thứ tự tên file. Kiểm tra lại rồi bấm **▶ Chạy extract**.
   Log hiện trực tiếp trên màn hình.
2. **Tab Bước 2** — bấm **▶ Chạy make_montages**.
3. **Tab Bước 3** — mở terminal mới, chạy:
   ```bash
   streamlit run label_manual.py --server.port 8502
   ```
   rồi gán nhãn (xem mục 4 bên dưới).
4. **Tab Bước 4** — bấm **▶ Build dataset**.

### Cách 2 — Chạy từng script

Mỗi script có **CONFIG block** ở đầu file. Mở file, sửa biến, chạy
`python <script>.py`. Không cần truyền tham số dòng lệnh.

```python
# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
VIDEO_PATH    = "video/cam1.mp4"
CAM_ID        = 1
...
# =================================================================
```

Với 3 camera: chạy `extract_crops.py` **3 lần**, mỗi lần đổi
`VIDEO_PATH` + `CAM_ID`.

---

## 4. Chi tiết từng bước

### Bước 1 — `extract_crops.py` (tự động)

Đọc video → detect đầu người từng frame (`head_detect.py`) → mở rộng box
đầu thành vùng upbody (`scale_box.py`) → track qua frame bằng **ByteTrack**
→ lưu crop.

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `SAVE_EVERY` | 8 | lưu 1 ảnh mỗi 8 frame (tracker vẫn chạy mọi frame) |
| `MIN_HEIGHT` | 80 | bỏ crop thấp hơn 80px (người quá xa) |
| `MAX_PER_TRACK` | 25 | tối đa 25 ảnh / track |
| `BATCH_ID` | "" | để trống = tự dùng ngày hôm nay (`b20260612`) |

**Khái niệm track:** một người đi qua camera được ByteTrack theo dõi liên
tục = 1 track = 1 thư mục ảnh. Nếu người bị che khuất rồi xuất hiện lại,
có thể bị **nhảy ID** thành track mới — sẽ xử lý ở bước gán nhãn.

Chạy xong in thống kê: tổng track, số track có ≥3 ảnh, tổng ảnh.

### Bước 2 — `make_montages.py` (tự động)

Mỗi track → 1 **ảnh lưới** (tối đa 10 ảnh đại diện chọn đều theo thời gian,
xếp 5 cột). Track có < 3 ảnh bị bỏ qua. Ảnh lưới chỉ dùng để **xem bằng
mắt** khi gán nhãn, không dùng để train.

### Bước 3 — `label_server.py` (thủ công — quan trọng nhất)

```bash
python label_server.py
```

Tự mở trình duyệt tại `http://127.0.0.1:8000`. **Không cần cài thư viện
ngoài** (chỉ dùng thư viện chuẩn Python) và **không cần AI/torchreid**.

Đây là bước duy nhất cần con người: xác nhận **track nào là người nào**.
Web app này thay cho `label_manual.py` (Streamlit) cũ — mượt hơn nhiều vì
trình duyệt giữ state, chuyển track tức thì, lưu CSV chạy ngầm.

**Gán nhãn bằng bàn phím (không cần chuột):**

| Phím | Hành động |
|---|---|
| `0`–`9` | gõ số PID |
| `Enter` | gán PID vừa gõ + tự sang track kế |
| `Space` | gán **cùng PID với track trước** + sang kế |
| `→` / `←` | sang track kế / trước (không gán) |
| `S` | bỏ qua track này (pid = -1) |
| `Delete` | xóa hẳn track (ảnh gốc + montage + dòng CSV, có hỏi xác nhận) |

**Giao diện:**
- **Giữa** — montage track đang xét + thông tin cam/track/số ảnh + PID.
- **Cột phải** — panel *📌 PID đã gán* (1 ảnh đại diện mỗi PID — nhìn vào
  đây để đối chiếu khi label camera khác; bấm vào 1 PID để gán cho track
  hiện tại).
- **Thanh trên** — lọc theo camera + thanh tiến trình.

> `label_manual.py` (Streamlit) vẫn giữ lại làm bản dự phòng nếu cần.

**PID (Person ID)** — số định danh người. Quy tắc duy nhất:

> **Cùng một người ngoài đời = cùng một số PID** — bất kể khác camera,
> khác track, hay khác batch. Khác người = khác số.

Số không cần liên tục, `build_dataset.py` tự đánh lại từ 0.

**Quy trình gán hiệu quả:**

1. Label hết **Cam 1** trước: mỗi người một số PID mới (0, 1, 2...).
2. Chuyển sang **Cam 2**: nhìn ảnh track → đối chiếu panel "PID đã gán"
   → người cũ thì gõ đúng PID cũ, người mới thì dùng số gợi ý.
3. Tương tự **Cam 3**.

**Các nút:**

| Nút | Khi nào dùng |
|---|---|
| ✅ Gán + Tiếp | xác nhận PID, tự nhảy sang track kế |
| 🗑 Bỏ qua | track nhiễu / mờ / không muốn dùng → `pid = -1`, bị loại khi build |
| 🗑 Xóa track này | crop **detect sai hoàn toàn** (dính nhiều người, không phải người...) → xóa cả ảnh gốc lẫn dòng CSV (có bước xác nhận) |

**Trường hợp đặc biệt:**
- *Một người bị nhảy ID thành nhiều track* → gán **cùng PID** cho tất cả
  các track đó (đừng xóa — càng nhiều ảnh model học càng tốt).
- *Crop dính 2 người trở lên* → Bỏ qua hoặc Xóa.

Kết quả lưu vào `labels.csv` — **tự động lưu** sau mỗi lần bấm nút.

### Bước 4 — `build_dataset.py` (tự động)

Đọc `labels.csv`, bỏ track có `pid = -1`, copy ảnh thành dataset chuẩn
Market-1501:

- **Tên file:** `{pid:04d}_c{cam}s1_{frame:06d}_{k:02d}.jpg`
- **Chia train/test theo identity:**
  - Người chỉ xuất hiện ở **1 camera** → luôn vào train.
  - Người xuất hiện ở **≥2 camera** → 30% (TEST_RATIO) vào test, còn lại
    train. Người trong test **không bao giờ** xuất hiện trong train.
- **Tập test:** mỗi camera lấy 1 ảnh làm **query**, còn lại vào **gallery**.

⚠️ Nếu không có người nào xuất hiện ở ≥2 camera, script cảnh báo —
tập test sẽ rỗng, không đánh giá cross-camera được.

---

## 5. Thêm dữ liệu nhiều đợt (batch)

Pipeline hỗ trợ **tích lũy dataset** — mỗi lần có video mới:

```
Lần 1 (12/06): video vào → extract → montage → label → build
Lần 2 (15/06): video MỚI vào → extract → montage → label → build
                                                            ↑
                                              dataset to hơn lần 1
```

Cơ chế:
- Mỗi đợt extract tạo thư mục riêng theo ngày (`cam1_b20260612`,
  `cam1_b20260615`) — **không bao giờ đụng dữ liệu cũ**.
- `label_manual.py` mở lên tự phát hiện track mới và **chỉ thêm** vào
  `labels.csv`, nhãn đã gán giữ nguyên.
- Khi label batch mới: người **đã từng xuất hiện** ở batch cũ → gán đúng
  PID cũ của họ (xem panel "PID đã gán"); người mới → PID mới.
- `build_dataset.py` mỗi lần chạy xóa `myreid/` cũ và build lại từ
  **toàn bộ** các batch.

---

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `missing ScriptRunContext` | chạy `python label_manual.py` thay vì `streamlit run label_manual.py` |
| `FutureWarning: ByteTrack was deprecated` | chỉ là cảnh báo thư viện supervision, không ảnh hưởng |
| Hỏi `Xóa dữ liệu cũ và chạy lại từ đầu? [y/N]` | extract cùng cam + cùng ngày 2 lần. `y` = làm lại từ đầu, `N` = hủy. Trong dashboard dùng checkbox "Xóa dữ liệu cũ" |
| `Model weights not found: head_detect.pt` | đặt file `head_detect.pt` cạnh `head_detect.py` |
| Bước 1 chạy rất lâu | bình thường — detect chạy trên từng frame; máy không GPU sẽ chậm |
| Track dính 2 người trong 1 crop | do box đầu mở rộng chồng lên người bên cạnh → Bỏ qua / Xóa ở bước 3 |

---

## 7. Train TransReID

Sau bước 4, trỏ config TransReID vào thư mục `myreid/` (cấu trúc giống
hệt Market-1501: `bounding_box_train`, `query`, `bounding_box_test`).
