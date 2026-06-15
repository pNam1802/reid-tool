def scale_box_down(box, frame_shape, scale_w=2.5, scale_h=2.5, scale_h_up=0.1):
    x1, y1, x2, y2 = box

    h, w = frame_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    # ===== scale chiều ngang (2 bên) =====
    cx = x1 + bw / 2
    new_w = bw * scale_w

    x1_new = int(cx - new_w / 2)
    x2_new = int(cx + new_w / 2)

    # ===== scale chiều dọc =====
    y1_new = int(y1 - bh * scale_h_up)  # kéo lên trên
    y2_new = int(y1 + bh * scale_h)     # kéo xuống dưới

    # ===== clamp =====
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(w, x2_new)
    y2_new = min(h, y2_new)

    return x1_new, y1_new, x2_new, y2_new