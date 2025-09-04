# utils/lung_quadrants.py
import cv2
import numpy as np

def _split_lr_kmeans(mask_bin: np.ndarray):
    """
    mask_bin: HxW uint8 (0/255) - 양쪽 폐가 합쳐진 이진 마스크
    return: left_mask, right_mask (둘 다 uint8 0/255)
    """
    h, w = mask_bin.shape
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        z = np.zeros_like(mask_bin)
        return z, z

    # x좌표만으로 K=2 k-means (OpenCV)
    samples = xs.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    # kmeans는 시드에 민감 → 여러 번 시도해도 됨 (여기선 한 번)
    compactness, labels, centers = cv2.kmeans(
        samples, K=2, bestLabels=None, criteria=criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )
    c1, c2 = centers.flatten()
    thresh = (c1 + c2) / 2.0

    X = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    left  = ((mask_bin > 0) & (X <= thresh)).astype(np.uint8) * 255
    right = ((mask_bin > 0) & (X >  thresh)).astype(np.uint8) * 255

    # 한쪽이 비면 중앙선으로 폴백
    if left.sum() == 0 or right.sum() == 0:
        mid = w // 2
        left  = ((mask_bin > 0) & (X <= mid)).astype(np.uint8) * 255
        right = ((mask_bin > 0) & (X >  mid)).astype(np.uint8) * 255
    return left, right

def _split_upper_lower(side_mask, method="kmeans", ratio=0.5):
    h, w = side_mask.shape
    ys, _ = np.where(side_mask > 0)  # xs는 안 쓰므로 버림
    if len(ys) == 0:
        z = np.zeros_like(side_mask); return z, z

    if method == "kmeans":
        samples = ys.astype(np.float32).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, _, centers = cv2.kmeans(   # labels도 안 쓰므로 버림
            samples, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        y_th = int(np.mean(centers))
    else:
        y_min, y_max = ys.min(), ys.max()
        y_th = int(y_min + (y_max - y_min) * ratio)

    Y = np.tile(np.arange(h)[:, None], (1, w))
    upper = ((side_mask > 0) & (Y <= y_th)).astype(np.uint8) * 255
    lower = ((side_mask > 0) & (Y >  y_th)).astype(np.uint8) * 255
    return upper, lower



def build_quadrant_masks(lung_mask_bin: np.ndarray,
                         left_ratio: float = 0.5,
                         right_ratio: float = 0.5):
    """
    lung_mask_bin: HxW uint8 (0/255) - 양쪽 폐 합쳐진 마스크(전처리에서 만든 것)
    left_ratio/right_ratio: 좌/우 각각 위/아래 컷 비율(미세 튜닝용)
    return: dict { "left_upper":mask, "left_lower":..., "right_upper":..., "right_lower":... } (uint8 0/255)
    """
    # 0) 전처리: 작은 구멍 메우고 매끈하게
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(lung_mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 1) 좌/우 분리
    left_mask, right_mask = _split_lr_kmeans(mask)

    # 2) 각 쪽을 위/아래로
    l_up, l_low = _split_upper_lower(left_mask,  left_ratio)
    r_up, r_low = _split_upper_lower(right_mask, right_ratio)

    # 3) 결과 dict
    quads = {
        "left_upper":  l_up,
        "left_lower":  l_low,
        "right_upper": r_up,
        "right_lower": r_low,
    }
    return quads
