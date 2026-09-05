import cv2
import numpy as np
import pytesseract
from pytesseract import Output

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

_easyocr_reader_cache = {}


def _get_easyocr_reader(langs):
    key = tuple(sorted(langs))
    if key not in _easyocr_reader_cache:
        _easyocr_reader_cache[key] = easyocr.Reader(list(langs), gpu=False, verbose=False)
    return _easyocr_reader_cache[key]


def _find_enclosed_regions(img, min_area=1200, max_area_ratio=0.35, close_kernel=5,
                            close_iter=2, min_fill_ratio=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    edges = cv2.dilate(edges, kernel, iterations=close_iter)

    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flooded = inv.copy()
    for x in range(w):
        for y in (0, h - 1):
            if flooded[y, x] == 255:
                cv2.floodFill(flooded, ff_mask, (x, y), 128)
    for y in range(h):
        for x in (0, w - 1):
            if flooded[y, x] == 255:
                cv2.floodFill(flooded, ff_mask, (x, y), 128)

    enclosed = np.where(flooded == 255, 255, 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=4)

    max_area = max_area_ratio * h * w
    regions = []
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        fill_ratio = area / float(bw * bh)
        if fill_ratio < min_fill_ratio:
            continue
        if bw > 0.85 * w and bh > 0.35 * h:
            continue
        mask = (labels == label).astype(np.uint8) * 255
        regions.append({'bbox': (x, y, bw, bh), 'area': area, 'interior_mask': mask})
    return regions


def _solidify(interior_mask):
    contours, _ = cv2.findContours(interior_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return interior_mask.copy()
    largest = max(contours, key=cv2.contourArea)
    full = np.zeros_like(interior_mask)
    cv2.drawContours(full, [largest], -1, 255, -1)
    return full


def _holes_mask(full_mask, interior_mask, bubble_area, min_hole=6, max_hole_ratio=0.6):
    holes = cv2.bitwise_and(full_mask, cv2.bitwise_not(interior_mask))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    out = np.zeros_like(holes)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_hole or area > max_hole_ratio * bubble_area:
            continue
        out[labels == label] = 255
    return out


def _crop_bbox(img, bbox, pad=4):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def _has_text_tesseract(img, bbox, langs, min_conf=35):
    crop, _ = _crop_bbox(img, bbox)
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lang_str = '+'.join(langs)
    try:
        data = pytesseract.image_to_data(gray, lang=lang_str, config='--psm 11', output_type=Output.DICT)
    except pytesseract.TesseractError:
        return False
    for i, txt in enumerate(data['text']):
        conf = float(data['conf'][i]) if data['conf'][i] not in ('-1', '') else -1
        if txt.strip() and conf > min_conf and sum(ch.isalpha() for ch in txt) >= 1:
            return True
    return False


def _has_text_easyocr(img, bbox, langs, min_conf=0.3):
    crop, _ = _crop_bbox(img, bbox)
    if crop.size == 0:
        return False
    reader = _get_easyocr_reader(langs)
    results = reader.readtext(crop)
    for (_, text, conf) in results:
        if text.strip() and conf > min_conf:
            return True
    return False


def _has_text(img, bbox, langs, engine):
    if engine == 'easyocr' and _EASYOCR_AVAILABLE:
        return _has_text_easyocr(img, bbox, langs)
    return _has_text_tesseract(img, bbox, langs)


def clean_balloons(img, langs=('eng',), engine='auto', uniform_std_thresh=14,
                    dilate_holes=1, debug=False):
    if engine == 'auto':
        engine = 'easyocr' if _EASYOCR_AVAILABLE else 'tesseract'

    result = img.copy()
    regions = _find_enclosed_regions(img)
    debug_info = []

    for r in regions:
        interior = r['interior_mask']
        full = _solidify(interior)
        bubble_area = int(np.count_nonzero(full))
        holes = _holes_mask(full, interior, bubble_area)
        if not np.any(holes):
            continue

        if not _has_text(img, r['bbox'], langs, engine):
            continue

        if dilate_holes > 0:
            k = np.ones((3, 3), np.uint8)
            holes = cv2.dilate(holes, k, iterations=dilate_holes)
            holes = cv2.bitwise_and(holes, full)

        interior_pixels = img[interior > 0]
        if len(interior_pixels) < 20:
            continue
        std = float(np.mean(np.std(interior_pixels.astype(np.float32), axis=0)))
        mean_color = interior_pixels.mean(axis=0)

        if std <= uniform_std_thresh:
            result[holes > 0] = mean_color
            mode = 'fill'
        else:
            x, y, w, h = r['bbox']
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            local_mask = holes[y0:y1, x0:x1]
            local_img = result[y0:y1, x0:x1]
            inpainted = cv2.inpaint(local_img, local_mask, 3, cv2.INPAINT_TELEA)
            result[y0:y1, x0:x1] = inpainted
            mode = 'inpaint'

        debug_info.append({'bbox': r['bbox'], 'mode': mode, 'std': std})

    if debug:
        return result, debug_info
    return result
