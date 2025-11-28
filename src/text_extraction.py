import re
import os
import cv2
from pathlib import Path

from src import preprocessing, ocr_engine

PATTERN_UNDERSCORE = re.compile(r'\b[0-9A-Za-z]{3,}_1_[0-9A-Za-z]{2,}\b')

PATTERN_FALLBACK = re.compile(r'\b[0-9A-Za-z_]{6,}\b')

def _clean_text(s):
    """Convert OCR text to single-line, trimmed text."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


def _repair_common_confusions(tok):
    if not tok:
        return tok

    t = tok.replace('-', '_')
    t = t.replace('|', '1').replace(' ', '')

    # If token contains mix of digits & letters or many digits, fix confusions
    digit_frac = sum(ch.isdigit() for ch in t) / max(1, len(t))
    if digit_frac > 0.2:
        t = t.replace('O', '0').replace('o', '0')
        t = t.replace('l', '1').replace('I', '1')

    return t


def _final_clean_token(tok):
    if not tok:
        return ""

    clean = re.sub(r'[^A-Za-z0-9_]', '', tok)
    if not clean:
        return ""

    if '_1_' in clean:
        return clean

    if '1' in clean and len(clean) >= 6:
        idxs = [i for i, c in enumerate(clean) if c == '1']
        if idxs:
            mid = idxs[len(idxs) // 2]
            repaired = clean[:mid] + '_1_' + clean[mid + 1:]
            if re.match(r'^[0-9A-Za-z]{2,}_1_[0-9A-Za-z]{2,}$', repaired):
                return repaired

    return clean

def _token_score(token, conf=30.0, source_votes=0):
    t = token or ""
    if not t:
        return -9999

    score = 0.0
    score += float(conf) * 0.6
    score += min(len(t), 60) * 0.5

    digit_frac = sum(ch.isdigit() for ch in t) / max(1, len(t))
    score += digit_frac * 30.0

    if '_1_' in t:
        score += 200.0        
    score += source_votes * 80.0
    return score
def _merge_ocr_results(tess_text, tess_conf, easy_text, easy_conf):
    candidates = {}
    def add_candidate(text, conf, engine_name):
        if text is None:
            return
        t = _clean_text(text)
        if not t:
            return
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = 30.0
        parts = [p.strip() for p in re.split(r'\s+', t) if len(p.strip()) >= 3]
        parts.insert(0, t.strip())

        for p in parts:
            if not p:
                continue
            repaired = _repair_common_confusions(p)
            repaired = repaired.strip()
            if not repaired:
                continue

            entry = candidates.get(repaired, {"text": repaired, "confs": [], "engines": set()})
            entry["confs"].append(conf_val)
            entry["engines"].add(engine_name)
            candidates[repaired] = entry

    add_candidate(tess_text, tess_conf, "tesseract")
    add_candidate(easy_text, easy_conf, "easyocr")
    out = []
    for k, v in candidates.items():
        avg_conf = sum(v["confs"]) / max(1, len(v["confs"]))
        votes = len(v["engines"])
        out.append({"text": k, "conf": avg_conf, "votes": votes})

    return out
def extract_target_line_from_image(image_path):
    p = Path(image_path)
    if not p.exists():
        return "", {"method": "file_missing", "confidence": 0}

    prep = preprocessing.preprocess_for_ocr(str(p))
    gray = prep.get("gray")
    bin_img = prep.get("binarized")
    try:
        tess_text, tess_conf = ocr_engine.ocr_tesseract(gray, psm=6)
    except Exception:
        tess_text, tess_conf = "", 20.0

    try:
        easy_text, easy_conf = ocr_engine.ocr_easyocr(gray)
    except Exception:
        easy_text, easy_conf = "", 20.0

    tess_text = _clean_text(tess_text)
    easy_text = _clean_text(easy_text)
    
    for raw, conf in [(tess_text, tess_conf), (easy_text, easy_conf)]:
        m = PATTERN_UNDERSCORE.search(raw)
        if m:
            tok = m.group(0)
            tok = _final_clean_token(_repair_common_confusions(tok))
            return tok, {"method": "pattern_fulltext_direct", "confidence": float(conf)}

    candidates = _merge_ocr_results(tess_text, tess_conf, easy_text, easy_conf)

    strict = [c for c in candidates if PATTERN_UNDERSCORE.search(c["text"])]
    if strict:
        best = max(strict, key=lambda c: _token_score(c["text"], c["conf"], c["votes"]))
        cleaned = _final_clean_token(best["text"])
        return cleaned, {"method": "pattern_fulltext_candidate", "confidence": best["conf"]}
    crop_candidates = []
    east_model = os.path.join(os.getcwd(), "models", "frozen_east_text_detection.pb")
    if os.path.exists(east_model):
        try:
            orig_img = prep.get("original") 
            boxes = text_detector.detect_text_east(orig_img, east_model, score_threshold=0.5, nms_threshold=0.4, width=640, height=640)
        except Exception:
            boxes = []
    else:
        boxes = []
    if boxes:
        for (x, y, w, h) in boxes:
           
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(orig_img.shape[1], x + w + pad)
            y1 = min(orig_img.shape[0], y + h + pad)
            crop = cv2.cvtColor(orig_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if len(orig_img.shape) == 3 else orig_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if crop.shape[1] < 200:
                crop = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            try:
                t_t, c_t = ocr_engine.ocr_tesseract(crop, psm=7)
            except Exception:
                t_t, c_t = "", 20.0
            try:
                t_e, c_e = ocr_engine.ocr_easyocr(crop)
            except Exception:
                t_e, c_e = "", 20.0
            merged = _merge_ocr_results(t_t, c_t, t_e, c_e)
            crop_candidates.extend(merged)
            
            for m in merged:
                if PATTERN_UNDERSCORE.search(m["text"]):
                    cleaned = _final_clean_token(_repair_common_confusions(m["text"]))
                    return cleaned, {"method": "pattern_in_east_crop_direct", "confidence": m["conf"]}
    else:
        if bin_img is not None and gray is not None:
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:80]
            H, W = gray.shape[:2]
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if w < 30 or h < 8:
                    continue
                pad = 6
                y0 = max(0, y - pad)
                y1 = min(H, y + h + pad)
                x0 = max(0, x - pad)
                x1 = min(W, x + w + pad)
                if y1 <= y0 or x1 <= x0:
                    continue
                crop = gray[y0:y1, x0:x1]
                if crop.shape[1] < 200:
                    crop = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                try:
                    t_t, c_t = ocr_engine.ocr_tesseract(crop, psm=7)
                except Exception:
                    t_t, c_t = "", 20.0
                try:
                    t_e, c_e = ocr_engine.ocr_easyocr(crop)
                except Exception:
                    t_e, c_e = "", 20.0
                merged = _merge_ocr_results(t_t, c_t, t_e, c_e)
                crop_candidates.extend(merged)
                for m in merged:
                    if PATTERN_UNDERSCORE.search(m["text"]):
                        cleaned = _final_clean_token(_repair_common_confusions(m["text"]))
                        return cleaned, {"method": "pattern_in_contour_crop_direct", "confidence": m["conf"]}

    combined = {}
    for c in candidates:
        combined[c["text"]] = {"text": c["text"], "conf": c["conf"], "votes": c["votes"]}

    for c in crop_candidates:
        if c["text"] in combined:
            existing = combined[c["text"]]
            existing["conf"] = (existing["conf"] + c["conf"]) / 2
            existing["votes"] = max(existing["votes"], c["votes"])
        else:
            combined[c["text"]] = {"text": c["text"], "conf": c["conf"], "votes": c["votes"]}

    all_candidates = list(combined.values())
    if not all_candidates:
        return "", {"method": "no_candidates", "confidence": 0}

    scored = []
    for c in all_candidates:
        cleaned = _final_clean_token(_repair_common_confusions(c["text"]))
        sc = _token_score(cleaned, c["conf"], c["votes"])

        if re.search(r'[A-Za-z]', cleaned) and re.search(r'\d', cleaned):
            sc += 5  # small bonus

        scored.append((cleaned, sc, c["conf"], c["votes"]))

    best_txt, best_score, best_conf, best_votes = max(scored, key=lambda x: x[1])

    if '_1_' not in best_txt:
        fallback = [s for s in scored if '1' in s[0] and len(s[0]) >= 6]
        if fallback:
            best_txt = max(fallback, key=lambda x: x[1])[0]
    def _trim_around_1(token, max_len=50):
        if not token or '_1_' not in token:
            return token
        i = token.find('_1_')
        left = i - 1
        while left >= 0 and (token[left].isalnum() or token[left] == '_') and (i - left) <= max_len:
            left -= 1
        left += 1
        right = i + 3
        while right < len(token) and (token[right].isalnum() or token[right] == '_') and (right - i) <= max_len:
            right += 1

        trimmed = token[left:right]
        return trimmed if len(trimmed) >= 6 else token

    best_txt = _trim_around_1(best_txt)
    best_txt = _trim_around_1(best_txt, max_len=40)
    best_txt = _final_clean_token(best_txt)

    if '_1_' not in best_txt and len(best_txt) < 6:
        return "", {"method": "low_conf_no_pattern", "confidence": float(best_conf)}

    return best_txt, {
        "method": "best_candidate",
        "confidence": float(best_conf),
        "votes": int(best_votes)
    }

