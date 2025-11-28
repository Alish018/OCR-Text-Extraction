import pytesseract
import easyocr
import numpy as np
import cv2

_reader = None
def get_easyocr_reader(langs=["en"]):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(langs, gpu=False)
    return _reader

def ocr_tesseract(image, psm=6, whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"):
    import pytesseract

    config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist={whitelist}'

    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        text = " ".join([w for w in data["text"] if w.strip()])
        confs = [float(c) for c in data["conf"] if c.isdigit()]
        avg_conf = sum(confs)/len(confs) if confs else 40.0
        return text, avg_conf
    except Exception:
        return "", 20.0

def ocr_easyocr(image):
        
    if len(image.shape) == 2:
        img_in = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reader = get_easyocr_reader()
    try:
        results = reader.readtext(img_in)
    except Exception:
        results = []

    if not results:
        return "", 25.0

    texts = [r[1] for r in results]
    confs = [r[2] for r in results]
    combined = " ".join(texts)
    avg_conf = float(np.mean(confs)) * 100.0  # scale to 0-100 like pytesseract
    return (combined.strip(), avg_conf)

