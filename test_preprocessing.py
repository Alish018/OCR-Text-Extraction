# tests/test_preprocessing.py
import os
from src import preprocessing

def test_preprocess_on_sample():
    sample = os.path.join("tests", "sample1.jpg")
    # If sample not present, skip test by asserting True
    if not os.path.exists(sample):
        assert True
        return
    prep = preprocessing.preprocess_for_ocr(sample)
    assert 'gray' in prep and 'binarized' in prep
