# OCR_Task — Extract `_1_` token

Simple beginner-friendly OCR project that extracts the token containing `_1_` from label images.

## How to use (step-by-step)

1. Put all your images into the `images/` folder.

2. Install Tesseract system binary:
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Windows: install from UB-Mannheim or Tesseract project and ensure it's in PATH.

3. Create Python env and install packages:


4. Run the Streamlit demo:
5. To evaluate against ground truth:
- Edit `tests/ground_truth.csv` with rows: `filename,expected`
- Run:
  ```
  python scripts/evaluate.py --images_folder images --gt tests/ground_truth.csv
  ```
- Results saved at `results/eval_results.csv`.