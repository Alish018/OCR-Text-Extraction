This project implements an end-to-end OCR system designed to extract a specific text line containing the pattern _1_ from shipping label / waybill images.
The goal aligns with the assignment requirement of building a production-ready OCR pipeline capable of achieving ≥75% accuracy on the target field.

The solution includes:

A full preprocessing pipeline

Combined OCR engines (Tesseract + EasyOCR)

Text detection (Contours / EAST)

A robust text-extraction logic specially optimized for _1_ pattern

A Streamlit web interface for demonstration

Evaluation script & accuracy reporting

This repository shows my complete approach, code structure, and accuracy evaluation methodology.
