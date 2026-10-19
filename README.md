# Question Studio – Bloom's-taxonomy question assistant

A chatbot that predicts the Bloom's level of exam questions, lets you **upgrade or degrade** them, **scans question papers (OCR)**,
and **writes questions from your own slides / notes / PDFs** in the format you ask for.

## Run
```bash
pip install -r requirements.txt
cd backend
python ml.py        # trains the Bloom-level and difficulty models from data/Dataset.xlsx
python app.py       # http://localhost:8000
```

## OCR (needed only for scanned papers and photos)
Install ONE of these, then restart the server:
* `pip install rapidocr-onnxruntime`   (easiest, pip only)
* Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) + `pip install pytesseract`
* or put a `GEMINI_API_KEY` in `.env` (Gemini reads the image)

## Optional AI mode
Copy `.env.example` to `.env` and add `GEMINI_API_KEY` (free key: https://aistudio.google.com/apikey).
Without a key everything still works offline; with a key, questions written from documents use the document's real facts and include model answers.

## What it does
| Feature | How |
|---|---|
| Interactive chat | Asks follow-up questions (topic, level, difficulty, count, type), remembers context ("more", "harder", "change level to Apply") |
| Predict Bloom level | Model trained on the 1,296-question dataset (about 87% accuracy in 5-fold cross-validation) |
| Upgrade / degrade | Rewrites a question one level up or down (or to a level you choose) |
| Question-paper analysis | PDF, Word, PowerPoint or a photo; OCR for scans; level + difficulty per question, marks and [CO/BL] tags understood, Excel/CSV export |
| Questions from documents | PPTX / DOCX / PDF / TXT / images; choose number, Bloom levels, type (short, long, MCQ, mixed), difficulty, marks; export to Word or Excel |

Legacy `.doc` / `.ppt` files must be re-saved as `.docx` / `.pptx`.
