
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from pdf_reader import extract_text_from_pdf
from ocr_reader import extract_text_from_image
from utils import load_text_notes
from config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PROVIDER,
    format_openai_error,
    get_openai_api_key,
)

load_dotenv()

OPENAI_API_KEY = get_openai_api_key()

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FOLDER = BASE_DIR / "data" / "pdfs"
IMAGE_FOLDER = BASE_DIR / "data" / "images"
NOTES_FOLDER = BASE_DIR / "data" / "notes"
EXTRACTED_FOLDER = BASE_DIR / "data" / "extracted"
IMAGE_TEXT_FILE = EXTRACTED_FOLDER / "image_ocr_text.txt"
VECTOR_FOLDER = BASE_DIR / "vectorstore"
VECTOR_METADATA_FILE = VECTOR_FOLDER / "metadata.json"

all_text = ""

print("\nReading PDFs...")

for pdf_file in PDF_FOLDER.glob("*.pdf"):
    print(f"Reading: {pdf_file.name}")
    all_text += extract_text_from_pdf(pdf_file) + "\n"

print("\nReading Images...")

image_sections = []

for image_file in sorted(IMAGE_FOLDER.glob("*")):
    if image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        print(f"Reading: {image_file.name}")
        image_text = extract_text_from_image(image_file)
        print(f"Extracted {len(image_text.strip())} image text characters")
        image_sections.append(
            f"IMAGE SOURCE: {image_file.name}\n\n{image_text.strip()}\n"
        )

EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)
extracted_image_text = "\n\n".join(image_sections)
IMAGE_TEXT_FILE.write_text(extracted_image_text + "\n", encoding="utf-8")
print(f"Saved extracted image text: {IMAGE_TEXT_FILE}")

all_text += extracted_image_text + "\n"

print("\nReading Notes...")

all_text += load_text_notes(NOTES_FOLDER)

if not all_text.strip():
    raise Exception("No text found. Add PDFs/images/notes first.")

print("\nSplitting text into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.create_documents([all_text])

print(f"Created {len(docs)} chunks")

print("\nCreating embeddings...")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

try:
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception as exc:
    print(f"\nERROR: {format_openai_error(exc)}")
    raise SystemExit(1) from exc

vectorstore.save_local(str(VECTOR_FOLDER))
VECTOR_METADATA_FILE.write_text(
    json.dumps(
        {
            "provider": VECTORSTORE_PROVIDER,
            "embedding_model": EMBEDDING_MODEL,
        },
        indent=2,
    )
)

print("\nSUCCESS: Knowledge base created.")
print("You can now run: python3 src/chat.py")
