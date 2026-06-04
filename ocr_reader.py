
import platform
import subprocess
from pathlib import Path

from PIL import Image
import pytesseract
from pytesseract import TesseractNotFoundError


def extract_text_with_macos_vision(image_path):
    helper_path = Path(__file__).with_name("macos_ocr.swift")
    result = subprocess.run(
        [
            "/usr/bin/swift",
            "-module-cache-path",
            "/private/tmp/prowork-swift-module-cache",
            str(helper_path),
            str(image_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Unknown macOS Vision OCR error."
        raise RuntimeError(f"Error reading image {image_path}: {detail}")
    return result.stdout

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except TesseractNotFoundError:
        if platform.system() == "Darwin":
            return extract_text_with_macos_vision(image_path)
        raise RuntimeError(
            "Image OCR requires the Tesseract program. Install it and run ingestion again."
        )
    except Exception as exc:
        raise RuntimeError(f"Error reading image {image_path}: {exc}") from exc
