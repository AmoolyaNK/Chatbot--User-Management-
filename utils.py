
from pathlib import Path

def load_text_notes(notes_folder):
    text = ""
    notes_path = Path(notes_folder)

    for file in notes_path.glob("*.txt"):
        try:
            text += file.read_text(encoding="utf-8") + "\n"
        except Exception:
            pass

    return text
