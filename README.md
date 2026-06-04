
# ProWork User Manager Chatbot

AI-powered terminal chatbot for the ProWork User Manager module.

## Features
- PDF ingestion
- Image OCR ingestion
- Notes ingestion
- FAISS vector database
- Terminal chatbot
- Optional macOS voice input and spoken answers
- OpenAI-powered answers
- Modular architecture

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Image OCR uses Tesseract when it is installed. On macOS, the chatbot falls
back to Apple's built-in Vision text recognition when Tesseract is unavailable.
Each ingestion run saves the readable extracted image text to:

```text
data/extracted/image_ocr_text.txt
```

The saved text is included in the chatbot knowledge base, with each image name
shown above the text extracted from it.

---

### 2. Create .env file

Copy:

```bash
.env.example
```

Rename to:

```bash
.env
```

Add your OpenAI API key.

---

### 3. Put your files here

```text
data/pdfs/
data/images/
data/notes/
```

---

### 4. Train chatbot

```bash
python src/ingest.py
```

Run ingestion again whenever you add or replace images so the extracted text
file and chatbot knowledge base are updated.

---

### 5. Start chatbot

```bash
python src/chat.py
```

Type `hi`, `help`, or `menu` to show suggested ProWork questions. You can
enter a listed number or type your own ProWork or User Manager question.

### Voice mode on macOS

To ask questions using the microphone and hear the chatbot answers aloud:

```bash
python src/chat.py --voice
```

In voice mode, press Enter at the prompt and ask your question in English. The
chatbot listens while you are speaking and automatically starts processing
after you pause. If the room is noisy and recording does not finish, press Enter
to finish manually. Type `stop` to cancel recording and return to the prompt.
You can also type a question normally.

While the bot is speaking, press Enter or type `stop` to stop the voice and
return to the next question prompt.

The first voice question may cause macOS to request Microphone permission for
Terminal. Allow it under **System Settings > Privacy & Security > Microphone**
if prompted. Voice input records in Terminal and uses OpenAI transcription;
spoken answers use the built-in `say` command. Spoken questions are sent to
the OpenAI API for transcription.

If the bot can speak answers but cannot hear your question, check that
Terminal is enabled in the **Microphone** privacy section, then restart voice
mode. Speech Recognition permission is not required.

---

## Example Questions

- How do I add a new user?
- What is dashboard access?
- What is Lite User?
- How do I reset a password?
- How do access levels work?

---

## Exit chatbot

Type:

```text
exit
```
