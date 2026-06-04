
import warnings
import logging
import json
import argparse
import math
import platform
import queue
import re
import select
import struct
import subprocess
import sys
import tempfile
import time
import wave

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
logging.getLogger("openai").setLevel(logging.ERROR)

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from prompts import SYSTEM_PROMPT
from config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    TRANSCRIPTION_MODEL,
    VECTORSTORE_PROVIDER,
    format_openai_error,
    get_openai_api_key,
)

load_dotenv()

OPENAI_API_KEY = get_openai_api_key()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_FOLDER = BASE_DIR / "vectorstore"
VECTOR_METADATA_FILE = VECTOR_FOLDER / "metadata.json"
AUDIO_SAMPLE_RATE = 16000
VOICE_BLOCK_SECONDS = 0.1
VOICE_CALIBRATION_SECONDS = 0.8
VOICE_PRE_ROLL_SECONDS = 0.5
VOICE_MIN_SPEECH_SECONDS = 0.4
VOICE_NO_SPEECH_TIMEOUT = 8.0
VOICE_MIN_RMS_THRESHOLD = 250
VOICE_MIN_PEAK_ABOVE_NOISE = 250
VOICE_SILENCE_ABOVE_NOISE = 180
VOICE_SILENCE_PEAK_RATIO = 0.22
CONTACT_DETAILS = (
    ""
)
SUPPORT_FALLBACK = (
    "I do not have enough information to answer that. For more detailed "
    f"information, please contact ProWork Support at {CONTACT_DETAILS}."
)
PARTIAL_SUPPORT_FALLBACK = (
    f"For the remaining details, please contact ProWork Support at {CONTACT_DETAILS}."
)
OUT_OF_SCOPE_RESPONSE = (
    "I'm designed to help with , so I can't "
    "answer general questions like that. Please ask a defined question, or contact "
    f" at {CONTACT_DETAILS} for more detailed information."
)
PREDEFINED_QUESTIONS = [
    "",
    "
]
MENU_REQUESTS = {
    "hi",
    "hii",
    "hie",
    "hlo",
    "hello",
    "helo",
    "hey",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "start",
    "menu",
    "help",
    "options",
    "see options",
    "see the options",
    "show options",
    "show questions",
    "list options",
    "list questions",
    "predefined questions",
    "suggest questions",
    "suggested questions",
    "what can i ask",
    "what can you do",
}
THANKS_REQUESTS = {
    "thank",
    "thanks",
    "thankies",
    "thank you",
    "thank u",
    "ok thanks",
    "okay thanks",
    "thx",
    "thnx",
    "tnx",
    "ty",
    "thnks",
}
CONVERSATIONAL_MENU_PHRASES = {
    "can you help",
    "could you help",
    "how can you help",
    "how can you help me",
    "how do you help",
    "i need help",
    "need help",
    "tell me what you can do",
}
THANKS_MARKERS = {
    "thank",
    "thanks",
    "thankies",
    "thx",
    "thnx",
    "tnx",
    "ty",
}
INTENT_FILLER_WORDS = {
    "assistant",
    "bot",
    "kindly",
    "maam",
    "madam",
    "mam",
    "please",
    "pls",
    "prowork",
    "sir",
}
USER_MANAGER_SCOPE_TERMS = {
    "",
    "",
    "",
    "",
    "product",
    "contact",
    "",
    "support",
    "address",
    "email",
    "job management",
    "vehicle manager",
    "equipment manager",
    "tm plan",
    "traffic management",
    "qualification manager",
    "toolbox talks",
    "user analytics",
    "asset management",
    "sswp",
    "safe systems",
    "incident reporting",
    "custom inspection",
    "user manager",
    "user management",
    "introduction",
    "introduction to user management",
    "user",
    "users",
    "add user",
    "new user",
    "edit user",
    "add and edit",
    "adding and editing",
    "profile",
    "password",
    "reset password",
    "dashboard",
    "dashboard access",
    "access",
    "access level",
    "access levels",
    "department",
    "departments",
    "add department",
    "area",
    "areas",
    "add area",
    "areas and departments",
    "full user",
    "lite user",
    "unsubscribed",
    "subscription",
    "permission",
    "permissions",
    "nick name",
    "nickname",
    "contact number",
    "first name",
    "last name",
    "suspended user",
}
VALID_INTENTS = {"menu", "thanks", "app_question", "out_of_scope"}
VOICE_STOP_COMMANDS = {"stop", "s", "skip"}

parser = argparse.ArgumentParser(description="User Manager terminal chatbot")
parser.add_argument(
    "--voice",
    action="store_true",
    help="accept spoken questions and read answers aloud on macOS",
)
parser.add_argument(
    "--listen-seconds",
    type=float,
    default=120.0,
    help=(
        "maximum safety seconds for each voice question; recording normally "
        "stops after you pause (default: 120)"
    ),
)
parser.add_argument(
    "--silence-seconds",
    type=float,
    default=1.8,
    help="seconds of silence that ends voice recording (default: 1.8)",
)
args = parser.parse_args()

if args.listen_seconds <= 0:
    parser.error("--listen-seconds must be greater than zero")
if args.silence_seconds <= 0:
    parser.error("--silence-seconds must be greater than zero")

if args.voice and platform.system() != "Darwin":
    raise SystemExit("Voice mode currently requires macOS.")


def calculate_audio_rms(audio_bytes):
    sample_count = len(audio_bytes) // 2
    if sample_count == 0:
        return 0

    samples = struct.unpack(f"<{sample_count}h", audio_bytes[: sample_count * 2])
    mean_square = sum(sample * sample for sample in samples) / sample_count
    return math.sqrt(mean_square)


def read_voice_control_command():
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return ""

    command = normalize_text(sys.stdin.readline())
    if command == "":
        return "finish"
    if command in VOICE_STOP_COMMANDS:
        return "cancel"
    return ""


def collect_voice_audio(audio_queue):
    chunks = []
    pre_speech_buffer = []
    pre_speech_limit = max(1, int(VOICE_PRE_ROLL_SECONDS / VOICE_BLOCK_SECONDS))
    speech_detected = False
    speech_seconds = 0.0
    last_voice_time = None
    peak_time = None
    peak_rms = 0
    started_at = time.monotonic()
    calibration_rms_values = []

    while time.monotonic() - started_at < VOICE_CALIBRATION_SECONDS:
        voice_control = read_voice_control_command()
        if voice_control == "finish":
            return b""
        if voice_control == "cancel":
            return None

        try:
            audio_chunk = audio_queue.get(timeout=VOICE_BLOCK_SECONDS)
        except queue.Empty:
            continue

        now = time.monotonic()
        chunk_rms = calculate_audio_rms(audio_chunk)
        chunks.append(audio_chunk)
        calibration_rms_values.append(chunk_rms)

        if chunk_rms > peak_rms:
            peak_rms = chunk_rms
            peak_time = now

    if calibration_rms_values:
        sorted_rms = sorted(calibration_rms_values)
        noise_floor = sorted_rms[max(0, int(len(sorted_rms) * 0.35) - 1)]
    else:
        noise_floor = VOICE_MIN_RMS_THRESHOLD

    while True:
        now = time.monotonic()
        if now - started_at >= args.listen_seconds:
            return b"".join(chunks) if speech_detected else b""

        voice_control = read_voice_control_command()
        if voice_control == "finish":
            return b"".join(chunks) if speech_detected else b""
        if voice_control == "cancel":
            return None

        try:
            audio_chunk = audio_queue.get(timeout=VOICE_BLOCK_SECONDS)
        except queue.Empty:
            continue

        now = time.monotonic()
        chunk_rms = calculate_audio_rms(audio_chunk)
        chunks.append(audio_chunk)

        pre_speech_buffer.append(audio_chunk)
        if len(pre_speech_buffer) > pre_speech_limit:
            pre_speech_buffer.pop(0)

        if chunk_rms < noise_floor:
            noise_floor = (noise_floor * 0.7) + (chunk_rms * 0.3)
        elif not speech_detected:
            noise_floor = (noise_floor * 0.98) + (chunk_rms * 0.02)

        if chunk_rms > peak_rms:
            peak_rms = chunk_rms
            peak_time = now

        speech_peak_threshold = max(
            VOICE_MIN_RMS_THRESHOLD,
            noise_floor + VOICE_MIN_PEAK_ABOVE_NOISE,
            noise_floor * 1.6,
        )
        if not speech_detected and peak_rms >= speech_peak_threshold:
            speech_detected = True
            speech_seconds = max(speech_seconds, VOICE_MIN_SPEECH_SECONDS)
            last_voice_time = peak_time or now

        if not speech_detected:
            if now - started_at >= VOICE_NO_SPEECH_TIMEOUT:
                return b""
            continue

        silence_threshold = max(
            VOICE_MIN_RMS_THRESHOLD,
            noise_floor + VOICE_SILENCE_ABOVE_NOISE,
            peak_rms * VOICE_SILENCE_PEAK_RATIO,
        )

        if chunk_rms >= silence_threshold:
            speech_seconds += VOICE_BLOCK_SECONDS
            last_voice_time = now
            continue

        if (
            last_voice_time
            and now - last_voice_time >= args.silence_seconds
            and speech_seconds >= VOICE_MIN_SPEECH_SECONDS
        ):
            return b"".join(chunks)


def looks_like_english_transcript(text):
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False

    english_letters = [
        character
        for character in letters
        if "a" <= character.lower() <= "z"
    ]
    return len(english_letters) / len(letters) >= 0.85


def is_english_transcript(text, chat_model):
    if not looks_like_english_transcript(text):
        return False

    prompt = f"""
Return only JSON in this shape:
{{"english": true}}

Is the following message primarily English? Accept short English greetings,
thanks, and commands. Return false for messages that are mainly another
language, even if they contain product names.

Message:
{json.dumps(text)}
"""

    try:
        response = chat_model.invoke(prompt)
        data = extract_json_object(response.content)
        english_value = data.get("english", True)
        if isinstance(english_value, bool):
            return english_value
        return normalize_text(str(english_value)) in {"true", "yes", "english"}
    except Exception:
        return True


def transcribe_english_audio(audio_path):
    transcription_options = {
        "model": TRANSCRIPTION_MODEL,
        "language": "en",
        "prompt": (
            "The speaker is asking an English question for a ProWork support "
            "chatbot. Transcribe English only; do not translate non-English "
            "speech."
        ),
    }

    try:
        with audio_path.open("rb") as recording:
            return OpenAI(api_key=OPENAI_API_KEY).audio.transcriptions.create(
                file=recording,
                **transcription_options,
            )
    except Exception as exc:
        detail = str(exc).lower()
        unsupported_option = (
            "language" in detail
            or "prompt" in detail
            or "unexpected" in detail
            or "unknown parameter" in detail
            or "unsupported" in detail
        )
        if not unsupported_option:
            raise

        with audio_path.open("rb") as recording:
            return OpenAI(api_key=OPENAI_API_KEY).audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=recording,
            )


def listen_for_question(chat_model):
    try:
        import sounddevice as sd
    except ImportError:
        print("\nVoice input: Install voice support with: pip install -r requirements.txt")
        return ""

    audio_file = tempfile.NamedTemporaryFile(
        prefix="prowork-voice-", suffix=".wav", dir="/private/tmp", delete=False
    )
    audio_path = Path(audio_file.name)
    audio_file.close()
    audio_queue = queue.Queue()

    def capture_audio(indata, _frames, _time_info, status):
        if status:
            print(f"\nVoice input warning: {status}")
        audio_queue.put(bytes(indata))

    try:
        print(
            "\nListening now. Speak in English; I will stop after a pause. "
            "Press Enter to finish manually, or type stop to cancel recording."
        )
        blocksize = int(AUDIO_SAMPLE_RATE * VOICE_BLOCK_SECONDS)
        with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=blocksize,
            channels=1,
            dtype="int16",
            callback=capture_audio,
        ):
            audio_bytes = collect_voice_audio(audio_queue)

        if audio_bytes is None:
            print("\nVoice input: Recording cancelled.")
            return ""
        if not audio_bytes:
            print("\nVoice input: I did not hear a question.")
            return ""

        with wave.open(str(audio_path), "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(AUDIO_SAMPLE_RATE)
            recording.writeframes(audio_bytes)

        transcript = transcribe_english_audio(audio_path)
        question = transcript.text.strip()
        if not question:
            print("\nVoice input: I could not understand the question.")
            return ""
        if not is_english_transcript(question, chat_model):
            print("\nVoice input: Please ask your question in English.")
            return ""
        return question
    except Exception as exc:
        detail = str(exc).lower()
        if (
            isinstance(exc, sd.PortAudioError)
            or "permission" in detail
            or "input device" in detail
        ):
            detail = (
                "Could not access the microphone. Allow Microphone access for "
                "Terminal in System Settings > Privacy & Security > Microphone, "
                "then restart the chatbot."
            )
        else:
            detail = format_openai_error(exc)
        print(f"\nVoice input: {detail}")
        return ""
    finally:
        audio_path.unlink(missing_ok=True)


def speak_answer(answer):
    speech_file = tempfile.NamedTemporaryFile(
        prefix="prowork-answer-", suffix=".txt", dir="/private/tmp", delete=False
    )
    speech_path = Path(speech_file.name)
    speech_file.close()

    try:
        speech_path.write_text(answer, encoding="utf-8")
        print("\nVoice: speaking. Press Enter or type stop to stop the voice.")
        process = subprocess.Popen(["/usr/bin/say", "-f", str(speech_path)])

        while process.poll() is None:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue

            command = sys.stdin.readline().strip().lower()
            if command == "" or command in VOICE_STOP_COMMANDS:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                print("Voice: stopped.")
                break
    finally:
        speech_path.unlink(missing_ok=True)


def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def collapse_repeated_letters(text):
    return re.sub(r"([a-z0-9])\1+", r"\1", text)


def remove_intent_fillers(text):
    words = [
        word for word in text.split()
        if word not in INTENT_FILLER_WORDS
    ]
    return " ".join(words)


def intent_variants(text):
    normalized = normalize_text(text)
    without_fillers = remove_intent_fillers(normalized)
    variants = {
        normalized,
        collapse_repeated_letters(normalized),
    }
    if without_fillers:
        variants.add(without_fillers)
        variants.add(collapse_repeated_letters(without_fillers))
    return {variant for variant in variants if variant}


def edit_distance(left, right, max_distance):
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            best_cost = min(insert_cost, delete_cost, replace_cost)
            current.append(best_cost)
            row_minimum = min(row_minimum, best_cost)

        if row_minimum > max_distance:
            return max_distance + 1
        previous = current

    return previous[-1]


def allowed_intent_distance(text):
    if len(text) <= 3:
        return 0
    if len(text) <= 6:
        return 1
    if len(text) <= 14:
        return 2
    return 3


def matches_short_intent(question, accepted_phrases):
    question_variants = intent_variants(question)
    accepted_variants = set()
    for phrase in accepted_phrases:
        accepted_variants.update(intent_variants(phrase))

    if question_variants.intersection(accepted_variants):
        return True

    return any(
        len(question_variant) <= 24
        and edit_distance(
            question_variant,
            accepted_variant,
            allowed_intent_distance(accepted_variant),
        )
        <= allowed_intent_distance(accepted_variant)
        for question_variant in question_variants
        for accepted_variant in accepted_variants
    )


def format_predefined_questions():
    return "\n".join(
        f"{number}. {question}"
        for number, question in enumerate(PREDEFINED_QUESTIONS, start=1)
    )


def build_options_response(greeting=True):
    intro = (
        "Hi! I can help with ProWork and ProWork User Manager."
        if greeting
        else "Here are some ProWork and User Manager questions you can choose from."
    )
    return (
        f"{intro}\n\n"
        "Choose one of these questions:\n"
        f"{format_predefined_questions()}\n\n"
        "Type a number to select a question, or type your own ProWork question."
    )


def resolve_predefined_question(question):
    match = re.fullmatch(r"(?:option|question|q)?\s*(\d{1,2})\.?", question.strip(), re.I)
    if not match:
        return None

    option_number = int(match.group(1))
    if 1 <= option_number <= len(PREDEFINED_QUESTIONS):
        return PREDEFINED_QUESTIONS[option_number - 1]
    return None


def is_menu_request(question):
    normalized = normalize_text(question)
    compact = collapse_repeated_letters(normalized)

    if matches_short_intent(question, MENU_REQUESTS):
        return True

    return any(phrase in compact for phrase in CONVERSATIONAL_MENU_PHRASES)


def is_thanks_request(question):
    normalized = normalize_text(question)
    compact = collapse_repeated_letters(remove_intent_fillers(normalized))
    words = compact.split()

    if matches_short_intent(question, THANKS_REQUESTS):
        return True

    return (
        0 < len(words) <= 5
        and any(
            word in THANKS_MARKERS or word.startswith("thank")
            for word in words
        )
    )


def is_user_manager_question(question):
    normalized = normalize_text(question)
    return any(term in normalized for term in USER_MANAGER_SCOPE_TERMS)


def extract_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def fallback_intent(question):
    if is_menu_request(question):
        return "menu", ""
    if is_thanks_request(question):
        return "thanks", ""
    if is_user_manager_question(question):
        return "app_question", question
    return "out_of_scope", ""


def classify_question_intent(question, chat_model):
    prompt = f"""
You route messages for a ProWork User Manager chatbot.

Classify the user's message into exactly one intent:
- menu: greetings, help requests, capability questions, option/menu requests,
  or phrases like "hello how can you help me".
- thanks: thanks, appreciation, short acknowledgement, or misspelled thanks.
- app_question: any question that appears to be about ProWork User Manager,
  the ProWork product, or ProWork contact details, even with spelling mistakes,
  unclear wording, or different phrasing. This includes what ProWork is, what
  the product does, contact/support/address/email questions, users, adding or
  editing users, User Manager introduction, access levels, dashboard access,
  Full/Lite/Unsubscribed users, passwords, profiles, areas, departments,
  permissions, email, nick name, names, and contact number.
- out_of_scope: clearly unrelated questions such as general knowledge, current
  events, math, coding, personal advice, geography, or anything outside
  ProWork User Manager.

Return only JSON in this exact shape:
{{"intent":"menu|thanks|app_question|out_of_scope","corrected_question":"..."}}

For app_question, corrected_question should be a clean corrected version of the
user's question. For all other intents, corrected_question must be empty.
Do not answer the user.

User message:
{json.dumps(question)}
"""

    response = chat_model.invoke(prompt)
    data = extract_json_object(response.content)

    intent = normalize_text(str(data.get("intent", ""))).replace(" ", "_")
    if intent not in VALID_INTENTS:
        return fallback_intent(question)

    corrected_question = str(data.get("corrected_question", "")).strip()
    if intent == "app_question":
        return intent, corrected_question or question

    return intent, ""


def clean_answer(answer):
    cleaned = answer.strip()

    cleaned = re.sub(
        r"I could not find that information in the uploaded module documents\.?",
        SUPPORT_FALLBACK,
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"I could not find the remaining details in the uploaded module documents\.?",
        PARTIAL_SUPPORT_FALLBACK,
        cleaned,
        flags=re.I,
    )

    source_language_patterns = [
        r"\b(?:the\s+)?uploaded\s+(?:module\s+)?documents?\b",
        r"\b(?:the\s+)?provided\s+context\b",
        r"\b(?:the\s+)?source\s+documents?\b",
        r"\b(?:the\s+)?knowledge\s+base\b",
    ]
    for pattern in source_language_patterns:
        cleaned = re.sub(
            pattern,
            "the available ProWork information",
            cleaned,
            flags=re.I,
        )

    return cleaned


if not (VECTOR_FOLDER / "index.faiss").exists() or not (VECTOR_FOLDER / "index.pkl").exists():
    raise SystemExit("Knowledge base not found. Run: python3 src/ingest.py")

if not VECTOR_METADATA_FILE.exists():
    raise SystemExit(
        "Knowledge base was created with an older setup. Run: python3 src/ingest.py"
    )

metadata = json.loads(VECTOR_METADATA_FILE.read_text())
if (
    metadata.get("provider") != VECTORSTORE_PROVIDER
    or metadata.get("embedding_model") != EMBEDDING_MODEL
):
    raise SystemExit(
        "Knowledge base model does not match this OpenAI setup. Run: python3 src/ingest.py"
    )

print("\nLoading chatbot knowledge...")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

vectorstore = FAISS.load_local(
    str(VECTOR_FOLDER),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

print("\n==============================")
print(" ProWork User Manager Chatbot ")
print("==============================")
if args.voice:
    print(" Voice mode: press Enter to speak, or type a question.")
    print(" While the bot is speaking, press Enter or type stop to stop the voice.")
print(" Type hi, help, or menu to see suggested questions.")

while True:
    if args.voice:
        question = input("\nYou (press Enter to speak): ").strip()
        if not question:
            question = listen_for_question(llm)
            if question:
                print(f"\nYou (voice): {question}")
    else:
        question = input("\nYou: ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye.")
        break

    if args.voice and normalize_text(question) in VOICE_STOP_COMMANDS:
        print("\nBot: Voice is already stopped. Ask your next question when ready.")
        continue

    try:
        selected_question = resolve_predefined_question(question)
        if selected_question:
            question = selected_question
            print(f"\nSelected: {question}")
        elif is_menu_request(question):
            answer = build_options_response()
            print("\nBot:", answer)
            if args.voice:
                speak_answer(answer)
            continue
        elif is_thanks_request(question):
            answer = "You're welcome. Type hi, help, or menu to see suggested ProWork questions."
            print("\nBot:", answer)
            if args.voice:
                speak_answer(answer)
            continue
        else:
            intent, interpreted_question = classify_question_intent(question, llm)
            if intent == "menu":
                answer = build_options_response()
                print("\nBot:", answer)
                if args.voice:
                    speak_answer(answer)
                continue
            if intent == "thanks":
                answer = "You're welcome. Type hi, help, or menu to see suggested ProWork questions."
                print("\nBot:", answer)
                if args.voice:
                    speak_answer(answer)
                continue
            if intent == "out_of_scope":
                answer = f"{OUT_OF_SCOPE_RESPONSE}\n\n{build_options_response(greeting=False)}"
                print("\nBot:", answer)
                if args.voice:
                    speak_answer(answer)
                continue
            question = interpreted_question or question

        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context}


QUESTION:
{question}

ANSWER:
"""
        
        

        response = llm.invoke(final_prompt)

        answer = clean_answer(response.content)

        print("\nBot:", answer)
        if args.voice:
            speak_answer(answer)
    except Exception as exc:
        print(f"\nBot: {format_openai_error(exc)}")
