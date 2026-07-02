import os
import uuid
import asyncio
import subprocess
import tempfile
import wave
import struct
import traceback
import threading
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
import whisper
import edge_tts

app = Flask(__name__)

# ==================== تنظیمات ====================
GAPGPT_KEY = "sk-aWI1qROVYtcrMngmiMcfbdFtiGF24tY0Dc952nb8c7bQPGQY"
chat_client = OpenAI(api_key=GAPGPT_KEY, base_url="https://api.gapgpt.app/v1")
CHAT_MODEL = "gapgpt-qwen-3.5"
whisper_model = whisper.load_model("small")

UPLOAD_FOLDER = "received_audio"
RESPONSE_FOLDER = "response_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESPONSE_FOLDER, exist_ok=True)

tasks = {}
task_counter = 0
task_lock = threading.Lock()

# ==================== توابع کمکی ====================
def boost_wav(input_path, output_path, gain_db=6.0):
    with wave.open(input_path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    fmt = f"<{len(frames)//2}h"
    samples = list(struct.unpack(fmt, frames))
    if samples:
        offset = int(sum(samples) / len(samples))
        samples = [s - offset for s in samples]
    gain = 10 ** (gain_db / 20.0)
    samples = [max(min(int(s * gain), 32767), -32768) for s in samples]
    with wave.open(output_path, "w") as wf:
        wf.setparams(params)
        wf.writeframes(struct.pack(fmt, *samples))

async def tts_edge(text, mp3_path):
    comm = edge_tts.Communicate(text, "fa-IR-FaridNeural", rate="+0%")
    await comm.save(mp3_path)

def text_to_wav_file(text, wav_path):
    try:
        text = ' '.join(text.split())
        words = text.split()
        if len(words) > 1 and words[-1] == words[-2]:
            text = ' '.join(words[:-1])
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        asyncio.run(tts_edge(text, mp3_path))
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_path
        ], check=True, capture_output=True)
        os.remove(mp3_path)
        return True
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False

def process_task(task_id, in_path):
    try:
        tasks[task_id]["status"] = "processing"
        boosted = os.path.join(UPLOAD_FOLDER, f"boost_{uuid.uuid4().hex}.wav")
        boost_wav(in_path, boosted)
        result = whisper_model.transcribe(boosted, language="fa")
        os.remove(boosted)
        user_text = result["text"].strip()
        print(f"🗣️ کاربر: {user_text}")
        if not user_text:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["text"] = "متوجه نشدم"
            os.remove(in_path)
            return
        resp = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "دستیار فارسی‌زبان مفید و کوتاه."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7, max_tokens=300
        )
        answer = resp.choices[0].message.content.strip()
        print(f"🤖 پاسخ: {answer}")
        out_file = f"resp_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(RESPONSE_FOLDER, out_file)
        if not text_to_wav_file(answer, out_path):
            tasks[task_id]["status"] = "error"
            tasks[task_id]["text"] = "خطا در تولید صدا"
            os.remove(in_path)
            return
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["text"] = answer
        tasks[task_id]["audio_url"] = f"/audio/{out_file}"
        os.remove(in_path)
    except Exception as e:
        print(f"💥 خطا: {e}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["text"] = f"خطا: {str(e)}"

# ==================== مسیرها ====================
@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

@app.route('/upload', methods=['POST'])
def upload():
    global task_counter
    if request.content_type != 'audio/wav':
        return jsonify({"error": "wav required"}), 400
    in_file = f"in_{uuid.uuid4().hex}.wav"
    in_path = os.path.join(UPLOAD_FOLDER, in_file)
    with open(in_path, "wb") as f:
        f.write(request.data)
    with task_lock:
        task_counter += 1
        task_id = str(task_counter)
    tasks[task_id] = {"status": "pending", "text": None, "audio_url": None}
    thread = threading.Thread(target=process_task, args=(task_id, in_path))
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id, "status": "pending"})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "task not found"}), 404
    task = tasks[task_id]
    return jsonify({
        "status": task["status"],
        "text": task.get("text"),
        "audio_url": task.get("audio_url")
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    path = os.path.join(RESPONSE_FOLDER, filename)
    if not os.path.exists(path):
        return "فایل یافت نشد", 404
    return send_file(path, mimetype='audio/wav')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
