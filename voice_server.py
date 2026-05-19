"""
Clara Voice Server
Receives audio from echo_controller via WebSocket,
transcribes with Whisper, calls Clara via HTTP, returns TTS audio.
"""

import asyncio
import io
import logging
import numpy as np
import os
import struct
import wave

import httpx
import websockets
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("clara-voice")

# ── Config ────────────────────────────────────────────────────────────

WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
PIPER_MODEL    = os.environ.get("PIPER_MODEL", "/piper_models/en_GB-alba-medium.onnx")
CLARA_API      = os.environ.get("CLARA_API", "http://clara-bot:8766/message")
VOICE_USER     = os.environ.get("VOICE_USER", "your_username")
VOICE_CHAT_ID  = int(os.environ.get("VOICE_CHAT_ID", "0"))
WS_PORT        = int(os.environ.get("WS_PORT", "8765"))

SAMPLE_RATE      = 16000
SILENCE_THRESHOLD = 30.0   # absolute RMS — below this is silence
SPEECH_THRESHOLD  = 30.0   # peak RMS must exceed this for audio to be worth transcribing
SILENCE_CHUNKS   = 12      # consecutive silent chunks before stopping (~2.7s at 80ms/chunk)
MAX_CHUNKS       = 125     # hard cap — ~10 seconds of audio

# ── Whisper ───────────────────────────────────────────────────────────

log.info(f"Loading Whisper model {WHISPER_MODEL} on {WHISPER_DEVICE}...")
whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="float16")

# Warm up CUDA kernels — first inference is always slow due to JIT compilation
_dummy_buf = io.BytesIO()
with wave.open(_dummy_buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes((np.zeros(SAMPLE_RATE, dtype=np.int16)).tobytes())
_dummy_buf.seek(0)
list(whisper.transcribe(_dummy_buf, language="en")[0])
log.info("Whisper ready")


# ── Piper TTS ─────────────────────────────────────────────────────────

async def synthesise(text: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "piper",
        "--model", PIPER_MODEL,
        "--output_raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate(input=text.encode())
    return stdout


# ── Audio helpers ─────────────────────────────────────────────────────

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def chunk_rms(pcm_chunk: bytes) -> float:
    """Return absolute RMS of a mono S16_LE chunk."""
    if len(pcm_chunk) < 2:
        return 0.0
    samples = struct.unpack(f"{len(pcm_chunk)//2}h", pcm_chunk)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def is_silent(pcm_chunk: bytes) -> bool:
    return chunk_rms(pcm_chunk) < SILENCE_THRESHOLD


# ── Clara API ─────────────────────────────────────────────────────────

async def ask_clara(text: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(CLARA_API, json={
            "text": text,
            "username": VOICE_USER,
            "chat_id": VOICE_CHAT_ID,
        })
        r.raise_for_status()
        return r.json()["response"]


# ── WebSocket handler ─────────────────────────────────────────────────

async def handle_client(websocket):
    client = websocket.remote_address
    log.info(f"Client connected: {client}")

    audio_chunks = []
    recording    = False
    silence_count = 0
    peak_rms     = 0.0

    def reset():
        nonlocal audio_chunks, recording, silence_count, peak_rms
        audio_chunks  = []
        recording     = False
        silence_count = 0
        peak_rms      = 0.0

    def maybe_process():
        """Process if peak RMS indicates real speech, otherwise discard."""
        if peak_rms >= SPEECH_THRESHOLD:
            asyncio.create_task(process_and_respond(websocket, audio_chunks))
        else:
            log.info(
                f"Peak RMS {peak_rms:.1f} below speech threshold {SPEECH_THRESHOLD} "
                f"— discarding (likely silence or noise after wake word)"
            )

    try:
        async for message in websocket:
            if isinstance(message, str):
                if message == "START":
                    log.info("Recording started")
                    reset()
                    recording = True
                elif message == "END":
                    log.info(f"VAD end signal — processing (peak RMS {peak_rms:.1f})")
                    await websocket.send("THINKING")
                    maybe_process()
                    reset()

            elif isinstance(message, bytes) and recording:
                audio_chunks.append(message)
                rms      = chunk_rms(message)
                peak_rms = max(peak_rms, rms)

                # Hard cap — don't record forever
                if len(audio_chunks) >= MAX_CHUNKS:
                    log.info(
                        f"Max recording length reached ({MAX_CHUNKS} chunks) "
                        f"— processing (peak RMS {peak_rms:.1f})"
                    )
                    await websocket.send("THINKING")
                    maybe_process()
                    reset()
                    continue

                if is_silent(message):
                    silence_count += 1
                    if silence_count >= SILENCE_CHUNKS:
                        log.info(
                            f"Silence detected after {len(audio_chunks)} chunks "
                            f"(peak RMS {peak_rms:.1f})"
                        )
                        await websocket.send("THINKING")
                        maybe_process()
                        reset()
                else:
                    silence_count = 0

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Client disconnected: {client}")
    except Exception as e:
        log.error(f"Handler error: {e}")


async def process_and_respond(websocket, chunks: list[bytes]):
    try:
        pcm = b"".join(chunks)
        wav = pcm_to_wav(pcm)

        log.info(f"Transcribing {len(pcm)//2/SAMPLE_RATE:.1f}s of audio...")
        loop = asyncio.get_event_loop()
        wav_io = io.BytesIO(wav)
        segments, _ = await loop.run_in_executor(
            None, lambda: whisper.transcribe(wav_io, language="en")
        )
        text = " ".join(s.text for s in segments).strip()
        log.info(f"Transcribed: '{text}'")

        if not text:
            log.info("Empty transcription — ignoring")
            return

        log.info("Asking Clara...")
        response = await ask_clara(text)
        log.info(f"Clara: '{response}'")

        log.info("Synthesising...")
        audio = await synthesise(response)

        await websocket.send(audio)
        log.info(f"Sent {len(audio)} bytes of audio")

    except Exception as e:
        log.error(f"Process error: {e}")


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    log.info(f"Clara Voice Server starting on port {WS_PORT}")
    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        log.info("Ready")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())