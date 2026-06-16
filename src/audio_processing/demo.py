import os
# 重定向 HuggingFace 缓存目录到 E 盘，避免占用 C 盘空间
# Must be set BEFORE any speechbrain/huggingface imports
os.environ["HF_HOME"] = "E:/HuggingFaceCache"

import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from pydub import AudioSegment
import requests
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
from dotenv import load_dotenv
from pathlib import Path

# find config/.env
env_path = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)


def load_audio_as_tensor(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load any audio file (m4a, mp3, wav, etc.) as a 16kHz mono torch.Tensor.
    
    Strategy:
    1. Use pydub (calls ffmpeg.exe) to decode and resample any format → avoids torchcodec DLL issues.
    2. Export to a temporary in-memory wav buffer.
    3. Read the wav buffer with soundfile (libsndfile) → rock-solid on Windows.
    4. Convert numpy array to torch.Tensor.
    
    This completely bypasses torchaudio and torchcodec.
    """
    # Step 1: Use pydub to convert and resample to target_sr
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)  # mono, 16kHz

    # Step 2: Export to a temp WAV file (soundfile needs a file, not a buffer)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio.export(tmp_path, format="wav")
        # Step 3: Read with soundfile (no FFmpeg, no torchcodec, just libsndfile)
        samples, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
    finally:
        os.remove(tmp_path)

    # Step 4: numpy → torch Tensor, shape [time]
    return torch.from_numpy(samples)


def main():
    parser = argparse.ArgumentParser(description="Omi-Style Audio Processing Prototype")
    parser.add_argument("--profile", required=True, help="Path to your voice profile audio (e.g. myvoice.m4a)")
    parser.add_argument("--name",    required=True, help="Your name (e.g. 'Hongye')")
    parser.add_argument("--meeting", required=True, help="Path to the meeting audio to process (e.g. meeting.mp3)")
    args = parser.parse_args()

    # 0. Check API Key
    dg_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not dg_api_key:
        print("Error: Please set DEEPGRAM_API_KEY in config/.env")
        return

    # ------------------------------------------------------------------ #
    # Step 1: Load the SpeechBrain ECAPA-TDNN speaker embedding model     #
    # ------------------------------------------------------------------ #
    print("Step 1: Loading SpeechBrain speaker embedding model...")
    print("        (First run will download ~80 MB to E:/HuggingFaceCache)")
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_model",
        local_strategy=LocalStrategy.COPY   # Required on Windows without admin rights
    )
    print("        Model loaded ✅\n")

    # ------------------------------------------------------------------ #
    # Step 2: Pre-compute the embedding for the user's voice profile      #
    # ------------------------------------------------------------------ #
    print(f"Step 2: Extracting voice embedding for '{args.name}' from '{args.profile}'...")
    wav_profile = load_audio_as_tensor(args.profile)             # [time]
    wav_profile_batch = wav_profile.unsqueeze(0)                  # [1, time]
    emb_profile = model.encode_batch(wav_profile_batch)           # [1, 1, emb_dim]
    emb_profile = emb_profile.squeeze()                           # [emb_dim]
    emb_profile = F.normalize(emb_profile, dim=0)
    print("        Done ✅\n")

    # ------------------------------------------------------------------ #
    # Step 3: Send meeting audio to Deepgram for transcription+diarization#
    # ------------------------------------------------------------------ #
    print(f"Step 3: Sending '{args.meeting}' to Deepgram for transcription + diarization...")
    url = "https://api.deepgram.com/v1/listen?diarize=true&model=nova-2&smart_format=true"
    headers = {"Authorization": f"Token {dg_api_key}"}
    with open(args.meeting, "rb") as f:
        res = requests.post(url, headers=headers, data=f)

    if res.status_code != 200:
        print(f"Deepgram API Error {res.status_code}: {res.text}")
        return

    words = (res.json()
             .get("results", {})
             .get("channels", [{}])[0]
             .get("alternatives", [{}])[0]
             .get("words", []))
    if not words:
        print("No words returned from Deepgram.")
        return
    print("        Done ✅\n")

    # ------------------------------------------------------------------ #
    # Step 4: Build speaker segments (first occurrence of each speaker)   #
    # ------------------------------------------------------------------ #
    print("Step 4: Building speaker segments from diarization output...")
    speaker_segments = {}
    for word in words:
        spk = word.get("speaker", 0)
        if spk not in speaker_segments:
            # Record the start time of this speaker's FIRST segment
            speaker_segments[spk] = {"start": word.get("start", 0), "end": word.get("end", 0)}
        else:
            speaker_segments[spk]["end"] = word.get("end", 0)

    print(f"        Found {len(speaker_segments)} speaker(s): {list(speaker_segments.keys())}\n")

    # ------------------------------------------------------------------ #
    # Step 5: Match each speaker against the user's voice profile         #
    # ------------------------------------------------------------------ #
    print("Step 5: Matching each speaker against your voice profile...")
    audio_full = AudioSegment.from_file(args.meeting)

    speaker_map = {}
    for spk, times in speaker_segments.items():
        duration_s = times["end"] - times["start"]
        if duration_s < 1.0:
            print(f"  [Speaker {spk}] Segment too short ({duration_s:.1f}s), skipping.")
            speaker_map[spk] = f"Speaker {spk}"
            continue

        # Grab up to 5 seconds from this speaker's first segment
        start_ms = int(times["start"] * 1000)
        end_ms   = int(min(times["start"] + 5.0, times["end"]) * 1000)
        chunk    = audio_full[start_ms:end_ms]

        # Export chunk as a temporary WAV, then load via soundfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = tmp.name
        try:
            chunk.export(chunk_path, format="wav")
            wav_chunk = load_audio_as_tensor(chunk_path)           # [time]
        finally:
            os.remove(chunk_path)

        wav_chunk_batch = wav_chunk.unsqueeze(0)                    # [1, time]
        emb_chunk = model.encode_batch(wav_chunk_batch).squeeze()   # [emb_dim]
        emb_chunk = F.normalize(emb_chunk, dim=0)

        # Cosine similarity (both vectors are already unit-normalized)
        score_val = torch.dot(emb_chunk, emb_profile).item()

        print(f"  [Speaker {spk}] Cosine similarity score: {score_val:.4f}")
        if score_val > 0.3:
            print(f"  ✅  Speaker {spk} identified as: {args.name}")
            speaker_map[spk] = args.name
        else:
            print(f"  ❌  Speaker {spk} is NOT {args.name}")
            speaker_map[spk] = f"Speaker {spk}"

    # ------------------------------------------------------------------ #
    # Final output                                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "="*50)
    print("FINAL SPEAKER MAPPING (Ready for RAG system):")
    for spk, name in speaker_map.items():
        print(f"  Speaker {spk}  →  {name}")
    print("="*50)


if __name__ == "__main__":
    main()
