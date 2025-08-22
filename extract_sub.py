# pip install yt-dlp openai-whisper
# ffmpeg must be installed and on PATH

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tempfile
import shutil
import yt_dlp
import whisper


def get_subtitle_whisper(video_url: str, model_name: str = "small", language: str | None = None) -> str:
    """
    Download audio from a YouTube video and transcribe it with Whisper.

    Args:
        video_url (str): Full YouTube video URL
        model_name (str): Whisper model to use ("tiny", "base", "small", "medium", "large")
        language (str|None): Force language code (e.g., "en"); None = auto-detect

    Returns:
        str: Transcript text
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_whisper_")
    try:
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        ydl_opts = {
            "quiet": True,
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
            ],
        }

        # --- download audio ---
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            base = ydl.prepare_filename(info)

        audio_path = os.path.splitext(base)[0] + ".wav"
        if not os.path.isfile(audio_path):
            # fallback: pick any wav in tempdir
            wavs = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".wav")]
            if not wavs:
                raise RuntimeError("Failed to download audio with yt-dlp")
            audio_path = wavs[0]

        # --- transcribe with Whisper ---
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language=language, fp16=False)
        return result.get("text", "").strip()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


#transcript = get_subtitle_whisper("https://youtu.be/yF6fCCz3IJU?si=yvSSJAhmnYPXwEdB",model_name="base", language="en")
#print(transcript[:500])  # print first 500 chars
