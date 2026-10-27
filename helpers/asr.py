#!/usr/bin/env python3
"""Transcription helper, run inside .venv where faster-whisper lives.

Kept separate from highlights.py so the main tool stays stdlib-only and still
runs (audio ranking, no political scoring) when the venv is absent.

Prints JSON: {"text": str, "segments": [{"start", "end", "text"}]}
Times are relative to the audio file it is given. The caller adds the offset.

usage: asr.py <audio> <lang> <model> [cpu_threads] [words]
With "words", each segment also carries [{"start","end","word"}] - used to
bleep individual swear words in rendered videos.
"""
import json
import sys


def main():
    audio, lang, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    want_words = len(sys.argv) > 5 and sys.argv[5] == "words"
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=threads)
    segments, _info = model.transcribe(
        audio,
        language=None if lang == "auto" else lang,
        vad_filter=True,
        beam_size=5,
        word_timestamps=want_words,
    )
    out = []
    for s in segments:
        text = s.text.strip()
        if text:
            item = {"start": round(s.start, 2), "end": round(s.end, 2), "text": text}
            if want_words:
                item["words"] = [{"start": round(w.start, 2), "end": round(w.end, 2),
                                  "word": w.word.strip()} for w in (s.words or [])]
            out.append(item)
    json.dump({"text": " ".join(s["text"] for s in out), "segments": out},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
