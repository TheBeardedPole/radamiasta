#!/usr/bin/env python3
"""Who spoke when. Runs inside .venv, where sherpa-onnx lives.

usage: diarize.py <wav 16k mono> <models dir> <threshold> <threads>

Prints JSON:
  {"segments": [{"start", "end", "cluster", "embedding"}],
   "clusters": {id: {"seconds", "turns", "embedding"}},
   "model": "<embedding model file name>"}

Embeddings are L2-normalised, so cosine similarity is a plain dot product.
Clustering is deliberately on the eager side (a lower threshold): splitting one
person into two clusters is harmless because identification picks every cluster
that is them, while merging two people into one cluster is not recoverable.
"""
import json
import sys
import wave

import numpy as np
import sherpa_onnx

EMBEDDING_MODEL = "wespeaker_en_voxceleb_resnet34_LM.onnx"
SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"


def read_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit("expected 16 kHz mono wav")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def main():
    wav, models, threshold, threads = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    audio = read_wav(wav)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=f"{models}/{SEGMENTATION_MODEL}"),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=f"{models}/{EMBEDDING_MODEL}", num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if not sd:
        raise SystemExit("could not load the diarization models")
    result = sd.process(audio).sort_by_start_time()

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=f"{models}/{EMBEDDING_MODEL}",
                                                    num_threads=threads))
    segments = []
    sums, secs, turns = {}, {}, {}
    for seg in result:
        a, b = int(seg.start * 16000), int(seg.end * 16000)
        chunk = audio[a:b]
        emb = None
        if len(chunk) >= 16000:          # under 1 s the fingerprint is too noisy to trust
            stream = extractor.create_stream()
            stream.accept_waveform(16000, chunk)
            stream.input_finished()
            emb = norm(extractor.compute(stream))
            dur = seg.end - seg.start
            sums[seg.speaker] = sums.get(seg.speaker, 0) + emb * dur
        secs[seg.speaker] = secs.get(seg.speaker, 0) + (seg.end - seg.start)
        turns[seg.speaker] = turns.get(seg.speaker, 0) + 1
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                         "cluster": int(seg.speaker),
                         "embedding": None if emb is None else [round(float(x), 5) for x in emb]})

    clusters = {}
    for spk in secs:
        emb = norm(sums[spk]) if spk in sums else None
        clusters[str(int(spk))] = {
            "seconds": round(secs[spk], 1), "turns": turns[spk],
            "embedding": None if emb is None else [round(float(x), 5) for x in emb]}
    json.dump({"segments": segments, "clusters": clusters, "model": EMBEDDING_MODEL},
              sys.stdout)


if __name__ == "__main__":
    main()
