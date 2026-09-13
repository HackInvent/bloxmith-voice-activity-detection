"""Pinned, offline Silero inference; every capture owns its recurrent state."""
from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "assets" / "models" / "silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
FRAME_SAMPLES = 512
THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


def verified_model(path: Path = MODEL_PATH) -> Path:
    """Require the packaged v6.2.1 weights; never download or silently fall back at Run."""
    if not path.is_file() or path.stat().st_size > 3 * 1024 * 1024:
        raise ValueError("Modèle Silero absent/invalide : réinstallez le paquet du bloc VAD.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Modèle Silero altéré : réinstallez les poids vérifiés du bloc VAD.")
    return path


class SileroSpeech:
    """Score 32 ms of mono PCM16 at 16 kHz using CPU-only ONNX, without torch or network IO."""

    def __init__(self):
        """Load verified weights and independent context/state for this audio stream."""
        import numpy as np
        import onnxruntime as ort

        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(verified_model()), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, pcm: bytes) -> float:
        """Advance recurrent state once per complete window and return the model's speech score."""
        if len(pcm) != FRAME_SAMPLES * 2:
            raise ValueError("Silero attend exactement 512 échantillons PCM16 mono.")
        np = self.np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(1, -1) / 32768.0
        probability, state = self.session.run(None, {
            "input": np.concatenate((self.context, samples), axis=1),
            "state": self.state, "sr": np.array(16000, dtype=np.int64),
        })
        value = float(probability[0, 0])
        if not np.isfinite(value) or not 0 <= value <= 1 or not np.isfinite(state).all():
            raise ValueError("Score ou état Silero invalide ; aucune parole inventée.")
        self.state, self.context = state, samples[:, -64:].copy()
        return value
