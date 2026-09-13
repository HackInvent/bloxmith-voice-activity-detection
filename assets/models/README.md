# Speech Detection Model

`silero_vad.onnx` is the official **Silero VAD v6.2.1** model, executed locally by ONNX Runtime. No weights are downloaded during Run, and no key is required.

- Source: [Silero VAD](https://github.com/snakers4/silero-vad)
- Revision: `7e30209a3e901f9842f81b225f3e93d8199902b1`
- Upstream file: `src/silero_vad/data/silero_vad.onnx`
- SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- MIT license reproduced in [LICENSE.silero](LICENSE.silero).

The block verifies integrity before loading the weights. A missing or modified model produces an explicit error, never a silent fallback to WebRTC.
