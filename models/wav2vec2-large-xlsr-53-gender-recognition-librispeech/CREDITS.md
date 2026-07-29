# Audio credits — XLS-R acoustic-label model-risk exercises

## Bundled LibriSpeech recording

- **Local file:** `libri-61-70968-0000-16k-mono.wav`
- **Corpus record:** LibriSpeech `test-clean`, utterance `61-70968-0000`; original corpus path `test-clean/61/70968/61-70968-0000.flac`
- **Reader / work:** reader 61, Paul-Gabriel Wiener, reading *Robin Hood*, chapter 4 (LibriVox)
- **Corpus creators:** Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur
- **Record-level source:** [OpenSLR SLR12](https://www.openslr.org/12), [`test-clean.tar.gz`](https://www.openslr.org/resources/12/test-clean.tar.gz)
- **Licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Derivation:** the corpus FLAC was losslessly decoded to 16 kHz mono 16-bit PCM WAV; no content edit. Duration 4.91 s; 157,038 bytes.
- **SHA-256:** `1fb553adb5a6389eef5b7ebdbf9ed2a6082518a646ce9bf872bf33b964eedc14`
- **Source-local lineage:** byte-identical copy of `models/whisper-speech-to-text/libri-61-70968-0000-16k-mono.wav`, whose family credit records the same OpenSLR source and hash.
- **Corpus retrieval date:** 2026-07-27
- **Adopted by this family:** 2026-07-30
- **Reference transcript:** “He began a confused complaint against the wizard, who had vanished behind the curtain on the left.”

## Transformations used by these routes

- **Overview and Basics:** the complete 4.91 s WAV is decoded to mono float PCM, then zero-padded by 0.09 s to the model card's deterministic 5.000 s input.
- **Practical stability audit:** four disclosed within-utterance windows (`0.00–1.25 s`, `1.25–2.50 s`, `2.50–3.75 s`, `3.75–4.91 s`) are each zero-padded to exactly 5.000 s. The output is aggregate acoustic-label sensitivity/variation only; no expected identity label or person-level classification is accepted or exported.
- **Wild sensitivity probe:** the complete recording is conditionally resampled at the displayed playback rates. This simultaneously changes pitch, spectrum/formants, tempo, duration, and temporal structure. Each resulting waveform is then cropped or zero-padded to exactly 5.000 s before inference.
- **Acceptance known-output probes:** the original decoding and the documented `1.10×` rate-resampled transform are distinct model inputs derived from this one approved recording. Their raw logits are retained from genuine worker responses; no second recording is bundled.
