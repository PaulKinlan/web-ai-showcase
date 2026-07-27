# Sample audio credits

The bundled sample is a complete, openly licensed **song** used by the demo's one-click sample
button. All model inference happens on-device; nothing is uploaded.

| File     | Song             | Artist                                         | Source                                                                                                                           | License                                                   |
| -------- | ---------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| song.mp3 | "The CC BY Song" | loveshadow (additional lyrics by Victor Stone) | [ccMixter — The CC BY Song](https://ccmixter.org/files/Loveshadow/29635) ([artist page](https://ccmixter.org/people/loveshadow)) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |

Retrieved 2026-07-27 from ccMixter's canonical API
([`https://ccmixter.org/api/query?f=json&ids=29635`](https://ccmixter.org/api/query?f=json&ids=29635)):
upload 29635 "The CC BY Song" by Loveshadow, featuring "Additional lyrics by Victor Stone", license
"Attribution (3.0)", uploaded 2010-12-11, tagged `acoustic, country, humour, male_vocals` — a song
about Creative Commons licences, which makes it a fitting bundle for a demo built on CC audio. The
bundled file is the upload's **primary file, unmodified**,
[downloaded in full](https://ccmixter.org/content/Loveshadow/Loveshadow_-_The_CC_BY_Song.mp3) (the
`download_url` from the API record): MP3 VBR, 44.1 kHz stereo, 2:22 (141.67 s), 4,625,972 bytes,
**SHA-256 `9ae38593020674f9f89879f79163031392f00a937b5332365f504be24b2e91aa`** (ccMixter file SHA-1
`AALIHCF55XNXLUYBQ4USS3XFQYJBUWD2`). The same upload also publishes the **separate stem assets**
under the same licence — "CC BY Vox" (`Loveshadow_-_The_CC_BY_Song_3.mp3`), "CC BY Guitars"
(`Loveshadow_-_The_CC_BY_Song_4.mp3`) and "CC By Bass" (`Loveshadow_-_The_CC_BY_Song_1.mp3`), each
2:25 — not bundled: the demo's whole point is separating the complete mix on-device. A
machine-readable record of these facts (URLs, retrieval date, bytes, SHA-256, creator/contributor,
licence, local path) lives in [provenance.json](provenance.json), and each route carries a
route-local `CREDITS.md` pointing back here. Attribution required by the license is shown next to
the sample button on every page: **"The CC BY Song" by loveshadow, CC BY 3.0, via ccMixter**.

**Sample window:** the demo separates the model's fixed ~7.8 s window starting at **0.0 s** — the
plain opening of the song. Measured with the real Demucs engine in-browser (2026-07-27): the song
sings from the first bar (vocals-stem RMS 0.064 at 0 s and strong throughout, 0.064–0.098), so no
offset is needed; the window is exposed honestly on every page, and the bundled file is the
complete, unmodified song.

**History:** the sample was previously "Swansong" (Josh Woodward, CC BY 3.0 US) and then "Falling"
(Donnie Drost, CC BY 3.0) — the latter after the approved "The Same Song" by Donnie Drost turned out
not to exist on ccMixter (canonical API lists all 50 donniedrost uploads; the approved file-page id
belongs to a different artist's NC track; the MP3 URL 404s).

Model weights (not bundled, downloaded on-device at the visitor's request): Hybrid Transformer
Demucs (htdemucs), Meta AI — [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), served
from the ONNX export at [MrCitron/demucs-v4-onnx](https://huggingface.co/MrCitron/demucs-v4-onnx).

## Validation-only ASR speech fixture (not part of any demo)

| File                                      | Utterance                                                                                                 | Creator                                                                                 | Source                                                                                                                    | License                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| fixtures/libri-61-70968-0000-16k-mono.wav | LibriSpeech test-clean `61-70968-0000` (reader 61, Paul-Gabriel Wiener; LibriVox "Robin Hood", chapter 4) | LibriSpeech ASR corpus — Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur | [OpenSLR SLR12](https://www.openslr.org/12) ([test-clean.tar.gz](https://www.openslr.org/resources/12/test-clean.tar.gz)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

This clip is **not demo content**: no route plays, links, or references it. It exists solely so
`scripts/validate-music-source-separation.mjs` can prove the multi-model route's whisper-tiny.en
worker genuinely transcribes real speech (the anti-stub health check) from a **family-local,
rights-safe** asset — replacing a former cross-family dependency on another family's sample. The
bundled file is a lossless 16 kHz mono 16-bit WAV derivative of the corpus's original FLAC
`test-clean/61/70968/61-70968-0000.flac` (4.91 s, 157,038 bytes, **SHA-256
`1fb553adb5a6389eef5b7ebdbf9ed2a6082518a646ce9bf872bf33b964eedc14`**); reference transcript
(`61-70968.trans.txt`): "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND
THE CURTAIN ON THE LEFT". Corpus retrieved 2026-07-27; full machine-readable record in
[provenance.json](provenance.json).

The multi-model route also downloads
[onnx-community/whisper-tiny.en](https://huggingface.co/onnx-community/whisper-tiny.en) (Whisper
tiny English ASR, OpenAI — [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), ONNX export
for transformers.js, ~40 MB) — only when the visitor presses its Download button. Its second example
also downloads
[onnx-community/Musical-Instrument-Classification-ONNX](https://huggingface.co/onnx-community/Musical-Instrument-Classification-ONNX)
(the 9-class instrument specialist, [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), ~95
MB) — again only on an explicit Download press.
