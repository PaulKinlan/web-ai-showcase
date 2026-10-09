# JFK speech sample — separate static attribution audit (2026-10-09)

**Scope:** read-only provenance/source and source-text link inventory while `web-ai-showcase-z79` replaces TED bytes. **No JFK bytes or UI were changed** by this audit. No browser was launched; link visibility/click destinations and model behavior remain unverified in this audit. This report does **not** revise the established public-domain classification in `audio-provenance/ledger.json`.

## Content identity and upstream derivation

The 17 tracked paths under the JFK ledger entry all hash to `627f0e49f927ffcd4120ed60a035ff2f8d448e2e7469452c7ecaffed06fe135b` (352,078 bytes, 11.0 s, mono 16 kHz PCM16), across 15 carrier families; six paths use names other than `jfk.wav`. The ledger already records `rightsCleared: true` and JFK inaugural/public-domain attribution. The named [Xenova/transformers.js-docs `jfk.wav` mirror](https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav) returned an 11.0 s, 44.1 kHz stereo WAV on this date (1,940,478 bytes, SHA-256 `aa81c2552465568567e670f3823117e633900d16bd6202346a72f3c8464c74c8`). **The bytes are not identical because the formats differ.** FFmpeg decoding that source with `-ac 1 -ar 16000 -f s16le` produced 176,000 samples **byte-identical to the bundled WAV's PCM**. This establishes a concrete derivation from the linked mirror, not a claim that the linked original file has the bundled hash.

`media/manifest.json:1188-1193` describes this same local 11-second file as from a “CMU/LibriSpeech sample set”, links a Wikimedia Commons `.ogg` as `sourceUrl`, and links Wikipedia as `licenseUrl`. The exact linked Xenova WAV conversion above supports the ledger's Xenova source; the manifest's CMU/LibriSpeech attribution is inconsistent with that derivation, while equivalence of its separate Commons `.ogg` was **not checked**. Wikipedia is not itself a licence deed. **Follow-up:** reconcile the manifest source and licence pointer with independently sourced public-domain authority rather than silently choosing a URL. Do not conflate JFK with OpenSLR LibriSpeech merely because it is another speech sample.

## Route-credit source-text inventory (not browser acceptance)

Method: group the 17 ledger paths by `models/<family>/`; find HTML routes whose source text names that family's exact WAV basename; count pages containing either the previously accepted `JFK audio source and attribution record` link or a route-local `CREDITS.md` link. Counts below are **static source matches only**, not proof of visible, clickable, content-verified attribution in a browser. The same `CREDITS.md` might not actually be served at every route depth; that remains to be tested.

| Carrier family | Matching route pages with a source-credit link / total matching pages |
|---|---:|
| ast-audio-classification | 3 / 3 |
| distil-whisper-asr | **0 / 5** |
| gtcrn-speech-enhancement | 4 / 4 |
| hubert-features | **0 / 5** |
| mms-forced-alignment | 3 / 3 |
| moonshine-asr | 4 / 4 |
| silero-vad | 4 / 4 |
| speaker-diarization | **0 / 4** |
| speech-commands | 1 / 1 |
| speech-emotion-recognition | **0 / 5** |
| spoken-language-id | **1 / 5** (overview links `CREDITS.md`; remaining four do not) |
| wav2vec2-asr | 5 / 5 |
| wavlm-features | **0 / 5** |
| whisper-large-v3-turbo | **0 / 5** |
| whisper-speech-to-text | 5 / 5 |

Total **30 / 63** source-referencing pages contain one of these credit-link patterns; **33** pages across seven families lack them. `hubert-features` and `wavlm-features` have local `CREDITS.md` files listing JFK public domain but no link on their matching pages. These are an attribution **coverage gap**, not evidence the JFK audio itself is TED-like or rights-unclear. Other families' previously captured browser JFK-credit waves were not rerun here.

**Next action:** create/assign a separate JFK source-metadata and remaining seven-family route-credit remediation bead. Browser validation must wait for independently reviewed Chrome containment and an exclusive slot. Keep this finding separate from TED/LibriSpeech asset changes; do not count this static source inventory as a passing click-through matrix.
