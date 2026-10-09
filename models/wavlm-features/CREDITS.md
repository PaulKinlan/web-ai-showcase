# Sample audio credits

The bundled sample clips are short (~10s) 16 kHz mono excerpts, reused from the other audio demos in
this showcase to demonstrate WavLM's self-supervised speech features (similarity, search, clustering).
All model inference happens on-device; nothing is uploaded.

| File | Speaker / language | Source | License |
|------|--------------------|--------|---------|
| jfk.wav | English (JFK) | John F. Kennedy inaugural excerpt | Public domain |
| librispeech.wav | English (JenniferRutters) | [LibriSpeech dev-clean, `7976-110523-0006`](https://www.openslr.org/12), *My Book Of Favourite Fairy Tales*; 16 kHz WAV decode. [Exact hash and modification record](../../audio-provenance/librispeech-replacements.md) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — corpus © 2014 Vassil Panayotov |
| sample-spa.wav | Spanish | Wikimedia Commons: Spanish Spoken Wikipedia — Dengue - Historia | CC BY-SA |
| sample-deu.wav | German | Wikimedia Commons: De-Thekenschaaf-article | CC BY-SA |
| sample-rus.wav | Russian | Wikimedia Commons: Ru-Russian language part 4 1 Old Russian period | CC BY-SA |

Clips were resampled to 16 kHz mono. Full Spoken-Wikipedia attributions are in
`models/spoken-language-id/CREDITS.md`.
