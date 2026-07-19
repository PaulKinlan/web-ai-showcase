# Sample audio credits

The bundled sample clips are short (~10s) excerpts of volunteer "Spoken Wikipedia" recordings from
Wikimedia Commons, used to demonstrate spoken-language identification. Each is licensed CC BY-SA and
attributed to its Commons file page below. `sample-eng.wav` is a public-domain excerpt of a John F.
Kennedy speech (reused from the Whisper demo).

| File | Language | Source (Wikimedia Commons) | License |
|------|----------|----------------------------|---------|
| sample-eng.wav | English | John F. Kennedy inaugural excerpt (public domain) | Public domain |
| sample-spa.wav | Spanish | File:Spanish Spoken Wikipedia - Dengue - Historia.ogg | CC BY-SA |
| sample-deu.wav | German | File:De-Thekenschaaf-article.ogg | CC BY-SA |
| sample-ita.wav | Italian | File:Itwiki-Albena Denkova.ogg | CC BY-SA |
| sample-rus.wav | Russian | File:Ru-Russian language part 4 1 Old Russian period.ogg | CC BY-SA |
| sample-nld.wav | Dutch | File:Nl-Wikipedia-Wikiproject gesproken Wikipedia-article.ogg | CC BY-SA |
| sample-cmn.wav | Mandarin Chinese | File:Zh-cn-spoken-wikipedia-Bubbling Well Road (Shanghai) paragraph 1-3.ogg | CC BY-SA |

Clips were resampled to 16 kHz mono and trimmed with ffmpeg. All model inference happens on-device.
