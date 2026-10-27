# Rada Miasta – transkrypcje komisji

Transcripts of Szczecin City Council committee meetings, split by speaker, published at
**[radamiasta.adamkoscielak.pl](https://radamiasta.adamkoscielak.pl)**.

The council streams its committee meetings on
[YouTube](https://www.youtube.com/@radamiastaszczecin7369/streams) but publishes no
transcripts. Once a day this tool picks up new committee streams, transcribes them, lets a
language model work out who is speaking, and rebuilds a static site where you can read
along, search, jump to any moment of the video, or download the text.

## How it works

One script, `radamiasta.py`, one pass:

1. **List** the channel's streams with yt-dlp and keep titles mentioning a committee
   ("Komisja"). Streams still live or still processing wait for the next run.
2. **Transcribe** the audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (`large-v3-turbo`, CPU) and **separate voices** with
   [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (`helpers/`).
3. **Edit** with Claude (Sonnet, via the `claude` CLI). The model receives the sentences
   with voice labels and returns only:
   - where each new speaker starts and who it is (name and role, only when said in the
     recording; otherwise the role alone),
   - exact-substring fixes of obvious speech-recognition errors.

   It never rewrites the text. Each fix is underlined on the page with the original on
   hover.
4. **Build** `site/` (one page per meeting, an index, `.txt` downloads) and **rsync** it to
   the server.

Every step caches its output in `data/<video id>/`, so an interrupted run resumes.

### Why speakers come from the text

The meeting rooms record through shared microphones, and acoustic diarization does badly
on them: at usual thresholds one cluster takes ~90% of the speech. The voice labels are
therefore only a hint (clustered at a deliberately low threshold) and the speaker turns
come from what is said: the chair giving the floor, "Dziękuję, Panie Przewodniczący",
questions and answers.

## Running it

Needs Python 3.10+, ffmpeg, [yt-dlp](https://github.com/yt-dlp/yt-dlp) with Node.js,
and [Claude Code](https://claude.com/claude-code) logged in.

```sh
./setup.sh                  # venv + voice models
./radamiasta.py list        # committee meetings on the channel
./radamiasta.py run         # process new ones, build, upload
./radamiasta.py build       # rebuild site/ only
```

yt-dlp reads YouTube cookies from Brave (`--cookies-from-browser brave` in `YTDLP`);
change that to your browser. Uploading needs `config.env` (see `config.env.example`);
without it the site is only built locally. Settings via environment: `WHISPER_MODEL`,
`CLAUDE_MODEL`, `THREADS`, `RADAMIASTA_PYTHON`.

`systemd/` has a user timer that runs it daily:

```sh
cp systemd/* ~/.config/systemd/user/ && systemctl --user enable --now radamiasta.timer
```

A 50-minute meeting takes about 8 minutes on a 16-core desktop.

`radni.txt` lists councillors only to help with spelling names; the model is told not to
attach a name to anyone unless it is said in the recording.

## Caveats

Transcripts are automatic and may contain errors; the recording is what counts. Speaker
attribution is the model's reading of the conversation, not an official record.

## License

[GNU GPL v3](LICENSE).
