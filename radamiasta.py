#!/usr/bin/env python3
"""Szczecin City Council committee meetings: YouTube -> transcript -> website.

One pass (`./radamiasta.py`, run daily by a systemd timer):
  1. list the council's YouTube streams and keep the committee meetings ("Komisja");
  2. for each one not done yet: download the audio, transcribe it (faster-whisper,
     word timestamps) and separate the voices (sherpa-onnx), both via the helper
     scripts in helpers/, run with the venv that setup.sh makes;
  3. ask Claude (Sonnet, through the Claude Code subscription) to name the speakers
     and fix obvious recognition errors - nothing is rephrased or removed;
  4. build a static site in site/ and rsync it to Hostinger.

Every step caches its output in data/<video id>/, so a failed run picks up where it
stopped. Subcommands: run (default), build, deploy, list.
"""
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SITE = HERE / "site"
HELPERS = HERE / "helpers"
MODELS = HERE / "models"
PY = Path(os.environ.get("RADAMIASTA_PYTHON", HERE / ".venv/bin/python"))
CHANNEL = "https://www.youtube.com/@radamiastaszczecin7369/streams"
COMMITTEE = re.compile(r"komisj|committee", re.I)
YTDLP = [shutil.which("yt-dlp") or str(Path.home() / ".local/bin/yt-dlp"),
         "--cookies-from-browser", "brave", "--js-runtimes", "node",
         "--remote-components", "ejs:github", "--extractor-args", "youtube:lang=pl",
         "--no-warnings"]
WHISPER = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
THREADS = os.environ.get("THREADS", "16")
THRESHOLD = "0.2"           # room mics flatten voices: 0.3-0.7 lump most speakers into one
                            # cluster; 0.2 over-splits, a better hint for Claude
MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")
CHUNK_WORDS = 6000          # transcript words per Claude call


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{(p.stderr or p.stdout)[-800:]}")
    return p.stdout


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- 1. listing

def candidates():
    """Committee meetings on the channel, newest first (flat listing: id + title)."""
    out = run(YTDLP + ["--flat-playlist", "--print", "%(id)s\t%(title)s", CHANNEL])
    for line in out.splitlines():
        vid, _, title = line.partition("\t")
        if COMMITTEE.search(title):
            yield vid


def metadata(vid):
    d = DATA / vid
    meta = load(d / "meta.json")
    if meta:
        return meta
    info = json.loads(run(YTDLP + ["-J", "--skip-download", f"https://youtu.be/{vid}"]))
    if info.get("live_status") not in ("was_live", "not_live", None):
        return None         # still live or still processing: next run takes it
    date = info.get("release_date") or info.get("upload_date")
    meta = {"id": vid, "title": info["title"].strip(),
            "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "duration": info.get("duration") or 0}
    d.mkdir(parents=True, exist_ok=True)
    save(d / "meta.json", meta)
    return meta


# ---------------------------------------------------- 2. audio, ASR, diarization

def audio(vid):
    wav = DATA / vid / "audio.wav"
    if wav.exists():
        return wav
    log("  downloading audio")
    src = DATA / vid / "src"
    # YouTube adds AI-dubbed tracks (English) and marks them as the default: plain
    # "bestaudio" then transcribes the dub. Ask for the original track first.
    run(YTDLP + ["-f", "ba[language^=pl]/ba[format_note*=original]/ba[format_note*=oryginal]/ba", "-o",
                 f"{src}.%(ext)s", f"https://youtu.be/{vid}"])
    got = next(DATA.joinpath(vid).glob("src.*"))
    run(["ffmpeg", "-y", "-nostats", "-hide_banner", "-i", str(got), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    got.unlink()
    return wav


def transcribe(vid):
    out = DATA / vid / "asr.json"
    if not out.exists():
        log(f"  transcribing ({WHISPER})")
        data = json.loads(run([str(PY), str(HELPERS / "asr.py"), str(audio(vid)), "pl",
                               WHISPER, THREADS, "words"]))
        en = len(re.findall(r"\b(the|and|of|is|that)\b", data["text"]))
        pl = len(re.findall(r"\b(się|jest|nie|że|to)\b", data["text"]))
        if en > pl:                    # a dubbed track slipped through
            (DATA / vid / "audio.wav").unlink(missing_ok=True)
            raise RuntimeError(f"transcript looks English ({en} vs {pl}): wrong audio track")
        save(out, data)
    return load(out)


def diarize(vid):
    out = DATA / vid / "diar.json"
    if not out.exists():
        log("  separating voices")
        data = json.loads(run([str(PY), str(HELPERS / "diarize.py"), str(audio(vid)),
                               str(MODELS), THRESHOLD, THREADS]))
        for s in data["segments"]:
            s.pop("embedding", None)         # only needed for face/voice matching
        save(out, {"segments": data["segments"]})
    return load(out)


def label(n):
    """Cluster 0, 1, ... -> A, B, ..., Z, AA, AB ..."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def segments_from(asr, diar):
    """Whisper segments (roughly sentences), each with the voice that overlaps it most.
    The voices are only a hint for Claude: in these rooms one microphone carries
    several people and the clustering is unreliable at any threshold."""
    segs, out, prev = diar["segments"], [], "?"
    for s in asr["segments"]:
        overlap = {}
        for d in segs:
            if d["end"] > s["start"] and d["start"] < s["end"]:
                o = min(d["end"], s["end"]) - max(d["start"], s["start"])
                overlap[d["cluster"]] = overlap.get(d["cluster"], 0) + o
        prev = label(max(overlap, key=overlap.get)) if overlap else prev
        out.append({"start": s["start"], "end": s["end"], "text": s["text"], "voice": prev})
    return out


# ------------------------------------------------------------------ 3. Claude

SYSTEM = """Jesteś redaktorem automatycznych transkrypcji posiedzeń komisji Rady Miasta \
Szczecin. Dostajesz transkrypcję z rozpoznawania mowy w zdaniach. Każde zdanie ma numer, \
czas i etykietę głosu (A, B, C...) z automatycznego rozpoznawania mówców. Etykiety są \
tylko wskazówką: bywają błędne, jedna osoba ma często kilka etykiet, a jedna etykieta \
czasem obejmuje kilka osób.

Twoje zadania:
1. Podziel transkrypcję na wypowiedzi: podaj numer zdania, od którego mówi kolejna \
osoba, i kto to jest. Opieraj się przede wszystkim na treści (przewodniczący udziela \
głosu, "Dziękuję, Panie Przewodniczący", pytanie i odpowiedź, ktoś się przedstawia, \
zmiana tematu lub tonu), a na etykietach głosu pomocniczo. Pierwszy wpis w "turns" \
zaczyna się od pierwszego zdania tej części.
2. Każdej osobie nadaj stały identyfikator (P1, P2...), imię i nazwisko oraz funkcję \
(np. "przewodniczący komisji", "radna", "dyrektor MOSiR"), jeśli wynikają z nagrania. \
Gdy nie wiesz, name = null - nie zgaduj. Lista radnych służy WYŁĄCZNIE do pisowni: nie \
przypisuj nazwiska z listy, jeśli to nazwisko nie pada w nagraniu w związku z tą osobą. \
Osoby ustalone we wcześniejszych częściach mają już identyfikatory - używaj ich.
3. Popraw tylko OCZYWISTE błędy rozpoznawania mowy: przekręcone nazwiska, nazwy ulic, \
instytucji, słowa niepasujące do kontekstu, które brzmią podobnie do właściwych. Nie \
zmieniaj stylu, szyku, gramatyki mówiącego, nie usuwaj powtórzeń ani wtrąceń, nie \
dopisuj niczego. Nie zmieniaj zdrobnień ani form zwracania się ("Panie Heniu" zostaje). \
Nazwy szczecińskich dzielnic, ulic i klubów odmieniają się ("na Pogodnie" to dzielnica \
Pogodno). Poprawiaj tylko, gdy jesteś pewien; w razie wątpliwości zostaw. \
Nie poprawiaj, jeśli poprawka wymaga wiedzy spoza nagrania (np. nazwa obiektu, której \
nie słychać wyraźnie) albo gdy nie wiesz, co padło - lepiej zostawić błąd niż zgadnąć. "from" musi \
być dokładnym, możliwie krótkim fragmentem tekstu zdania o numerze "segment", "to" - \
poprawionym brzmieniem."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["people", "turns", "fixes"],
    "properties": {
        "people": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "name", "role"],
            "properties": {"id": {"type": "string"},
                           "name": {"type": ["string", "null"]},
                           "role": {"type": ["string", "null"]}}}},
        "turns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["segment", "person"],
            "properties": {"segment": {"type": "integer"}, "person": {"type": "string"}}}},
        "fixes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["segment", "from", "to"],
            "properties": {"segment": {"type": "integer"}, "from": {"type": "string"},
                           "to": {"type": "string"}}}}}}


def hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def claude(instruction, text):
    cmd = [shutil.which("claude") or "claude", "-p", instruction,
           "--system-prompt", SYSTEM, "--model", MODEL, "--effort", "high",
           "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--no-session-persistence", "--output-format", "json",
           "--json-schema", json.dumps(SCHEMA)]
    data = json.loads(run(cmd, input=text, cwd=HERE, timeout=3600))
    if data.get("is_error"):
        raise RuntimeError(f"Claude: {str(data.get('result'))[:400]}")
    return data.get("structured_output") or json.loads(data["result"])


def people_so_far(done):
    people = {}
    for c in done["chunks"]:
        for p in c["people"]:
            if p["name"] or p["role"] or p["id"] not in people:
                people[p["id"]] = {"name": p["name"], "role": p["role"]}
    return people


def edit(vid, meta, segs):
    """Run Claude over the transcript in chunks, carrying the people found so far."""
    out = DATA / vid / "llm.json"
    done = load(out) or {"chunks": []}
    radni = HERE / "radni.txt"
    roster = (f"\n\nSkład Rady Miasta Szczecin (tylko pisownia nazwisk):\n{radni.read_text()}"
              if radni.exists() else "")
    chunks, cur, words = [], [], 0
    for n, s in enumerate(segs):
        cur.append(n)
        words += len(s["text"].split())
        if words >= CHUNK_WORDS:
            chunks.append(cur)
            cur, words = [], 0
    if cur:
        chunks.append(cur)
    for k, idx in enumerate(chunks):
        if k < len(done["chunks"]):
            continue
        log(f"  Claude ({MODEL}): part {k + 1}/{len(chunks)}")
        known = people_so_far(done)
        ctx = ("\nOsoby z wcześniejszych części: " + "; ".join(
            f'{i} = {p["name"] or "?"} ({p["role"] or "?"})' for i, p in known.items())
            if known else "")
        text = "\n".join(f'[{n}] {hms(segs[n]["start"])} {segs[n]["voice"]}: '
                         f'{segs[n]["text"]}' for n in idx)
        instr = (f'Posiedzenie: {meta["title"]}, {meta["date"]}. Część {k + 1} z '
                 f'{len(chunks)} (zdania {idx[0]}-{idx[-1]}).{ctx}{roster}\n\n'
                 f'Transkrypcja jest na wejściu (stdin).')
        done["chunks"].append(claude(instr, text))
        save(out, done)
    return done


def apply_edits(segs, llm):
    people = people_so_far(llm)
    starts = {t["segment"]: t["person"] for c in llm["chunks"] for t in c["turns"]}
    fixes = {}
    for c in llm["chunks"]:
        for f in c["fixes"]:
            fixes.setdefault(f["segment"], []).append(f)
    turns, who = [], "?"
    for n, s in enumerate(segs):
        text, done = s["text"], []
        for f in fixes.get(n, []):
            if f["from"] and f["from"] in text and f["from"] != f["to"]:
                text = text.replace(f["from"], f["to"], 1)
                done.append({"from": f["from"], "to": f["to"]})
        new = n in starts and starts[n] != who
        who = starts.get(n, who)
        if turns and not new:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = s["end"]
            turns[-1]["fixes"] += done
        else:
            turns.append({"speaker": who, "start": s["start"], "end": s["end"],
                          "text": text, "fixes": done})
    return people, turns


def process(meta):
    vid, d = meta["id"], DATA / meta["id"]
    if (d / "final.json").exists():
        return False
    log(f'{meta["date"]} {meta["title"]}')
    segs = segments_from(transcribe(vid), diarize(vid))
    names, final = apply_edits(segs, edit(vid, meta, segs))
    save(d / "final.json", {**meta, "speakers": names, "turns": final,
                            "whisper": WHISPER, "model": MODEL})
    (d / "audio.wav").unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------- 4. site

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e1dc;--acc:#b0164a;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--mut:#9a9aa1;
--line:#2c2c30;--acc:#ff5c8a;--card:#1c1c1f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 80px}
a{color:var(--acc)}h1{font-size:1.6rem;line-height:1.25;margin:.2em 0}
.mut{color:var(--mut);font-size:.9rem}
.list{list-style:none;padding:0;margin:24px 0}.list li{padding:14px 0;border-top:1px solid var(--line)}
.list a{font-weight:600;text-decoration:none}.list a:hover{text-decoration:underline}
.player{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0}
.player div{position:relative;aspect-ratio:16/9;max-height:38vh;margin:0 auto}
.player iframe{position:absolute;inset:0;width:100%;height:100%;border:0;border-radius:8px}
.who{display:flex;flex-wrap:wrap;gap:6px 14px;margin:14px 0;padding:0;list-style:none;font-size:.9rem}
.turn{display:grid;grid-template-columns:5.2em 1fr;gap:4px 12px;padding:10px 0;border-top:1px solid var(--line)}
.turn button{font:inherit;font-size:.85rem;color:var(--acc);background:none;border:0;
padding:0;cursor:pointer;text-align:left;font-variant-numeric:tabular-nums}
.turn b{display:block;font-size:.9rem}.turn b i{font-weight:400;color:var(--mut)}
.turn p{margin:2px 0 0}.fix{text-decoration:underline dotted var(--mut);cursor:help}
input{font:inherit;width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--fg)}
@media (max-width:600px){.turn{grid-template-columns:1fr}}
"""

JS = """
var p;function onYouTubeIframeAPIReady(){p=new YT.Player('yt')}
document.querySelectorAll('.turn button').forEach(function(b){b.onclick=function(){
 if(p&&p.seekTo){p.seekTo(+b.dataset.t,true);p.playVideo()}}});
var q=document.getElementById('q');q&&q.addEventListener('input',function(){
 var s=q.value.toLowerCase();document.querySelectorAll('.turn').forEach(function(t){
 t.hidden=s&&t.textContent.toLowerCase().indexOf(s)<0})});
"""

NOTE = ("Transkrypcja automatyczna (Whisper). Mówców rozpoznał i oczywiste błędy "
        "rozpoznawania mowy poprawił model AI (Claude) - poprawki są podkreślone, "
        "po najechaniu widać pierwotny zapis. Tekst może zawierać błędy; "
        "rozstrzyga nagranie.")


def page(title, body, desc=""):
    e = html.escape
    return (f'<!doctype html><html lang="pl"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{e(title)}</title><meta name="description" content="{e(desc)}">'
            f'<style>{CSS}</style></head><body><main>{body}</main></body></html>')


def roster_surnames():
    radni = HERE / "radni.txt"
    text = radni.read_text() if radni.exists() else ""
    return {m.group(2) for m in re.finditer(r"([A-ZŁŚŻŹĆ]\w+) ([A-ZŁŚŻŹĆ][\w-]+)", text)}


SURNAMES = roster_surnames()


def who(names, spk):
    """Only names whose surname is on the roster are published: guests' names come out
    as Whisper heard them, and a first name alone says little. Others keep their role."""
    n = names.get(spk) or {}
    name = n.get("name") or ""
    parts = name.split()
    if len(parts) < 2 or parts[-1] not in SURNAMES:
        name = ""
    return name or f"Mówca {spk}", n.get("role") or ""


def render_text(t):
    s = html.escape(t["text"])
    for f in t["fixes"]:
        to = html.escape(f["to"])
        s = s.replace(to, f'<span class="fix" title="pierwotnie: '
                          f'{html.escape(f["from"], quote=True)}">{to}</span>', 1)
    return s


def build():
    SITE.mkdir(exist_ok=True)
    meetings = sorted((load(p) for p in DATA.glob("*/final.json")),
                      key=lambda m: m["date"], reverse=True)
    e = html.escape
    for m in meetings:
        secs = {}
        for t in m["turns"]:
            k = who(m["speakers"], t["speaker"])
            secs[k] = secs.get(k, 0) + t["end"] - t["start"]
        legend = "".join(f"<li>{e(n)}{' – ' + e(r) if r else ''} "
                         f"<span class=mut>{round(s / 60)} min</span></li>"
                         for (n, r), s in sorted(secs.items(), key=lambda x: -x[1]))
        turns = "".join(
            f'<div class=turn><button data-t="{int(t["start"])}">{hms(t["start"])}</button>'
            f'<div><b>{e(n)}{f" <i>{e(r)}</i>" if r else ""}</b><p>{render_text(t)}</p>'
            f'</div></div>'
            for t in m["turns"] for n, r in [who(m["speakers"], t["speaker"])])
        body = (f'<p class=mut><a href="./">← Wszystkie posiedzenia</a></p>'
                f'<h1>{e(m["title"])}</h1><p class=mut>{m["date"]} · {hms(m["duration"])} · '
                f'<a href="https://youtu.be/{m["id"]}">YouTube</a> · '
                f'<a href="{m["id"]}.txt">tekst (.txt)</a></p>'
                f'<div class=player><div><iframe id=yt allowfullscreen '
                f'src="https://www.youtube-nocookie.com/embed/{m["id"]}?enablejsapi=1">'
                f'</iframe></div></div><ul class=who>{legend}</ul>'
                f'<input id=q type=search placeholder="Szukaj w transkrypcji…">'
                f'{turns}<p class=mut>{NOTE}</p>'
                f'<script src="https://www.youtube.com/iframe_api"></script><script>{JS}</script>')
        (SITE / f'{m["id"]}.html').write_text(page(f'{m["title"]} – transkrypcja', body,
                                                   f'Transkrypcja: {m["title"]}'))
        (SITE / f'{m["id"]}.txt').write_text(
            f'{m["title"]}\n{m["date"]} · https://youtu.be/{m["id"]}\n\n' + "\n\n".join(
                f'[{hms(t["start"])}] {" – ".join(x for x in who(m["speakers"], t["speaker"]) if x)}'
                f':\n{t["text"]}' for t in m["turns"]) + f"\n\n{NOTE}\n")
    items = "".join(f'<li><a href="{m["id"]}.html">{e(m["title"])}</a><br>'
                    f'<span class=mut>{m["date"]} · {hms(m["duration"])}</span></li>'
                    for m in meetings)
    (SITE / "index.html").write_text(page(
        "Komisje Rady Miasta Szczecin – transkrypcje",
        f'<h1>Komisje Rady Miasta Szczecin</h1><p class=mut>Transkrypcje posiedzeń komisji '
        f'z <a href="{CHANNEL}">transmisji Rady Miasta</a>, z podziałem na mówców. '
        f'Aktualizowane codziennie.</p><ul class=list>{items}</ul><p class=mut>{NOTE}</p>',
        "Transkrypcje posiedzeń komisji Rady Miasta Szczecin z podziałem na mówców."))
    for stale in SITE.glob("*.html"):
        if stale.stem != "index" and not (DATA / stale.stem / "final.json").exists():
            stale.unlink()
    log(f"site: {len(meetings)} meetings")


def deploy():
    cfg = dict(l.split("=", 1) for l in (HERE / "config.env").read_text().split("\n")
               if "=" in l and not l.startswith("#"))
    ssh = (f'ssh -p {cfg.get("SSH_PORT", "65002")} -i {os.path.expanduser(cfg["SSH_KEY"])} '
           f'-o StrictHostKeyChecking=accept-new')
    run(["rsync", "-rz", "--checksum", "--delete", "-e", ssh, f"{SITE}/",
         f'{cfg["SSH_USER"]}@{cfg["SSH_HOST"]}:{cfg["SSH_DIR"].rstrip("/")}/'])
    log("deployed")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "list":
        for vid in candidates():
            m = metadata(vid)
            print(vid, m and m["date"], m and m["title"],
                  "done" if (DATA / vid / "final.json").exists() else "")
        return
    if cmd == "run":
        new = 0
        for vid in candidates():
            meta = metadata(vid)
            if meta:
                try:
                    new += process(meta)
                except Exception as e:      # one bad meeting must not block the rest
                    log(f"  FAILED: {e}")
        if not new and cmd == "run" and (SITE / "index.html").exists():
            log("nothing new")
            return
    if cmd in ("run", "build"):
        build()
    if cmd in ("run", "deploy"):
        if (HERE / "config.env").exists():
            deploy()
        else:
            log("no config.env: skipping upload")


if __name__ == "__main__":
    main()
