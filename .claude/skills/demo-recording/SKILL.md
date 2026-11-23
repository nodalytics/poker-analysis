---
name: demo-recording
description: Record a screen video of Kestrel playing, or of tests and checks running live, to show teammates, partners or reviewers what the engine does — using a local Kestrel server and a scripted headless-browser (Playwright) recording, converted to MP4, with every frame checked for secrets. Use it whenever someone asks for a demo, a video, a screen recording, "show him it working", proof that a test was run, or visual evidence for a review, even if they don't say Playwright.
---

# Demo recordings

The recordings go to `Quants Pub/demo-videos/` (not `~/Videos`) as `<what>-<YYYY-MM-DD>.mp4`. Keep them short
(1-2 minutes, about 1.5 MB at 1280x800) and honest: what's on screen must be the real engine and real output, never
replayed or mocked.

## 1. Run Kestrel locally, from current code

```sh
cmake --build kestrel/build -j8                       # rebuild first: a stale binary is not "the engine"
S=<scratch>/kdemo; mkdir -p $S/models $S/out $S/config
kestrel/build/kestreld models install research/dist/models/mtt9-2.tar.gz --dir $S/models
cd kestrel && (setsid nohup ./build/kestreld serve --port 18891 --lan off --pack ../research/blueprint/pack \
    --models $S/models --web web --assets assets --connectors connectors --out $S/out --db off --config $S/config \
    > $S/serve.log 2>&1 &)
curl -s 127.0.0.1:18891/v1/engines        # mtt9: available true, practice true (practice needs --pack)
```

`--lan off` means no access token and nothing private in the URL. Don't record the lab's own Kestrel: it needs its
token, and its address would show.

## 2. Script the recording

Playwright comes from the cached MCP install (`~/.npm/_npx/*/node_modules/playwright`). Its browser is Chromium from
`~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome`, passed as `executablePath`, because the matching headless
shell may not be downloaded.

```js
const { chromium } = require(process.env.PW);
const b = await chromium.launch({ executablePath: process.env.CHROME });
const ctx = await b.newContext({ viewport: { width: 1280, height: 800 },
                                 recordVideo: { dir: OUT, size: { width: 1280, height: 800 } } });
const p = await ctx.newPage(); await p.goto('http://127.0.0.1:18891/');
await p.click('#new'); await p.getByText('Practice tournament (9-max)').click();
await p.selectOption('#pwho', 'watch'); await p.fill('#pavg', '12'); await p.selectOption('#ppace', '1800');
await p.click('#start'); await p.waitForTimeout(90000);
await ctx.close();                                     // the video is written on close
```

- First take a screenshot and list the page's controls (`button, select, input` with their ids), so the script uses
  real selectors.
- Pick settings that show the point being made. For example, a 12bb average shows ICM short-stack play; "watch" shows
  each move's probabilities in the side panel.

**Tests running live:** don't screenshot a terminal and don't replay saved output. Serve a small page that runs the
real commands (over ssh to the lab if needed) and streams their output as Server-Sent Events, and record that page.
- Show a title and a one-line "what this tests" for each step, the command as typed, its real output, and a
  RESULT line.
- Filter `/home/<user>` to `~` in the stream.

## 3. Convert, check, clean up

```sh
ffmpeg -loglevel error -y -i OUT/*.webm -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart <name>.mp4
```

- **Look at frames before sharing:** take a mid-run and a final screenshot, and read them. Look for the lab address
  or login, tokens (`?token=`), API keys, personal emails, and terminal prompts that show user@host. If any appear,
  fix the source and record again. Blurring is a last resort.
- Stop the local servers by their ports' PIDs (`ss -ltnp | grep :18891`).
- Tell the user each file's name, length and what it shows, step by step. Offer a one-line caption to send with it.
