# web

The 9-max watch page: `index.html`, one self-contained file (plain HTML, CSS and JavaScript, no build step, no
external libraries). It is served, with its data, by `scripts/ninemax_watch.py` on the lab:

```
PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_watch.py        # on the lab, from llm-research
ssh -N -L 8790:127.0.0.1:8790 -i lab.pem labuser@LAB_HOST      # on your machine
```

then open http://localhost:8790. Views: training charts and checkpoint hand replays (default), real tables and
live hands (tabs). Details: `docs/2026-11-09-nine-max-plan.md`, "Watching it learn and play".

## Why there is no Vite (yet)

Vite is a build tool: it bundles an app written as many source files and npm packages (React, Vue, chart
libraries, TypeScript) into the files a browser runs, and serves them with hot reload while you develop. This page
needs none of that today: it is a single file with plain JavaScript, its charts are hand-drawn SVG, and it uses no
packages and no internet. The browser runs it as it is, and `scripts/ninemax_watch.py` serves it.

**When to move to Vite:** when the page outgrows one file, for example several views split into components, a
chart or table library, TypeScript, or a design system.

**How the move would go** (nothing on the server changes; it keeps the same `/api/...` endpoints):

1. Create the project here: `npm create vite@latest . -- --template react-ts` (or `vanilla-ts`), then `npm install`.
2. Develop with `npm run dev`, proxying the API to the lab server through the tunnel, in `vite.config.ts`:
   ```ts
   export default defineConfig({ server: { proxy: { "/api": "http://localhost:8790" } } });
   ```
3. Build with `npm run build`; the result lands in `web/dist/`.
4. Point the server at the build: in `scripts/ninemax_watch.py`, serve `web/dist/index.html` and the files under
   `web/dist/assets/` instead of `web/index.html`.
5. Keep `node_modules/` and `dist/` out of git; commit `package.json` and `package-lock.json`.

The lab then needs Node only to build, not to serve: the built files are static.
