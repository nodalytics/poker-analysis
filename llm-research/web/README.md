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

## Public address: https://poker.stbfarm.org (`deploy/`)

| File | What it is |
|---|---|
| `deploy/nginx/poker.stbfarm.org.conf` | Host nginx vhost: HTTPS on 443 with the lab's origin certificate (Cloudflare SSL mode Full, as for the other stbfarm.org sites), HTTP redirected to HTTPS, proxied to the app on 127.0.0.1:8790; GET only; **only Cloudflare's ranges (and the lab itself) may connect**, others get 403; the visitor's real address is taken from `CF-Connecting-IP`, trusted only from those ranges, so `/api/live` is limited to 20 requests a minute **per visitor**, not per Cloudflare server; an optional password (commented) |
| `deploy/poker-watch.service` | systemd service running `scripts/ninemax_watch.py` as labuser, restarted on failure and started at boot |
| `deploy/install.sh` | Installs both (run once with sudo); `--remove` takes them out again |

The app stays bound to 127.0.0.1; only nginx faces the network. Cloudflare's ranges are listed in the vhost (fetched
2026-11-10 from https://www.cloudflare.com/ips-v4 and ips-v6); refresh them if Cloudflare adds ranges. DNS: `poker.stbfarm.org` is a Cloudflare-proxied
record pointing at the lab.

Install (needs the sudo password, so run it yourself in a terminal):

```
ssh -t -i lab.pem labuser@LAB_HOST 'cd ~/llm-research && sudo sh web/deploy/install.sh'
```

The vhost was syntax-checked with the lab's nginx 1.24 (`nginx -t` against a throwaway certificate) before the
first install.

**Who can see it:** without a password or Cloudflare Access, anyone with the address. The page shows training
curves and sample hands, nothing secret, but it does show how the bot plays. To restrict it, either add a
Cloudflare Access application for poker.stbfarm.org (Zero Trust, email one-time code), or enable the nginx
password in the vhost.
