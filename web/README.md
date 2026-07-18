# Answer — demo site

A static, offline **replay** of one frozen run of the Answer RAG pipeline. Pick a
question → press **Ask** → the pipeline plays back stage by stage (plan → search →
read → index → rank → judge → refine-loop → answer → cite), driven entirely by
pre-captured data. No backend, no network, no model calls.

Built with **Vite + React + TypeScript** (plain CSS design tokens, no UI framework).

## Data

The site's single input is `src/run.json`, assembled from the frozen `eval/fixtures/*`
by the Python tool. Re-run it whenever a fixture changes:

```bash
# from the repo root
uv run python eval/build_site_data.py     # writes web/src/run.json
```

## Run it

Node is required. It works from **Windows** (`node`/`npm` on PATH) or **WSL** — the
dev config polls for file changes so hot-reload is reliable in either environment
(WSL's inotify can't see edits on the `/mnt/*` Windows mount).

> Note: `node_modules` holds platform-native binaries (esbuild/rollup). If you
> switch between Windows and WSL, delete `node_modules` and re-run `npm install`
> in the target environment.

```bash
cd web
npm install
npm run dev        # dev server with hot-reload — opens http://localhost:5173
```

Other scripts:

```bash
npm run build      # type-check + bundle to web/dist/ (the deployable static site)
npm run preview    # serve the built web/dist/ locally to sanity-check the build
```

## Deploy

`npm run build` produces a fully self-contained `web/dist/` (data bundled in). The
Vite `base` is relative (`"./"`), so it works under any GitHub Pages sub-path
(`username.github.io/<repo>/`) with no config. A GitHub Actions workflow can build
and publish it on push.
