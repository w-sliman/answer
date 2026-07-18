import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the built site works under any GitHub Pages sub-path
// (e.g. user.github.io/<repo>/) without hard-coding the repo name.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    // On WSL, inotify can't see edits to files on the Windows /mnt/* mount,
    // so hot-reload silently misses them. Polling makes the watcher reliable.
    watch: { usePolling: true, interval: 200 },
  },
});
