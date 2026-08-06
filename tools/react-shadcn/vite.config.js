import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Same shape as tools/react-tw: relative base + a single IIFE bundle, because
// the in-OS browser runs a page's scripts as a flat program list from
// `file:///samples/html/…` — there is no module loader and no HTTP origin.
export default defineConfig({
  plugins: [react()],
  base: "./",
  resolve: { alias: { "@": path.resolve(process.cwd(), "src") } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: ".",
    cssCodeSplit: false,
    modulePreload: false,
    rollupOptions: {
      output: {
        format: "iife",
        name: "ChittiShadcn",
        inlineDynamicImports: true,
        entryFileNames: "shadcn.js",
        assetFileNames: "shadcn.[ext]",
      },
    },
  },
});
