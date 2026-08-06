import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so /browse file:///samples/html/react-tw.html can load
// ./react-tw.js + ./react-tw.css without an HTTP origin.
// IIFE (not ES modules): the in-OS browser runs scripts as a flat program list;
// classic <script src> is the reliable path.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: ".",
    cssCodeSplit: false,
    modulePreload: false,
    rollupOptions: {
      output: {
        format: "iife",
        name: "ChittiReactTw",
        inlineDynamicImports: true,
        entryFileNames: "react-tw.js",
        assetFileNames: "react-tw.[ext]",
      },
    },
  },
});
