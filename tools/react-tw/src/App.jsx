import { useState } from "react";

function Check({ label, ok }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 py-2 last:border-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <span
        className={
          ok
            ? "rounded-full border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-800"
            : "rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-800"
        }
      >
        {ok ? "loaded" : "waiting"}
      </span>
    </div>
  );
}

export default function App() {
  const [clicks, setClicks] = useState(0);

  // Reaching this component means the React bundle executed.
  const reactOk = true;
  // Bare document lookups — avoid `typeof document` (always defined here).
  const rootOk = !!document.getElementById("root");
  // Avoid attribute `*=` selectors — the in-OS engine only matches tag/#id/.class.
  const link = document.querySelector("link");
  const href =
    link && (link.getAttribute ? link.getAttribute("href") : link.href);
  const cssOk = !!(href && String(href).indexOf("react-tw.css") >= 0);
  const all = reactOk && rootOk && cssOk;

  if (all) {
    console.log("react-tw ALL PASS");
  } else {
    console.log(
      "react-tw FAIL react=" + reactOk + " css=" + cssOk + " root=" + rootOk
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 shadow-lg">
        <p className="mb-1 text-xs font-semibold text-brand">ChittiOS /samples</p>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          React + Tailwind
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          Built with Vite + React 18 + Tailwind 3 (
          <code className="font-mono text-xs">tools/react-tw</code>
          ). Dist copied into <code className="font-mono text-xs">/samples/html</code>.
        </p>

        <div className="mt-4">
          <Check label="React app mounted" ok={reactOk} />
          <Check label="Tailwind CSS linked" ok={cssOk} />
          <Check label="#root present" ok={rootOk} />
        </div>

        <div className="mt-4 flex items-center justify-between gap-3">
          <button
            type="button"
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-700"
            onClick={() => setClicks((n) => n + 1)}
          >
            Count {clicks}
          </button>
          <a
            href="index.html"
            className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-sky-700 hover:bg-slate-50"
          >
            ← samples
          </a>
        </div>

        <p
          id="status"
          className={
            "mt-4 text-center font-mono text-sm " +
            (all ? "text-emerald-800" : "text-amber-800")
          }
        >
          {all ? "react-tw ALL PASS" : "react-tw loading…"}
        </p>
      </div>
    </div>
  );
}
