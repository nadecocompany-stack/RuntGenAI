import React, { useState, useRef, useCallback } from "react";
import {
  Upload, ScanLine, Brain, Wind, Activity, Radio, ShieldAlert,
  Crosshair, Layers, ChevronRight, X, FileType2, Zap,
} from "lucide-react";

/* ------------------------------------------------------------------ *
 * Reading-room palette. Custom hexes go through inline styles because
 * the artifact runtime only ships Tailwind's base utilities (no JIT for
 * arbitrary values). Tailwind handles layout; style handles color.
 * ------------------------------------------------------------------ */
const C = {
  bg: "#0c141d", panel: "#131f2b", panelHi: "#1a2836", line: "#24384a",
  ink: "#e8eef4", mute: "#8296a8", faint: "#5b6b7c",
  teal: "#3fd0c9", tealDim: "#173f43", amber: "#f6b13c", amberDim: "#3d2e17",
  good: "#4ec9a8",
};

/* Taxonomy mirrors src/label_taxonomies.py so simulated output is accurate. */
const CATEGORIES = [
  {
    id: "ct_chest", icon: ScanLine, label: "CT · Chest",
    modelKey: "ct_chest_nodule_seg", dataset: "LIDC-IDRI",
    task: "segmentation", loc: "mask", unit: "voxels",
    findings: ["nodule"],
  },
  {
    id: "ct_head", icon: Activity, label: "CT · Head",
    modelKey: "ct_head_ich_classifier", dataset: "RSNA ICH",
    task: "classification", loc: "grad-cam", unit: "region",
    findings: ["epidural", "intraparenchymal", "intraventricular",
               "subarachnoid", "subdural"],
  },
  {
    id: "mr_brain", icon: Brain, label: "MRI · Brain",
    modelKey: "mr_brain_tumor_seg", dataset: "BraTS",
    task: "segmentation", loc: "mask", unit: "voxels",
    findings: ["whole tumor", "tumor core", "enhancing tumor"],
  },
  {
    id: "us_breast", icon: Radio, label: "Ultrasound · Breast",
    modelKey: "us_breast_lesion", dataset: "BUSI",
    task: "detection", loc: "mask", unit: "pixels",
    findings: ["benign", "malignant"],
  },
  {
    id: "cxr", icon: Wind, label: "X-ray · Chest",
    modelKey: "cxr_multilabel_classifier", dataset: "CheXpert / MIMIC-CXR",
    task: "classification", loc: "grad-cam", unit: "region",
    findings: ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia",
               "Atelectasis", "Pneumothorax", "Pleural Effusion", "Lung Opacity"],
  },
];

const rand = (a, b) => a + Math.random() * (b - a);
const pick = (arr, n) =>
  [...arr].sort(() => Math.random() - 0.5).slice(0, n);

/* Client-side stand-in for POST /analyze. Uses the category's real findings
 * so the shape + labels match the backend; confidences/boxes are illustrative. */
function simulate(cat) {
  const n = 1 + Math.floor(rand(0, Math.min(3, cat.findings.length)));
  const chosen = pick(cat.findings, n);
  const dets = chosen.map((f) => {
    const w = rand(14, 30), h = rand(14, 30);
    const x = rand(8, 92 - w), y = rand(8, 92 - h);
    return {
      label: f,
      confidence: +rand(0.61, 0.96).toFixed(3),
      box: { x, y, w, h },
      size: Math.round(rand(120, 5200)),
    };
  }).sort((a, b) => b.confidence - a.confidence);
  return { detections: dets, top: dets[0] };
}

export default function RadiologyIntake() {
  const [catId, setCatId] = useState(null);
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | scanning | done
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const cat = CATEGORIES.find((c) => c.id === catId) || null;
  const armed = !!cat;
  const canRun = armed && files.length > 0 && status !== "scanning";

  const addFiles = useCallback((list) => {
    const items = Array.from(list).map((f) => ({
      name: f.name, size: f.size,
    }));
    setFiles((prev) => [...prev, ...items]);
    setResult(null); setStatus("idle");
  }, []);

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    if (armed && e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const run = () => {
    if (!canRun) return;
    setStatus("scanning"); setResult(null); setHeatmap(false);
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    setTimeout(() => {
      setResult(simulate(cat));
      setStatus("done");
    }, reduce ? 120 : 1400);
  };

  const reset = (nextCat) => {
    setCatId(nextCat); setFiles([]); setResult(null);
    setStatus("idle"); setHeatmap(false);
  };

  return (
    <div className="min-h-screen w-full" style={{ background: C.bg, color: C.ink }}>
      <style>{`
        .rd-focus:focus-visible { outline: none; box-shadow: 0 0 0 2px ${C.bg}, 0 0 0 3px ${C.teal}; }
        @keyframes rd-sweep { 0%{top:0;opacity:0} 8%{opacity:1} 92%{opacity:1} 100%{top:100%;opacity:0} }
        @keyframes rd-box { from{opacity:0;transform:scale(.92)} to{opacity:1;transform:scale(1)} }
        @keyframes rd-pulse { 0%,100%{opacity:.55} 50%{opacity:1} }
        .rd-scanline{ animation: rd-sweep 1.4s ease-in-out; }
        .rd-boxin{ animation: rd-box .5s ease-out both; }
        .rd-live{ animation: rd-pulse 1.6s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce){
          .rd-scanline,.rd-boxin,.rd-live{ animation: none !important; }
        }
      `}</style>

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4"
        style={{ borderBottom: `1px solid ${C.line}` }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center rounded-md" style={{
            width: 34, height: 34, background: C.tealDim,
            border: `1px solid ${C.line}` }}>
            <Crosshair size={18} style={{ color: C.teal }} />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">Radiology Intake</div>
            <div className="font-mono text-xs" style={{ color: C.faint }}>
              abnormality detection · multi-modal
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 rounded-full px-3 py-1"
          style={{ background: C.amberDim, border: `1px solid ${C.line}` }}>
          <ShieldAlert size={14} style={{ color: C.amber }} />
          <span className="font-mono text-xs" style={{ color: C.amber }}>
            Research prototype — not for clinical use
          </span>
        </div>
      </header>

      <div className="grid gap-5 p-5 lg:grid-cols-12">
        {/* ---------------- Category column ---------------- */}
        <section className="lg:col-span-4 xl:col-span-3">
          <SectionLabel n="01" text="Choose scan category" />
          <p className="mb-3 text-xs leading-relaxed" style={{ color: C.mute }}>
            The category selects the specialized model. Picking it here means
            detection never depends on missing or mislabeled DICOM tags.
          </p>
          <div className="flex flex-col gap-2">
            {CATEGORIES.map((c) => {
              const active = c.id === catId;
              const Icon = c.icon;
              return (
                <button key={c.id} onClick={() => reset(c.id)}
                  className="rd-focus text-left rounded-lg p-3 transition-colors"
                  style={{
                    background: active ? C.panelHi : C.panel,
                    border: `1px solid ${active ? C.teal : C.line}`,
                  }}>
                  <div className="flex items-center gap-3">
                    <Icon size={18} style={{ color: active ? C.teal : C.mute }} />
                    <div className="flex-1">
                      <div className="text-sm font-medium">{c.label}</div>
                      <div className="font-mono text-xs" style={{ color: C.faint }}>
                        {c.dataset} · {c.task}
                      </div>
                    </div>
                    {active && <ChevronRight size={16} style={{ color: C.teal }} />}
                  </div>
                  {active && (
                    <div className="mt-3 flex flex-wrap gap-1">
                      {c.findings.map((f) => (
                        <span key={f} className="font-mono rounded px-1.5 py-0.5"
                          style={{ fontSize: 10.5, background: C.bg,
                            color: C.mute, border: `1px solid ${C.line}` }}>
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          {/* Armed status */}
          <div className="mt-4 rounded-lg p-3" style={{
            background: C.panel, border: `1px solid ${armed ? C.tealDim : C.line}` }}>
            <div className="flex items-center gap-2">
              <span className={armed ? "rd-live" : ""} style={{
                width: 8, height: 8, borderRadius: 99,
                background: armed ? C.teal : C.faint, display: "inline-block" }} />
              <span className="font-mono text-xs tracking-wide"
                style={{ color: armed ? C.teal : C.faint }}>
                {armed ? "PIPELINE ARMED" : "PIPELINE IDLE"}
              </span>
            </div>
            <div className="mt-2 font-mono text-xs" style={{ color: C.mute }}>
              {armed ? `${cat.modelKey} · loc=${cat.loc}` : "select a category to arm"}
            </div>
          </div>
        </section>

        {/* ---------------- Intake + viewport column ---------------- */}
        <section className="lg:col-span-8 xl:col-span-9">
          <SectionLabel n="02" text="Upload & analyze" />

          {/* Dropzone */}
          <div
            onDragOver={(e) => { e.preventDefault(); if (armed) setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className="rounded-lg p-5 transition-colors"
            style={{
              background: C.panel,
              border: `1px dashed ${dragging ? C.teal : C.line}`,
              opacity: armed ? 1 : 0.55,
            }}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center rounded-md"
                  style={{ width: 40, height: 40, background: C.bg,
                    border: `1px solid ${C.line}` }}>
                  <Upload size={18} style={{ color: armed ? C.teal : C.faint }} />
                </div>
                <div>
                  <div className="text-sm font-medium">
                    {armed ? "Drop DICOM series or .zip" : "Choose a category first"}
                  </div>
                  <div className="font-mono text-xs" style={{ color: C.faint }}>
                    {armed ? "single frame, multi-slice series, or zipped study"
                           : "upload unlocks once a model is armed"}
                  </div>
                </div>
              </div>
              <button
                disabled={!armed}
                onClick={() => inputRef.current?.click()}
                className="rd-focus rounded-md px-3 py-2 text-sm font-medium"
                style={{
                  background: armed ? C.tealDim : C.panelHi,
                  color: armed ? C.teal : C.faint,
                  border: `1px solid ${armed ? C.teal : C.line}`,
                  cursor: armed ? "pointer" : "not-allowed",
                }}>
                Browse files
              </button>
              <input ref={inputRef} type="file" multiple hidden
                onChange={(e) => e.target.files && addFiles(e.target.files)} />
            </div>

            {files.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {files.map((f, i) => (
                  <span key={i} className="flex items-center gap-2 rounded px-2 py-1"
                    style={{ background: C.bg, border: `1px solid ${C.line}` }}>
                    <FileType2 size={13} style={{ color: C.mute }} />
                    <span className="font-mono text-xs" style={{ color: C.ink }}>
                      {f.name}
                    </span>
                    <button className="rd-focus" aria-label={`Remove ${f.name}`}
                      onClick={() => setFiles(files.filter((_, j) => j !== i))}>
                      <X size={13} style={{ color: C.faint }} />
                    </button>
                  </span>
                ))}
              </div>
            )}

            <div className="mt-4 flex items-center gap-3">
              <button onClick={run} disabled={!canRun}
                className="rd-focus flex items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold"
                style={{
                  background: canRun ? C.teal : C.panelHi,
                  color: canRun ? C.bg : C.faint,
                  cursor: canRun ? "pointer" : "not-allowed",
                }}>
                <Zap size={15} />
                {status === "scanning" ? "Analyzing…" : "Run analysis"}
              </button>
              {status === "scanning" && (
                <span className="font-mono text-xs rd-live" style={{ color: C.teal }}>
                  routing → {cat.modelKey}
                </span>
              )}
            </div>
          </div>

          {/* Viewport + results */}
          <div className="mt-5 grid gap-5 md:grid-cols-2">
            {/* Viewport */}
            <div className="rounded-lg p-3" style={{
              background: C.panel, border: `1px solid ${C.line}` }}>
              <div className="mb-2 flex items-center justify-between">
                <span className="font-mono text-xs" style={{ color: C.mute }}>
                  VIEWPORT {cat ? `· ${cat.label}` : ""}
                </span>
                {result && (
                  <button onClick={() => setHeatmap(!heatmap)}
                    className="rd-focus flex items-center gap-1.5 rounded px-2 py-1"
                    style={{ background: heatmap ? C.amberDim : C.bg,
                      border: `1px solid ${heatmap ? C.amber : C.line}` }}>
                    <Layers size={12} style={{ color: heatmap ? C.amber : C.mute }} />
                    <span className="font-mono" style={{ fontSize: 10.5,
                      color: heatmap ? C.amber : C.mute }}>Grad-CAM</span>
                  </button>
                )}
              </div>

              <div className="relative overflow-hidden rounded-md"
                style={{ aspectRatio: "1 / 1", background: "#070c12",
                  border: `1px solid ${C.line}` }}>
                {/* faint grid */}
                <div className="absolute inset-0" style={{
                  backgroundImage:
                    `linear-gradient(${C.line} 1px, transparent 1px),
                     linear-gradient(90deg, ${C.line} 1px, transparent 1px)`,
                  backgroundSize: "12.5% 12.5%", opacity: 0.18 }} />
                {/* synthetic anatomy glow */}
                <div className="absolute" style={{
                  inset: "18%", borderRadius: "50%",
                  background: `radial-gradient(circle at 45% 40%, ${C.panelHi}, transparent 70%)`,
                  opacity: 0.9 }} />

                {status === "idle" && !result && (
                  <Centered>
                    <ScanLine size={26} style={{ color: C.faint }} />
                    <p className="mt-2 font-mono text-xs" style={{ color: C.faint }}>
                      {armed ? "awaiting scan" : "no model armed"}
                    </p>
                  </Centered>
                )}
                {status === "scanning" && (
                  <div className="rd-scanline absolute left-0 right-0" style={{
                    height: 2, background: C.teal,
                    boxShadow: `0 0 12px 2px ${C.teal}` }} />
                )}

                {/* Grad-CAM overlay */}
                {result && heatmap && result.detections.map((d, i) => (
                  <div key={`h${i}`} className="absolute" style={{
                    left: `${d.box.x - 6}%`, top: `${d.box.y - 6}%`,
                    width: `${d.box.w + 12}%`, height: `${d.box.h + 12}%`,
                    background: `radial-gradient(circle, ${C.amber}, transparent 68%)`,
                    opacity: 0.5, filter: "blur(2px)" }} />
                ))}

                {/* Detection boxes */}
                {result && result.detections.map((d, i) => (
                  <div key={`b${i}`} className="rd-boxin absolute" style={{
                    left: `${d.box.x}%`, top: `${d.box.y}%`,
                    width: `${d.box.w}%`, height: `${d.box.h}%`,
                    border: `1.5px solid ${C.amber}`, borderRadius: 3,
                    animationDelay: `${i * 0.12}s` }}>
                    <span className="absolute font-mono whitespace-nowrap"
                      style={{ top: -16, left: -1, fontSize: 10,
                        color: C.bg, background: C.amber, padding: "0 4px",
                        borderRadius: 2 }}>
                      {d.label} {(d.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 font-mono" style={{ fontSize: 10.5, color: C.faint }}>
                boxes in viewport coords · scale by patch/stride for pixel space
              </p>
            </div>

            {/* Results */}
            <div className="rounded-lg p-4" style={{
              background: C.panel, border: `1px solid ${C.line}` }}>
              <span className="font-mono text-xs" style={{ color: C.mute }}>
                FINDINGS
              </span>

              {!result && (
                <Empty armed={armed} />
              )}

              {result && (
                <div className="mt-3">
                  {/* top-line confidence */}
                  <div className="mb-4">
                    <div className="flex items-baseline justify-between">
                      <span className="text-sm font-semibold">
                        {result.top.label}
                      </span>
                      <span className="font-mono text-lg" style={{ color: C.teal }}>
                        {(result.top.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Gauge value={result.top.confidence} />
                    <div className="mt-1 font-mono" style={{ fontSize: 10.5,
                      color: C.faint }}>
                      calibrated confidence · {cat.task}
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    {result.detections.map((d, i) => (
                      <div key={i} className="rounded-md p-2.5"
                        style={{ background: C.bg, border: `1px solid ${C.line}` }}>
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-2 text-sm">
                            <span style={{ width: 7, height: 7, borderRadius: 99,
                              background: C.amber, display: "inline-block" }} />
                            {d.label}
                          </span>
                          <span className="font-mono text-xs" style={{ color: C.teal }}>
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-1.5 font-mono" style={{ fontSize: 10.5,
                          color: C.mute }}>
                          bbox x{d.box.x.toFixed(0)} y{d.box.y.toFixed(0)}
                          {" "}w{d.box.w.toFixed(0)} h{d.box.h.toFixed(0)}
                          {" · "}{d.size} {cat.unit}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 flex items-center gap-2 rounded-md p-2"
                    style={{ background: C.bg, border: `1px solid ${C.line}` }}>
                    <span className="font-mono" style={{ fontSize: 10.5,
                      color: C.faint }}>
                      model {cat.modelKey} · simulated inference
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* ----------------------------- bits ----------------------------- */
function SectionLabel({ n, text }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="font-mono text-xs" style={{ color: C.teal }}>{n}</span>
      <span className="text-sm font-semibold tracking-wide">{text}</span>
      <span className="flex-1" style={{ height: 1, background: C.line }} />
    </div>
  );
}

function Centered({ children }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
      {children}
    </div>
  );
}

function Empty({ armed }) {
  return (
    <div className="mt-6 flex flex-col items-center justify-center py-8 text-center">
      <Crosshair size={22} style={{ color: C.faint }} />
      <p className="mt-2 text-sm" style={{ color: C.mute }}>No analysis yet</p>
      <p className="mt-1 font-mono text-xs" style={{ color: C.faint }}>
        {armed ? "upload a scan and run analysis" : "arm a model to begin"}
      </p>
    </div>
  );
}

function Gauge({ value }) {
  const ticks = 24;
  const filled = Math.round(value * ticks);
  return (
    <div className="mt-2 flex gap-0.5">
      {Array.from({ length: ticks }).map((_, i) => (
        <span key={i} className="flex-1" style={{
          height: 8, borderRadius: 1,
          background: i < filled ? C.teal : C.line,
          opacity: i < filled ? (0.5 + 0.5 * (i / ticks)) : 1,
        }} />
      ))}
    </div>
  );
}
