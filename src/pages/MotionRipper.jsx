import React, { useEffect, useRef, useState } from 'react';
import './MotionRipper.css';

const EXTERNAL_SCRIPTS = [
  ['motion-ripper-tfjs-core', 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core'],
  ['motion-ripper-tfjs-converter', 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter'],
  ['motion-ripper-tfjs-backend-webgl', 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl'],
  ['motion-ripper-pose-detection', 'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection'],
];

function loadScript(id, src) {
  const existing = document.getElementById(id);
  if (existing?.dataset.loaded === 'true') return Promise.resolve();

  return new Promise((resolve, reject) => {
    const script = existing || document.createElement('script');
    script.id = id;
    script.src = src;
    script.async = false;

    script.addEventListener('load', () => {
      script.dataset.loaded = 'true';
      resolve();
    }, { once: true });
    script.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });

    if (!existing) document.head.appendChild(script);
  });
}

async function loadExternalScripts() {
  for (const [id, src] of EXTERNAL_SCRIPTS) {
    await loadScript(id, src);
  }
}

export default function MotionRipper() {
  const scriptRef = useRef(null);
  const [status, setStatus] = useState('Loading Motion Ripper...');

  useEffect(() => {
    let cancelled = false;

    async function mountMotionRipper() {
      try {
        await loadExternalScripts();
        if (cancelled) return;

        const script = document.createElement('script');
        script.type = 'module';
        script.src = `/motion-ripper/ripper.js?v=${Date.now()}`;
        scriptRef.current = script;
        script.addEventListener('error', () => setStatus('Motion Ripper failed to start'), { once: true });
        document.body.appendChild(script);
        setStatus('');
      } catch (error) {
        console.error(error);
        if (!cancelled) setStatus('Motion Ripper failed to load');
      }
    }

    document.body.classList.add('motion-ripper-active');
    mountMotionRipper();

    return () => {
      cancelled = true;
      window.__motionRipperCleanup?.();
      delete window.__motionRipperCleanup;
      scriptRef.current?.remove();
      scriptRef.current = null;
      document.body.classList.remove('motion-ripper-active');
    };
  }, []);

  return (
    <main className="motion-ripper-page">
      {status && (
        <div className="motion-ripper-loading" role="status">
          <div className="flex flex-col items-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-cyan-400 border-t-transparent"></div>
            <p className="text-sm font-medium text-slate-300">{status}</p>
          </div>
        </div>
      )}
      <div className="motion-ripper-host text-slate-100 antialiased">
        <div className="relative min-h-screen p-4 lg:p-6">
          <div className="motion-grid grid h-[calc(100vh-2rem)] grid-cols-1 gap-4 xl:h-[calc(100vh-3rem)] xl:grid-cols-[22rem,minmax(0,1fr)]">
            <aside className="glass-card flex min-h-0 flex-col rounded-[1.6rem] p-6">
              {/* Header */}
              <div className="mb-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300/80">Motion Capture</p>
                <h1 className="mt-1.5 text-2xl font-semibold text-white">Motion Ripper</h1>
              </div>

              {/* Status Banner */}
              <div className="mb-6 rounded-2xl bg-black/30 border border-white/5 p-4 flex items-start gap-3">
                <div className="mt-1.5 h-2 w-2 rounded-full bg-cyan-400 animate-pulse flex-shrink-0"></div>
                <div id="status-text" className="text-xs text-slate-300 leading-relaxed font-medium">
                  Initializing and warming up AI models...
                </div>
              </div>

              <div className="flex-1 overflow-y-auto space-y-6 pr-1">
                {/* 1. Capture Source */}
                <div className="rounded-2xl">
                  <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400/90 mb-4 flex items-center justify-between">
                    1. Capture Source
                    <span id="video-meta" className="text-[10px] font-semibold text-cyan-400 bg-cyan-400/10 px-2 py-0.5 rounded-full">No stream</span>
                  </h2>
                  <div className="grid-2col gap-3">
                    <button id="upload-video-btn">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                      Upload Video
                    </button>
                    <button id="share-screen-btn">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
                      Share Screen
                    </button>
                  </div>
                  <button id="stop-share-btn" className="hidden mt-3">
                    Stop Screen Share
                  </button>
                  <input id="video-file-input" type="file" accept="video/*" className="hidden" />
                </div>

                {/* 2. Record Animation */}
                <div className="rounded-2xl">
                  <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400/90 mb-4 flex items-center justify-between">
                    2. Record Animation
                    <span id="recording-badge" className="rounded-full">Idle</span>
                  </h2>
                  
                  <div className="space-y-4">
                    <div>
                      <span className="block text-[10px] font-medium uppercase tracking-[0.08em] text-slate-500 mb-1.5">Animation Name</span>
                      <input 
                        id="animation-name" 
                        type="text" 
                        placeholder="animation_name" 
                        defaultValue="motion_capture"
                      />
                    </div>
                    
                    <div className="grid-2col gap-3">
                      <button id="neutral-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                        Calibrate Hips
                      </button>
                      <button id="record-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3" fill="currentColor"></circle></svg>
                        Start Record
                      </button>
                    </div>

                    <div className="grid-2col gap-3">
                      <button id="clear-btn">
                        Clear
                      </button>
                      <button id="save-library-btn">
                        Save Library
                      </button>
                    </div>

                    <button id="export-html-btn" className="w-full mt-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                      Export Standalone HTML
                    </button>

                    {/* Stats Grid */}
                    <div className="mt-4 grid-2col border-t border-white/5 pt-4 text-[11px]">
                      <div className="flex justify-between text-slate-400 border-r border-white/5 pr-3">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Tracked</span>
                        <span id="tracked-state" className="font-semibold text-slate-300">Waiting</span>
                      </div>
                      <div className="flex justify-between text-slate-400 pl-3">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Confidence</span>
                        <span id="confidence-value" className="font-semibold text-slate-300">0%</span>
                      </div>
                      <div className="flex justify-between text-slate-400 border-r border-white/5 pr-3 pt-2">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Frames</span>
                        <span id="frame-count" className="font-semibold text-slate-300">0</span>
                      </div>
                      <div className="flex justify-between text-slate-400 pl-3 pt-2">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Duration</span>
                        <span id="duration-value" className="font-semibold text-slate-300">0.0s</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 3. Capture Options */}
                <div className="rounded-2xl">
                  <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400/90 mb-4">
                    3. Capture Options
                  </h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-3 rounded-xl border border-white/[0.04] bg-black/20 px-3.5 py-2.5">
                      <span className="text-xs font-medium text-slate-200">Multi-character tracking</span>
                      <input id="multi-character" type="checkbox" />
                    </div>

                    <div className="grid-2col gap-3">
                      <div>
                        <span className="mb-1.5 block text-[10px] font-medium uppercase tracking-wider text-slate-500">Char Color 1</span>
                        <input id="character-color" type="color" defaultValue="#5eead4" />
                      </div>
                      <div id="second-character-color-field" className="hidden">
                        <span className="mb-1.5 block text-[10px] font-medium uppercase tracking-wider text-slate-500">Char Color 2</span>
                        <input id="second-character-color" type="color" defaultValue="#a855f7" disabled />
                      </div>
                    </div>

                    <div className="grid-2col gap-3">
                      <div>
                        <span className="mb-1.5 block text-[10px] font-medium uppercase tracking-wider text-slate-500">Sample Rate</span>
                        <select id="sample-rate">
                          <option value="10">10 fps</option>
                          <option value="5">5 fps</option>
                        </select>
                      </div>
                      <div>
                        <span className="mb-1.5 block text-[10px] font-medium uppercase tracking-wider text-slate-500">Pose Backend</span>
                        <select id="pose-backend">
                          <option value="mediapipe-video">MediaPipe Video</option>
                          <option value="mediapipe-image">MediaPipe Image</option>
                          <option value="movenet">MoveNet</option>
                          <option value="manual">Manual Assist</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <div className="mb-1.5 flex items-center justify-between">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Pose Smoothing</span>
                        <span id="smoothing-value" className="text-xs font-semibold text-cyan-400">0.55</span>
                      </div>
                      <input id="smoothing" type="range" min="0" max="0.85" step="0.05" defaultValue="0.55" />
                    </div>

                    <label id="root-motion-field" className="flex items-start gap-3 rounded-xl border border-white/[0.04] bg-black/20 p-3 cursor-pointer transition hover:bg-black/30">
                      <input id="root-motion" type="checkbox" defaultChecked className="mt-0.5" />
                      <span>
                        <span className="block text-xs font-medium text-slate-200">Track root motion</span>
                        <span className="mt-1 block text-[10px] leading-relaxed text-slate-500">Moves hips in X/Y/Z. Turn off for static hips.</span>
                      </span>
                    </label>

                    <label className="flex items-start gap-3 rounded-xl border border-white/[0.04] bg-black/20 p-3 cursor-pointer transition hover:bg-black/30">
                      <input id="upper-body-only" type="checkbox" className="mt-0.5" />
                      <span>
                        <span className="block text-xs font-medium text-slate-200">Upper-body source</span>
                        <span className="mt-1 block text-[10px] leading-relaxed text-slate-500">Disables hip/leg tracking. Plants feet.</span>
                      </span>
                    </label>

                    <div className="rounded-xl border border-white/[0.04] bg-black/20 p-3">
                      <label className="flex items-start gap-3 cursor-pointer">
                        <input id="focus-crop" type="checkbox" className="mt-0.5" />
                        <span>
                          <span className="block text-xs font-medium text-slate-200">Focus crop</span>
                          <span className="mt-1 block text-[10px] leading-relaxed text-slate-500">Drag feed overlay to crop the tracked area.</span>
                        </span>
                      </label>
                      <button id="reset-crop-btn" className="mt-3 w-full rounded-xl border border-white/10 bg-white/5 px-2 py-2 text-[11px] font-medium text-slate-300 hover:bg-white/10 hover:text-white transition cursor-pointer" type="button">
                        Reset Crop
                      </button>
                    </div>
                  </div>
                </div>

                {/* 4. Manual Assist Panel */}
                <div id="manual-assist-panel" className="hidden rounded-2xl p-4">
                  <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-amber-400 mb-4 flex items-center justify-between">
                    Manual Assist
                    <span className="text-[10px] font-semibold text-amber-400 bg-amber-400/15 px-2 py-0.5 rounded-full">
                      Keys: <span id="assist-key-count">0</span>
                    </span>
                  </h2>
                  
                  <div className="space-y-4">
                    <div>
                      <span className="block text-[10px] font-medium uppercase tracking-wider text-slate-500 mb-1.5">Target Character</span>
                      <select id="assist-character">
                        <option value="0">Character 1</option>
                        <option value="1">Character 2</option>
                      </select>
                    </div>

                    <div className="grid-2col gap-3">
                      <button id="add-assist-key-btn">
                        Add Key
                      </button>
                      <button id="clear-assist-keys-btn">
                        Clear Keys
                      </button>
                    </div>

                    <button id="reset-assist-btn">
                      Reset Assist
                    </button>
                  </div>
                </div>
              </div>
            </aside>

            {/* Main Previews */}
            <main className="grid min-h-0 grid-cols-1 gap-4 2xl:grid-cols-[1.08fr,0.92fr]">
              {/* Pose Feed */}
              <section className="glass-card flex min-h-0 flex-col rounded-[1.8rem] p-5">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Shared Video</p>
                    <h2 className="mt-1 text-lg font-semibold text-white">Pose Feed</h2>
                  </div>
                </div>
                <div id="video-stage" className="min-h-0 flex-1 rounded-[1.5rem] border border-white/10 bg-slate-950/90 relative aspect-video">
                  <video id="source-video" autoPlay muted playsInline className="absolute inset-0 w-full h-full object-contain"></video>
                  <canvas id="pose-overlay" className="absolute inset-0 w-full h-full pointer-events-none"></canvas>
                </div>
              </section>

              {/* Mapped Character */}
              <section className="glass-card flex min-h-0 flex-col rounded-[1.8rem] p-5">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Rig Preview</p>
                    <h2 className="mt-1 text-lg font-semibold text-white">Mapped Character</h2>
                  </div>
                </div>
                <div className="mb-4">
                  <label className="block">
                    <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">Character Model</span>
                    <select id="character-model">
                      <option value="box">Box Character</option>
                      <option value="lady-x-bot">Lady-X Bot</option>
                    </select>
                  </label>
                </div>
                <div className="min-h-0 flex-1 overflow-hidden rounded-[1.5rem] border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(34,197,94,0.16),_transparent_36%),linear-gradient(180deg,_rgba(15,23,42,0.95),_rgba(2,6,23,1))] relative">
                  <canvas id="preview-canvas" className="absolute inset-0 w-full h-full"></canvas>
                </div>
              </section>
            </main>
          </div>
        </div>
      </div>
    </main>
  );
}
