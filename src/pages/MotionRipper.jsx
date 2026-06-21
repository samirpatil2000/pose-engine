import React, { useEffect, useRef, useState } from 'react';
import './MotionRipper.css';

const EXTERNAL_SCRIPTS = [
  ['motion-ripper-tailwind', 'https://cdn.tailwindcss.com'],
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
  const mountRef = useRef(null);
  const scriptRef = useRef(null);
  const [status, setStatus] = useState('Loading Motion Ripper...');

  useEffect(() => {
    let cancelled = false;

    async function mountMotionRipper() {
      try {
        await loadExternalScripts();
        const response = await fetch('/motion-ripper/markup.html');
        if (!response.ok) throw new Error('Motion Ripper markup failed to load');

        const markup = await response.text();
        if (cancelled || !mountRef.current) return;

        mountRef.current.innerHTML = markup;

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
      if (mountRef.current) mountRef.current.innerHTML = '';
      document.body.classList.remove('motion-ripper-active');
    };
  }, []);

  return (
    <main className="motion-ripper-page">
      {status && (
        <div className="motion-ripper-loading" role="status">
          {status}
        </div>
      )}
      <div
        ref={mountRef}
        className="motion-ripper-host text-slate-100 antialiased"
      />
    </main>
  );
}
