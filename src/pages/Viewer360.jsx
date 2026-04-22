import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RotateCw, Upload, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './Viewer360.css';

export default function Viewer360() {
  const navigate = useNavigate();
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const sphereRef = useRef(null);
  const controlsRef = useRef(null);
  const animFrameRef = useRef(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const objectUrlRef = useRef(null);

  const handleClose = () => {
    // Prefer back navigation; fall back to Home.
    if (window.history.length > 1) navigate(-1);
    else navigate('/');
  };

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    try {
      // Scene setup
      const scene = new THREE.Scene();
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      
      const camera = new THREE.PerspectiveCamera(
        75,
        width / height,
        0.1,
        1000
      );
      camera.position.set(0, 0, 0);

      const renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        alpha: false,
        powerPreference: 'high-performance'
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setClearColor(0x000000);
      containerRef.current.appendChild(renderer.domElement);

      sceneRef.current = scene;
      cameraRef.current = camera;
      rendererRef.current = renderer;

      // Orbit controls state
      const controls = {
        theta: 0,
        phi: Math.PI / 2,
        fov: 75,
        isDragging: false,
        previousMousePosition: { x: 0, y: 0 },
      };
      controlsRef.current = controls;

      // Mouse events
      const onMouseDown = (e) => {
        controls.isDragging = true;
        controls.previousMousePosition = { x: e.clientX, y: e.clientY };
      };

      const onMouseMove = (e) => {
        if (!controls.isDragging) return;

        const deltaX = e.clientX - controls.previousMousePosition.x;
        const deltaY = e.clientY - controls.previousMousePosition.y;

        controls.theta -= deltaX * 0.005;
        controls.phi -= deltaY * 0.005;

        controls.phi = Math.max(0.1, Math.min(Math.PI - 0.1, controls.phi));
        controls.previousMousePosition = { x: e.clientX, y: e.clientY };
      };

      const onMouseUp = () => {
        controls.isDragging = false;
      };

      // Wheel zoom
      const onWheel = (e) => {
        e.preventDefault();
        const zoomSpeed = 0.05;
        const delta = e.deltaY > 0 ? 1 : -1;
        controls.fov += delta * zoomSpeed * controls.fov;
        controls.fov = Math.max(20, Math.min(120, controls.fov));
        camera.fov = controls.fov;
        camera.updateProjectionMatrix();
      };

      renderer.domElement.addEventListener('mousedown', onMouseDown);
      renderer.domElement.addEventListener('mousemove', onMouseMove);
      renderer.domElement.addEventListener('mouseup', onMouseUp);
      renderer.domElement.addEventListener('mouseleave', onMouseUp);
      renderer.domElement.addEventListener('wheel', onWheel, { passive: false });

      // Handle window resize
      const onWindowResize = () => {
        if (!containerRef.current) return;
        const w = containerRef.current.clientWidth;
        const h = containerRef.current.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', onWindowResize);

      // Animation loop
      const animate = () => {
        animFrameRef.current = requestAnimationFrame(animate);

        // Update camera position using spherical coordinates
        const radius = 0.01;
        const x = radius * Math.sin(controls.phi) * Math.cos(controls.theta);
        const y = radius * Math.cos(controls.phi);
        const z = radius * Math.sin(controls.phi) * Math.sin(controls.theta);
        
        camera.position.set(x, y, z);
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
      };
      animate();

      return () => {
        renderer.domElement.removeEventListener('mousedown', onMouseDown);
        renderer.domElement.removeEventListener('mousemove', onMouseMove);
        renderer.domElement.removeEventListener('mouseup', onMouseUp);
        renderer.domElement.removeEventListener('mouseleave', onMouseUp);
        renderer.domElement.removeEventListener('wheel', onWheel);
        window.removeEventListener('resize', onWindowResize);
        if (animFrameRef.current) {
          cancelAnimationFrame(animFrameRef.current);
        }
        if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
          containerRef.current.removeChild(renderer.domElement);
        }
        renderer.dispose();
      };
    } catch (err) {
      console.error('Three.js initialization error:', err);
      setError('Failed to initialize 3D viewer');
    }
  }, []);

  // ESC exits, like the reference UX
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // Load panorama image
  useEffect(() => {
    if (!imageUrl || !sceneRef.current || !cameraRef.current) return;

    setIsLoading(true);
    setError('');

    const textureLoader = new THREE.TextureLoader();
    
    textureLoader.load(
      imageUrl,
      (texture) => {
        try {
          // Remove old sphere if it exists
          if (sphereRef.current) {
            sceneRef.current.remove(sphereRef.current);
            sphereRef.current.geometry.dispose();
            sphereRef.current.material.map?.dispose();
            sphereRef.current.material.dispose();
          }

          // Create new panorama sphere with higher resolution
          const geometry = new THREE.SphereGeometry(100, 128, 64);
          const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.BackSide,
          });
          
          const sphere = new THREE.Mesh(geometry, material);
          sceneRef.current.add(sphere);
          sphereRef.current = sphere;

          setIsLoading(false);
          setError('');
        } catch (err) {
          console.error('Sphere creation error:', err);
          setError('Failed to create panorama');
          setIsLoading(false);
        }
      },
      undefined,
      (err) => {
        console.error('Texture load error:', err);
        setError('Failed to load image. Please try another file.');
        setIsLoading(false);
      }
    );
  }, [imageUrl]);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    setError('');

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setImageUrl(url);
  };

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  const handleReset = () => {
    if (controlsRef.current && cameraRef.current) {
      controlsRef.current.fov = 75;
      controlsRef.current.theta = 0;
      controlsRef.current.phi = Math.PI / 2;
      cameraRef.current.fov = 75;
      cameraRef.current.updateProjectionMatrix();
    }
  };

  return (
    <div className="viewer360-wrapper">
      <div className="viewer360-hud" aria-hidden={false}>
        <div className="viewer360-hud-left">
          <div className="viewer360-hud-badge" aria-hidden="true" />
          <div className="viewer360-hud-text">
            <div className="viewer360-hud-title">360° View</div>
            <div className="viewer360-hud-subtitle">Drag to explore</div>
          </div>
        </div>

        <div className="viewer360-hud-right">
          {imageUrl && (
            <div className="viewer360-hud-actions">
              <button
                className="viewer360-iconbtn"
                onClick={() => fileInputRef.current?.click()}
                title="Change image"
                type="button"
              >
                <Upload size={16} />
              </button>
              <button
                className="viewer360-iconbtn"
                onClick={handleReset}
                title="Reset view"
                type="button"
              >
                <RotateCw size={16} />
              </button>
            </div>
          )}

          <button
            className="viewer360-close"
            onClick={handleClose}
            title="Close"
            type="button"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      <div className="viewer360-container" ref={containerRef}>
        {!imageUrl && (
          <div className="viewer360-placeholder">
            <div className="placeholder-content">
              <Upload size={48} strokeWidth={1.5} />
              <h2>Load 360° Image</h2>
              <p>Upload an equirectangular panorama image to view</p>
              <button
                className="viewer360-upload-btn"
                onClick={() => fileInputRef.current?.click()}
                type="button"
              >
                Choose Image
              </button>
            </div>
          </div>
        )}

        {error && <div className="viewer360-error">{error}</div>}

        {isLoading && (
          <div className="viewer360-loading">
            <div className="loading-spinner"></div>
            <p>Loading panorama…</p>
          </div>
        )}
      </div>

      <div className="viewer360-esc-hint">
        Press <kbd>ESC</kbd> to exit
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {/* Vignette effect */}
      <div className="viewer360-vignette"></div>
    </div>
  );
}
