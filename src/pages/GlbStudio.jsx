import React, { useEffect, useRef, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { 
  Upload, Play, Pause, RotateCcw, Trash2, Box, Settings, 
  Search, Download, Eye, EyeOff, Activity, Sliders, ChevronRight
} from 'lucide-react';
import './GlbStudio.css';

export default function GlbStudio() {
  const navigate = useNavigate();
  const viewportRef = useRef(null);
  const fileInputRef = useRef(null);
  const loadIdRef = useRef(0);
  
  // Three.js instances ref
  const stateRef = useRef({
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    modelContainer: null, // Holds the loaded model (allows translation/scale/rotation edits)
    currentModel: null, // The loaded GLTF scene
    mixer: null,
    clock: new THREE.Clock(),
    skeletonHelper: null,
    boneMap: {}, // Maps name -> THREE.Bone
    boneIndicator: null, // Spherical overlay on selected bone
    animationClips: [],
    activeAction: null,
    animFrameId: null,
    resizeObserver: null,
    isPlaying: false,
    playbackSpeed: 1,
    selectedBoneName: '',
  });

  // State lists
  const [models, setModels] = useState([
    { id: 'dancing-sample', name: 'Dancing.glb (Sample)', path: '/Dancing.glb', size: '1.9 MB', isSample: true },
    { id: 'dancing-complex-sample', name: 'Dancing-complex.glb (Sample)', path: '/Dancing-complex.glb', size: '2.1 MB', isSample: true },
    { id: 'default-character', name: 'sample-model.glb (Default)', path: '/sample-model.glb', size: '2.2 MB', isSample: true },
    { id: 'robot-character', name: 'lady-x-bot.glb (Robot)', path: '/pose-temp/lady-x-bot.glb', size: '8.9 MB', isSample: true },
  ]);
  const [selectedModelId, setSelectedModelId] = useState('dancing-sample');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  
  // Animation Player state
  const [animations, setAnimations] = useState([]);
  const [activeAnimName, setActiveAnimName] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [animProgress, setAnimProgress] = useState(0); // 0 to 100

  // Model coordinates state
  const [modelPos, setModelPos] = useState({ x: 0, y: 0, z: 0 });
  const [modelRot, setModelRot] = useState({ x: 0, y: 0, z: 0 }); // In degrees
  const [modelScale, setModelScale] = useState({ x: 1, y: 1, z: 1 });

  // Skeleton / Bone state
  const [boneNames, setBoneNames] = useState([]);
  const [selectedBoneName, setSelectedBoneName] = useState('');
  const [boneSearchQuery, setBoneSearchQuery] = useState('');
  const [boneRot, setBoneRot] = useState({ x: 0, y: 0, z: 0 }); // local rotation in degrees
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [showAllBonesDropdown, setShowAllBonesDropdown] = useState(false);

  // Notifications / Toast
  const [toast, setToast] = useState({ visible: false, text: '' });
  const toastTimeoutRef = useRef(null);
  const lastProgressRef = useRef(-1);

  const showToast = (text) => {
    setToast({ visible: true, text });
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    toastTimeoutRef.current = setTimeout(() => {
      setToast({ visible: false, text: '' });
    }, 3000);
  };

  // Cleanup toasts
  useEffect(() => {
    return () => {
      if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    };
  }, []);

  // Filter bones based on search
  const filteredBones = useMemo(() => {
    if (!boneSearchQuery) return boneNames;
    return boneNames.filter(b => b.toLowerCase().includes(boneSearchQuery.toLowerCase()));
  }, [boneNames, boneSearchQuery]);

  // Init ThreeJS scene
  useEffect(() => {
    if (!viewportRef.current) return;

    const container = viewportRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f12);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.01, 100);
    camera.position.set(0, 1.2, 4.0);

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // 4. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0.8, 0);
    controls.minDistance = 0.5;
    controls.maxDistance = 15;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(3, 6, 3);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xaaccff, 0.4);
    fillLight.position.set(-3, 2, -3);
    scene.add(fillLight);

    // 6. Helpers
    const gridHelper = new THREE.GridHelper(10, 20, 0xffffff, 0x444444);
    gridHelper.material.opacity = 0.15;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    // Bone highlight indicator
    const indicatorGeo = new THREE.SphereGeometry(0.02, 16, 16);
    const indicatorMat = new THREE.MeshBasicMaterial({ 
      color: 0x0a84ff, 
      transparent: true, 
      opacity: 0.8, 
      depthTest: false 
    });
    const boneIndicator = new THREE.Mesh(indicatorGeo, indicatorMat);
    boneIndicator.renderOrder = 999;
    boneIndicator.visible = false;
    scene.add(boneIndicator);

    // Model Container
    const modelContainer = new THREE.Group();
    scene.add(modelContainer);

    stateRef.current.scene = scene;
    stateRef.current.camera = camera;
    stateRef.current.renderer = renderer;
    stateRef.current.controls = controls;
    stateRef.current.modelContainer = modelContainer;
    stateRef.current.boneIndicator = boneIndicator;

    // 7. Render Loop
    let lastTime = 0;
    const animate = (time) => {
      stateRef.current.animFrameId = requestAnimationFrame(animate);

      const delta = stateRef.current.clock.getDelta();
      
      // Update mixer using ref to avoid stale closures
      if (stateRef.current.mixer && stateRef.current.isPlaying) {
        stateRef.current.mixer.update(delta * stateRef.current.playbackSpeed);

        // Update progress scrubber
        if (stateRef.current.activeAction) {
          const action = stateRef.current.activeAction;
          const duration = action.getClip().duration;
          if (duration > 0) {
            const time = action.time;
            const progress = (time / duration) * 100;
            const roundedProgress = Math.round(progress);
            if (lastProgressRef.current !== roundedProgress) {
              lastProgressRef.current = roundedProgress;
              setAnimProgress(roundedProgress);
            }
          }
        }
      }

      // Update bone indicator position using ref to avoid stale closures
      const activeBoneName = stateRef.current.selectedBoneName;
      if (activeBoneName && stateRef.current.boneMap[activeBoneName]) {
        const bone = stateRef.current.boneMap[activeBoneName];
        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);
        boneIndicator.position.copy(worldPos);
        boneIndicator.visible = true;
      } else {
        boneIndicator.visible = false;
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 8. Handle Resizing
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);
    stateRef.current.resizeObserver = resizeObserver;

    return () => {
      cancelAnimationFrame(stateRef.current.animFrameId);
      resizeObserver.disconnect();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();

      // Reset state references to avoid memory leaks/stale additions
      if (stateRef.current) {
        stateRef.current.scene = null;
        stateRef.current.camera = null;
        stateRef.current.renderer = null;
        stateRef.current.controls = null;
        stateRef.current.modelContainer = null;
        stateRef.current.boneIndicator = null;
        stateRef.current.skeletonHelper = null;
        stateRef.current.currentModel = null;
        stateRef.current.mixer = null;
      }
    };
  }, []);

  // Update Animation and Bone state inside ref to avoid stale closures in render loop
  useEffect(() => {
    stateRef.current.isPlaying = isPlaying;
    stateRef.current.playbackSpeed = playbackSpeed;
    stateRef.current.selectedBoneName = selectedBoneName;
  }, [isPlaying, playbackSpeed, selectedBoneName]);

  // Load selected model
  useEffect(() => {
    const activeModel = models.find(m => m.id === selectedModelId);
    if (!activeModel) return;

    loadModel(activeModel);
  }, [selectedModelId]);

  const loadModel = async (modelItem) => {
    setIsLoading(true);
    setLoadingText(`Loading ${modelItem.name}…`);
    setSelectedBoneName('');
    setBoneSearchQuery('');
    setActiveAnimName('');
    setIsPlaying(false);
    stateRef.current.isPlaying = false;
    setAnimProgress(0);

    const loadId = ++loadIdRef.current;

    // Reset transform inputs
    setModelPos({ x: 0, y: 0, z: 0 });
    setModelRot({ x: 0, y: 0, z: 0 });
    setModelScale({ x: 1, y: 1, z: 1 });

    const state = stateRef.current;
    
    // Clear previous model & mixer
    if (state.modelContainer) {
      state.modelContainer.clear();
      // Reset container transforms
      state.modelContainer.position.set(0, 0, 0);
      state.modelContainer.rotation.set(0, 0, 0);
      state.modelContainer.scale.set(1, 1, 1);
    }
    if (state.skeletonHelper) {
      if (state.scene) state.scene.remove(state.skeletonHelper);
      state.skeletonHelper = null;
    }
    state.mixer = null;
    state.activeAction = null;
    state.boneMap = {};
    setBoneNames([]);
    setAnimations([]);

    try {
      const loader = new GLTFLoader();
      const gltf = await new Promise((resolve, reject) => {
        loader.load(
          modelItem.path,
          (gltf) => {
            if (loadId !== loadIdRef.current) {
              reject(new Error('Stale load'));
              return;
            }
            resolve(gltf);
          },
          undefined,
          (err) => reject(err)
        );
      });

      if (loadId !== loadIdRef.current) return;

      const modelScene = gltf.scene;
      state.currentModel = modelScene;
      state.modelContainer.add(modelScene);

      // Auto-scale and center the model
      modelScene.position.set(0, 0, 0);
      modelScene.rotation.set(0, 0, 0);
      modelScene.scale.set(1, 1, 1);
      modelScene.updateMatrixWorld(true);

      const box = new THREE.Box3().setFromObject(modelScene);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());

      // Scale model to be ~1.8 units tall
      const maxDim = Math.max(size.x, size.y, size.z, 0.0001);
      const scaleFactor = 1.8 / maxDim;
      modelScene.scale.setScalar(scaleFactor);
      modelScene.updateMatrixWorld(true);

      // Center model horizontally (X, Z) and sit on the floor (Y)
      const newBox = new THREE.Box3().setFromObject(modelScene);
      const newCenter = newBox.getCenter(new THREE.Vector3());
      modelScene.position.x -= newCenter.x;
      modelScene.position.z -= newCenter.z;
      modelScene.position.y -= newBox.min.y; // Bottom sits on Y = 0
      modelScene.updateMatrixWorld(true);

      // Adjust camera distance to fit model
      const finalBox = new THREE.Box3().setFromObject(state.modelContainer);
      const finalCenter = finalBox.getCenter(new THREE.Vector3());
      const finalSize = finalBox.getSize(new THREE.Vector3());
      
      state.controls.target.copy(finalCenter);
      state.camera.position.set(0, finalCenter.y + 0.3, finalSize.y * 1.8 + 1.2);
      state.controls.update();

      // Find bones
      const tempBoneMap = {};
      const tempBoneNames = [];
      modelScene.traverse((child) => {
        if (child.isBone) {
          tempBoneMap[child.name] = child;
          tempBoneNames.push(child.name);
        }
      });
      state.boneMap = tempBoneMap;
      setBoneNames(tempBoneNames.sort());

      // Show skeleton helper if enabled
      if (showSkeleton) {
        const helper = new THREE.SkeletonHelper(modelScene);
        helper.material.linewidth = 1;
        state.scene.add(helper);
        state.skeletonHelper = helper;
      }

      // Set up mixer & animations
      state.animationClips = gltf.animations || [];
      if (state.animationClips.length > 0) {
        state.mixer = new THREE.AnimationMixer(modelScene);
        setAnimations(state.animationClips.map(clip => clip.name || 'Unnamed Clip'));
        
        // Auto-play the first animation
        const firstClipName = state.animationClips[0].name || 'Unnamed Clip';
        playAnimationByName(firstClipName);
      }

      showToast(`Successfully loaded ${modelItem.name}`);
    } catch (err) {
      if (err.message !== 'Stale load') {
        console.error('Error loading model:', err);
        showToast('Error loading 3D model.');
      }
    } finally {
      if (loadId === loadIdRef.current) {
        setIsLoading(false);
      }
    }
  };

  const playAnimationByName = (clipName) => {
    const state = stateRef.current;
    if (!state.mixer) return;

    const clip = state.animationClips.find(c => c.name === clipName);
    if (!clip) return;

    if (state.activeAction) {
      state.activeAction.stop();
    }

    const action = state.mixer.clipAction(clip);
    action.play();
    
    state.activeAction = action;
    setActiveAnimName(clipName);
    setIsPlaying(true);
    stateRef.current.isPlaying = true;
  };

  const handlePlayPause = () => {
    if (!stateRef.current.mixer) return;
    const nextPlaying = !isPlaying;
    setIsPlaying(nextPlaying);
    stateRef.current.isPlaying = nextPlaying;
  };

  const handleTimelineScrub = (val) => {
    const action = stateRef.current.activeAction;
    if (!action) return;

    const duration = action.getClip().duration;
    const nextTime = (val / 100) * duration;
    
    action.time = nextTime;
    setAnimProgress(val);
    
    // If paused, update renderer once
    if (!isPlaying && stateRef.current.renderer) {
      // Force mixer update with 0 delta to snap animation poses
      stateRef.current.mixer.update(0);
    }
  };

  // Model transform handlers
  const handleModelPosChange = (axis, val) => {
    const numVal = parseFloat(val) || 0;
    const next = { ...modelPos, [axis]: numVal };
    setModelPos(next);
    if (stateRef.current.modelContainer) {
      stateRef.current.modelContainer.position[axis] = numVal;
    }
  };

  const handleModelRotChange = (axis, val) => {
    const numVal = parseFloat(val) || 0;
    const next = { ...modelRot, [axis]: numVal };
    setModelRot(next);
    if (stateRef.current.modelContainer) {
      // Convert degrees to radians
      stateRef.current.modelContainer.rotation[axis] = THREE.MathUtils.degToRad(numVal);
    }
  };

  const handleModelScaleChange = (axis, val) => {
    const numVal = parseFloat(val) || 0.1; // scale shouldn't be 0
    const next = { ...modelScale, [axis]: numVal };
    setModelScale(next);
    if (stateRef.current.modelContainer) {
      stateRef.current.modelContainer.scale[axis] = numVal;
    }
  };

  // Bone transform handlers
  const selectBone = (boneName) => {
    setSelectedBoneName(boneName);
    setShowAllBonesDropdown(false);
    
    const bone = stateRef.current.boneMap[boneName];
    if (bone) {
      // Read current local rotation in degrees
      setBoneRot({
        x: Math.round(THREE.MathUtils.radToDeg(bone.rotation.x)),
        y: Math.round(THREE.MathUtils.radToDeg(bone.rotation.y)),
        z: Math.round(THREE.MathUtils.radToDeg(bone.rotation.z)),
      });

      // Highlight in view
      if (stateRef.current.controls) {
        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);
        
        // Option: Smoothly center camera controls on the selected bone
        // stateRef.current.controls.target.copy(worldPos);
      }
    }
  };

  const handleBoneRotChange = (axis, val) => {
    const numVal = parseFloat(val) || 0;
    const next = { ...boneRot, [axis]: numVal };
    setBoneRot(next);

    const bone = stateRef.current.boneMap[selectedBoneName];
    if (bone) {
      // Pause animation if they are editing bones manually, so it doesn't immediately overwrite
      if (stateRef.current.isPlaying) {
        setIsPlaying(false);
        stateRef.current.isPlaying = false;
        showToast('Animation paused for manual bone editing');
      }
      bone.rotation[axis] = THREE.MathUtils.degToRad(numVal);
      bone.updateMatrixWorld(true);
    }
  };

  // Toggle Skeleton Helper
  const handleToggleSkeleton = () => {
    const nextShow = !showSkeleton;
    setShowSkeleton(nextShow);
    
    const state = stateRef.current;
    if (nextShow) {
      if (state.currentModel && !state.skeletonHelper) {
        const helper = new THREE.SkeletonHelper(state.currentModel);
        helper.material.linewidth = 1;
        state.scene.add(helper);
        state.skeletonHelper = helper;
      }
    } else {
      if (state.skeletonHelper) {
        state.scene.remove(state.skeletonHelper);
        state.skeletonHelper = null;
      }
    }
  };

  // File Upload Handlers
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.glb') && !file.name.toLowerCase().endsWith('.gltf')) {
      showToast('Please select a .glb or .gltf file.');
      return;
    }

    const name = file.name;
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const sizeStr = `${sizeMB} MB`;
    const path = URL.createObjectURL(file);

    const newModel = {
      id: `uploaded-${Date.now()}`,
      name,
      path,
      size: sizeStr,
      isSample: false
    };

    setModels(prev => [...prev, newModel]);
    setSelectedModelId(newModel.id);
    showToast(`Added custom model: ${name}`);
  };

  const handleRemoveModel = (id, e) => {
    e.stopPropagation(); // Avoid selecting the item
    const modelItem = models.find(m => m.id === id);
    if (modelItem && !modelItem.isSample) {
      URL.revokeObjectURL(modelItem.path);
    }

    const nextModels = models.filter(m => m.id !== id);
    setModels(nextModels);

    // If active model is removed, fallback to the first model
    if (selectedModelId === id && nextModels.length > 0) {
      setSelectedModelId(nextModels[0].id);
    }
  };

  // Reset all changes
  const handleResetAll = () => {
    const state = stateRef.current;
    
    // 1. Reset Model Container Transforms
    setModelPos({ x: 0, y: 0, z: 0 });
    setModelRot({ x: 0, y: 0, z: 0 });
    setModelScale({ x: 1, y: 1, z: 1 });

    if (state.modelContainer) {
      state.modelContainer.position.set(0, 0, 0);
      state.modelContainer.rotation.set(0, 0, 0);
      state.modelContainer.scale.set(1, 1, 1);
    }

    // 2. Reset Bone Rotations to neutral
    Object.values(state.boneMap).forEach(bone => {
      bone.rotation.set(0, 0, 0);
    });

    if (selectedBoneName) {
      setBoneRot({ x: 0, y: 0, z: 0 });
    }

    // 3. Force update mixer
    if (state.mixer) {
      state.mixer.stopAllAction();
      if (state.activeAction) {
        state.activeAction.play();
        setIsPlaying(true);
        stateRef.current.isPlaying = true;
      }
    }

    showToast('Reset all changes to defaults');
  };

  // Export customized GLB model
  const handleExportGLB = () => {
    const state = stateRef.current;
    if (!state.modelContainer) {
      showToast('No model loaded to export.');
      return;
    }

    showToast('Preparing model for export…');
    
    const exporter = new GLTFExporter();
    
    // Export the model container which contains our translations, scale, and posed bones
    exporter.parse(
      state.modelContainer,
      (gltfBuffer) => {
        const activeModel = models.find(m => m.id === selectedModelId);
        const baseName = activeModel ? activeModel.name.replace(/\.[^/.]+$/, '') : 'model';
        const fileName = `${baseName}_custom.glb`;

        // Create blob and download link
        const blob = new Blob([gltfBuffer], { type: 'application/octet-stream' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        link.click();
        
        // Cleanup
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        showToast(`Exported ${fileName} successfully!`);
      },
      (err) => {
        console.error('GLTF Export error:', err);
        showToast('Error occurred during GLB export.');
      },
      { binary: true } // Export as GLB
    );
  };

  return (
    <div className="glb-studio-container">
      {/* Toast Notification */}
      <div className={`toast ${toast.visible ? 'show' : ''}`}>
        <span>{toast.text}</span>
      </div>

      {/* Left Sidebar - File list & Animation Player */}
      <aside className="glb-panel">
        {/* Model uploader */}
        <div className="panel-section">
          <h4 className="panel-section-title">
            <span>Models Studio</span>
            <Box size={14} />
          </h4>
          
          <div className="glb-upload-area" onClick={() => fileInputRef.current?.click()}>
            <Upload size={24} style={{ color: 'var(--accent-color)' }} />
            <p>Drag & Drop or <strong>browse</strong> for GLB/GLTF</p>
          </div>
          <input 
            type="file" 
            ref={fileInputRef} 
            accept=".glb,.gltf" 
            style={{ display: 'none' }} 
            onChange={handleFileUpload} 
          />

          <div className="model-list">
            {models.map((model) => (
              <div 
                key={model.id}
                className={`model-list-item ${selectedModelId === model.id ? 'active' : ''}`}
                onClick={() => setSelectedModelId(model.id)}
              >
                <div className="model-item-info">
                  <span className="model-item-name">{model.name}</span>
                  <span className="model-item-size">{model.size}</span>
                </div>
                {!model.isSample && (
                  <button 
                    className="model-item-remove"
                    onClick={(e) => handleRemoveModel(model.id, e)}
                    title="Remove custom model"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Animation Player Section */}
        <div className="panel-section">
          <h4 className="panel-section-title">
            <span>Animations</span>
            <Activity size={14} />
          </h4>

          {animations.length === 0 ? (
            <p style={{ fontSize: '11px', color: 'var(--muted)', margin: 0 }}>
              No animation clips found in this GLB file.
            </p>
          ) : (
            <>
              <div className="anim-list">
                {animations.map((animName, idx) => (
                  <button 
                    key={idx}
                    className={`anim-item ${activeAnimName === animName ? 'active' : ''}`}
                    onClick={() => playAnimationByName(animName)}
                  >
                    {animName}
                  </button>
                ))}
              </div>

              <div className="playback-controls">
                <div className="playback-buttons">
                  <button 
                    className="playback-play-btn"
                    onClick={handlePlayPause}
                    title={isPlaying ? 'Pause' : 'Play'}
                  >
                    {isPlaying ? <Pause size={14} fill="#fff" /> : <Play size={14} fill="#fff" />}
                  </button>

                  <div className="playback-slider-container">
                    <input 
                      type="range"
                      className="playback-slider"
                      min="0"
                      max="100"
                      value={animProgress}
                      onChange={(e) => handleTimelineScrub(parseFloat(e.target.value))}
                    />
                    <span className="playback-frame-text">
                      {Math.round(animProgress)}%
                    </span>
                  </div>
                </div>

                <div className="speed-selector">
                  <span>Speed:</span>
                  <select 
                    className="speed-select"
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                  >
                    <option value="0.25">0.25x</option>
                    <option value="0.5">0.5x</option>
                    <option value="1">1.0x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2">2.0x</option>
                  </select>
                </div>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Middle - 3D Canvas Viewport */}
      <section className="glb-viewport-container">
        {/* Loading Spinner overlay */}
        {isLoading && (
          <div className="viewer360-loading">
            <div className="loading-spinner"></div>
            <p>{loadingText}</p>
          </div>
        )}

        {/* 3D Canvas mounting point */}
        <div ref={viewportRef} style={{ width: '100%', height: '100%' }} />

        {/* Viewport Overlay Controls */}
        <div className="viewport-hud">
          <div className="hud-group">
            <button 
              className={`hud-btn ${showSkeleton ? 'active' : ''}`}
              onClick={handleToggleSkeleton}
              title="Visualize skeleton skeleton bones"
            >
              {showSkeleton ? <Eye size={13} /> : <EyeOff size={13} />}
              Skeleton
            </button>
            <button 
              className="hud-btn"
              onClick={handleResetAll}
              title="Reset all model coordinates and skeletal poses"
            >
              <RotateCcw size={13} />
              Reset
            </button>
          </div>

          <div className="hud-group">
            <button 
              className="hud-btn primary"
              onClick={handleExportGLB}
              title="Export customized GLB character"
            >
              <Download size={13} />
              Export GLB
            </button>
          </div>
        </div>
      </section>

      {/* Right Sidebar - Transform & Rig Coordinates */}
      <aside className="glb-panel right-panel">
        {/* Global Model Coordinates */}
        <div className="panel-section">
          <h4 className="panel-section-title">
            <span>Model XYZ Coordinates</span>
            <Sliders size={14} />
          </h4>

          <div className="control-group">
            {/* Position */}
            <div className="slider-row">
              <div className="slider-header">
                <span className="slider-label">Position X <span className="slider-axis x">(Right)</span></span>
                <span className="slider-value">{modelPos.x.toFixed(2)}</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-5"
                  max="5"
                  step="0.05"
                  value={modelPos.x}
                  onChange={(e) => handleModelPosChange('x', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelPos.x}
                  onChange={(e) => handleModelPosChange('x', e.target.value)}
                />
              </div>
            </div>

            <div className="slider-row">
              <div className="slider-header">
                <span className="slider-label">Position Y <span className="slider-axis y">(Height)</span></span>
                <span className="slider-value">{modelPos.y.toFixed(2)}</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-5"
                  max="5"
                  step="0.05"
                  value={modelPos.y}
                  onChange={(e) => handleModelPosChange('y', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelPos.y}
                  onChange={(e) => handleModelPosChange('y', e.target.value)}
                />
              </div>
            </div>

            <div className="slider-row">
              <div className="slider-header">
                <span className="slider-label">Position Z <span className="slider-axis z">(Depth)</span></span>
                <span className="slider-value">{modelPos.z.toFixed(2)}</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-5"
                  max="5"
                  step="0.05"
                  value={modelPos.z}
                  onChange={(e) => handleModelPosChange('z', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelPos.z}
                  onChange={(e) => handleModelPosChange('z', e.target.value)}
                />
              </div>
            </div>

            {/* Rotation */}
            <div className="slider-row" style={{ marginTop: '6px' }}>
              <div className="slider-header">
                <span className="slider-label">Rotation X <span className="slider-axis x">(Pitch)</span></span>
                <span className="slider-value">{modelRot.x}°</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-180"
                  max="180"
                  value={modelRot.x}
                  onChange={(e) => handleModelRotChange('x', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelRot.x}
                  onChange={(e) => handleModelRotChange('x', e.target.value)}
                />
              </div>
            </div>

            <div className="slider-row">
              <div className="slider-header">
                <span className="slider-label">Rotation Y <span className="slider-axis y">(Yaw)</span></span>
                <span className="slider-value">{modelRot.y}°</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-180"
                  max="180"
                  value={modelRot.y}
                  onChange={(e) => handleModelRotChange('y', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelRot.y}
                  onChange={(e) => handleModelRotChange('y', e.target.value)}
                />
              </div>
            </div>

            <div className="slider-row">
              <div className="slider-header">
                <span className="slider-label">Rotation Z <span className="slider-axis z">(Roll)</span></span>
                <span className="slider-value">{modelRot.z}°</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="-180"
                  max="180"
                  value={modelRot.z}
                  onChange={(e) => handleModelRotChange('z', e.target.value)}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelRot.z}
                  onChange={(e) => handleModelRotChange('z', e.target.value)}
                />
              </div>
            </div>

            {/* Uniform Scale */}
            <div className="slider-row" style={{ marginTop: '6px' }}>
              <div className="slider-header">
                <span className="slider-label">Uniform Scale</span>
                <span className="slider-value">{modelScale.x.toFixed(2)}x</span>
              </div>
              <div className="slider-input-wrapper">
                <input 
                  type="range"
                  className="slider-range"
                  min="0.1"
                  max="3.0"
                  step="0.05"
                  value={modelScale.x}
                  onChange={(e) => {
                    const val = e.target.value;
                    handleModelScaleChange('x', val);
                    handleModelScaleChange('y', val);
                    handleModelScaleChange('z', val);
                  }}
                />
                <input 
                  type="text"
                  className="slider-text-input"
                  value={modelScale.x}
                  onChange={(e) => {
                    const val = e.target.value;
                    handleModelScaleChange('x', val);
                    handleModelScaleChange('y', val);
                    handleModelScaleChange('z', val);
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Rig Skeletal Bone Editor */}
        <div className="panel-section">
          <h4 className="panel-section-title">
            <span>Rig Bone posing</span>
            <Settings size={14} />
          </h4>

          {boneNames.length === 0 ? (
            <p style={{ fontSize: '11px', color: 'var(--muted)', margin: 0 }}>
              No bones detected in this 3D model. (Unrigged model)
            </p>
          ) : (
            <div className="control-group">
              {/* Bone dropdown search */}
              <div className="bone-search-container">
                <div style={{ display: 'flex', gap: 6 }}>
                  <input 
                    type="text"
                    className="bone-search-input"
                    placeholder="Search bone..."
                    value={boneSearchQuery}
                    onChange={(e) => {
                      setBoneSearchQuery(e.target.value);
                      setShowAllBonesDropdown(true);
                    }}
                    onFocus={() => setShowAllBonesDropdown(true)}
                  />
                  <button 
                    className="hud-btn"
                    style={{ padding: '8px' }}
                    onClick={() => setShowAllBonesDropdown(!showAllBonesDropdown)}
                  >
                    <ChevronRight 
                      size={14} 
                      style={{ 
                        transform: showAllBonesDropdown ? 'rotate(90deg)' : 'none', 
                        transition: 'transform 0.2s' 
                      }} 
                    />
                  </button>
                </div>

                {showAllBonesDropdown && (
                  <div className="bone-dropdown-list">
                    {filteredBones.length === 0 ? (
                      <div style={{ padding: '8px 12px', fontSize: '11px', color: 'var(--muted)' }}>
                        No bones match search
                      </div>
                    ) : (
                      filteredBones.map((boneName) => (
                        <div 
                          key={boneName}
                          className={`bone-dropdown-item ${selectedBoneName === boneName ? 'active' : ''}`}
                          onClick={() => selectBone(boneName)}
                        >
                          {boneName}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>

              {/* Selected Bone Badge & Rotational Sliders */}
              {selectedBoneName ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div className="selected-bone-badge">
                    <span className="selected-bone-name" title={selectedBoneName}>
                      {selectedBoneName}
                    </span>
                    <button className="selected-bone-clear" onClick={() => setSelectedBoneName('')}>
                      Clear Selection
                    </button>
                  </div>

                  {/* Bone Sliders */}
                  <div className="slider-row">
                    <div className="slider-header">
                      <span className="slider-label">Bone Rotate X <span className="slider-axis x">(Pitch)</span></span>
                      <span className="slider-value">{boneRot.x}°</span>
                    </div>
                    <div className="slider-input-wrapper">
                      <input 
                        type="range"
                        className="slider-range"
                        min="-180"
                        max="180"
                        value={boneRot.x}
                        onChange={(e) => handleBoneRotChange('x', e.target.value)}
                      />
                      <input 
                        type="text"
                        className="slider-text-input"
                        value={boneRot.x}
                        onChange={(e) => handleBoneRotChange('x', e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="slider-row">
                    <div className="slider-header">
                      <span className="slider-label">Bone Rotate Y <span className="slider-axis y">(Yaw)</span></span>
                      <span className="slider-value">{boneRot.y}°</span>
                    </div>
                    <div className="slider-input-wrapper">
                      <input 
                        type="range"
                        className="slider-range"
                        min="-180"
                        max="180"
                        value={boneRot.y}
                        onChange={(e) => handleBoneRotChange('y', e.target.value)}
                      />
                      <input 
                        type="text"
                        className="slider-text-input"
                        value={boneRot.y}
                        onChange={(e) => handleBoneRotChange('y', e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="slider-row">
                    <div className="slider-header">
                      <span className="slider-label">Bone Rotate Z <span className="slider-axis z">(Roll)</span></span>
                      <span className="slider-value">{boneRot.z}°</span>
                    </div>
                    <div className="slider-input-wrapper">
                      <input 
                        type="range"
                        className="slider-range"
                        min="-180"
                        max="180"
                        value={boneRot.z}
                        onChange={(e) => handleBoneRotChange('z', e.target.value)}
                      />
                      <input 
                        type="text"
                        className="slider-text-input"
                        value={boneRot.z}
                        onChange={(e) => handleBoneRotChange('z', e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <p style={{ fontSize: '11px', color: 'var(--muted)', margin: 0, textAlign: 'center', padding: '10px' }}>
                  Select a bone from the search dropdown to edit its local rotation.
                </p>
              )}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
