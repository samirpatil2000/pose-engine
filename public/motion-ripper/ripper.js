import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
        import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
        import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';
        import { GLTFExporter } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/exporters/GLTFExporter.js';
        import { FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21-rc.20250105/vision_bundle.mjs';

        const MIXAMO_BONE_MAP = {
            Hips:           'mixamorig_Hips',
            Spine:          'mixamorig_Spine1',
            Head:           'mixamorig_Head',
            Left_Upper_Arm: 'mixamorig_LeftArm',
            Left_Lower_Arm: 'mixamorig_LeftForeArm',
            Right_Upper_Arm:'mixamorig_RightArm',
            Right_Lower_Arm:'mixamorig_RightForeArm',
            Left_Upper_Leg: 'mixamorig_LeftUpLeg',
            Left_Lower_Leg: 'mixamorig_LeftLeg',
            Right_Upper_Leg:'mixamorig_RightUpLeg',
            Right_Lower_Leg:'mixamorig_RightLeg',
        };

        const ASSET_FORMAT = 'fast-poser-asset';
        const ASSET_VERSION = 1;
        const MAIN_ANIMATION_STORAGE_KEY = 'fast-poser:animation-library';
        const MEDIAPIPE_WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21-rc.20250105/wasm';
        const MEDIAPIPE_MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
        const ROOT_BASE_POSITION = new THREE.Vector3(0, 2.6, 0);
        const DOWN_AXIS = new THREE.Vector3(0, -1, 0);
        const DEFAULT_COLORS = ['#5eead4', '#f472b6'];
        const DEFAULT_COLOR = DEFAULT_COLORS[0];
        const MAX_CAPTURE_CHARACTERS = 2;
        const MULTI_CHARACTER_SPACING = 3;
        const REGION_SCAN_MAX_WIDTH = 720;
        const REGION_SCAN_MIN_WIDTH = 320;
        const DUPLICATE_POSE_X_THRESHOLD = 0.12;
        const MIN_POSE_DETECTION_CONFIDENCE = 0.4;
        const MIN_POSE_PRESENCE_CONFIDENCE = 0.4;
        const MIN_POSE_TRACKING_CONFIDENCE = 0.4;
        const DEFAULT_LANDMARK_VISIBILITY = 0.42;
        const UPPER_BODY_LANDMARK_VISIBILITY = 0.25;
        const MOVENET_KEYPOINT_SCORE = 0.18;
        const MIN_CROP_SIZE = 0.08;
        const KEYFRAME_TIME_STEP = 0.1;
        const MANUAL_TRACK_MAX_WIDTH = 560;
        const MANUAL_TRACK_PATCH_RADIUS = 6;
        const MANUAL_TRACK_SEARCH_RADIUS = 30;
        const MANUAL_TRACK_SEARCH_STEP = 3;
        const MANUAL_TRACK_MAX_SCORE = 54;
        const MANUAL_TRACK_TEMPLATE_BLEND = 0.08;
        const POSE_BACKENDS = {
            MEDIAPIPE_VIDEO: 'mediapipe-video',
            MEDIAPIPE_IMAGE: 'mediapipe-image',
            MOVENET: 'movenet',
            MANUAL: 'manual'
        };
        const BASE_JOINT_ORDER = [
            'Hips',
            'Spine',
            'Head',
            'Left_Upper_Arm',
            'Left_Lower_Arm',
            'Right_Upper_Arm',
            'Right_Lower_Arm',
            'Left_Upper_Leg',
            'Left_Lower_Leg',
            'Right_Upper_Leg',
            'Right_Lower_Leg'
        ];
        const BASE_JOINT_PARENTS = {
            Hips: null,
            Spine: 'Hips',
            Head: 'Spine',
            Left_Upper_Arm: 'Spine',
            Left_Lower_Arm: 'Left_Upper_Arm',
            Right_Upper_Arm: 'Spine',
            Right_Lower_Arm: 'Right_Upper_Arm',
            Left_Upper_Leg: 'Hips',
            Left_Lower_Leg: 'Left_Upper_Leg',
            Right_Upper_Leg: 'Hips',
            Right_Lower_Leg: 'Right_Upper_Leg'
        };
        const BASE_JOINT_SET = new Set(BASE_JOINT_ORDER);
        const STANDING_LOWER_BODY_JOINTS = [
            'Hips',
            'Left_Upper_Leg',
            'Left_Lower_Leg',
            'Right_Upper_Leg',
            'Right_Lower_Leg'
        ];
        const CONNECTIONS = [
            [11, 12],
            [11, 13],
            [13, 15],
            [12, 14],
            [14, 16],
            [11, 23],
            [12, 24],
            [23, 24],
            [23, 25],
            [25, 27],
            [24, 26],
            [26, 28],
            [0, 11],
            [0, 12]
        ];
        const LM = {
            NOSE: 0,
            LEFT_EAR: 7,
            RIGHT_EAR: 8,
            LEFT_SHOULDER: 11,
            RIGHT_SHOULDER: 12,
            LEFT_ELBOW: 13,
            RIGHT_ELBOW: 14,
            LEFT_WRIST: 15,
            RIGHT_WRIST: 16,
            LEFT_HIP: 23,
            RIGHT_HIP: 24,
            LEFT_KNEE: 25,
            RIGHT_KNEE: 26,
            LEFT_ANKLE: 27,
            RIGHT_ANKLE: 28
        };
        const MOVENET_TO_MEDIAPIPE_LANDMARKS = {
            nose: LM.NOSE,
            left_ear: LM.LEFT_EAR,
            right_ear: LM.RIGHT_EAR,
            left_shoulder: LM.LEFT_SHOULDER,
            right_shoulder: LM.RIGHT_SHOULDER,
            left_elbow: LM.LEFT_ELBOW,
            right_elbow: LM.RIGHT_ELBOW,
            left_wrist: LM.LEFT_WRIST,
            right_wrist: LM.RIGHT_WRIST,
            left_hip: LM.LEFT_HIP,
            right_hip: LM.RIGHT_HIP,
            left_knee: LM.LEFT_KNEE,
            right_knee: LM.RIGHT_KNEE,
            left_ankle: LM.LEFT_ANKLE,
            right_ankle: LM.RIGHT_ANKLE
        };
        const MANUAL_HANDLE_DEFS = [
            { id: 'head', label: 'Head', landmark: LM.NOSE },
            { id: 'leftShoulder', label: 'L Shoulder', landmark: LM.LEFT_SHOULDER },
            { id: 'rightShoulder', label: 'R Shoulder', landmark: LM.RIGHT_SHOULDER },
            { id: 'leftElbow', label: 'L Elbow', landmark: LM.LEFT_ELBOW },
            { id: 'rightElbow', label: 'R Elbow', landmark: LM.RIGHT_ELBOW },
            { id: 'leftWrist', label: 'L Wrist', landmark: LM.LEFT_WRIST },
            { id: 'rightWrist', label: 'R Wrist', landmark: LM.RIGHT_WRIST }
        ];

        const ui = {};
        let scene;
        let camera;
        let renderer;
        let orbitControls;
        let previewRoot;
        let previewRoots = [];
        let resizeObserver = null;
        let regionScanCanvas = null;
        let regionScanContext = null;
        let sourceCropRect = null;
        let cropDragState = null;
        let manualHandles = [];
        let manualHandleKeyframes = [];
        let manualTrackTemplates = [];
        let manualTrackCanvas = null;
        let manualTrackContext = null;
        let manualDragState = null;
        let defaultPoseState;
        let currentPoseState;
        let poseLandmarker = null;
        let poseLandmarkerPoseCount = 0;
        let poseLandmarkerMode = '';
        let moveNetDetector = null;
        let moveNetModelType = '';
        let mediaStream = null;
        let animationFrameId = 0;
        let lastProcessedVideoTime = -1;
        let isProcessingFrame = false;
        let isRecording = false;
        let recordingStartedAt = 0;
        let lastSampledAt = -Infinity;
        let recordedFrames = [];
        let latestPosePackets = [];
        let rootBaselines = [];
        let isPlayingUploadedVideo = false;

        let glbModels = [];
        let glbBonesMaps = [];
        let glbRestQuaternions = [];

        window.__motionRipperCleanup?.();
        init();

        function init() {
            cacheUi();
            bindUi();
            initScene();
            createPreviewCharacters();
            defaultPoseState = captureCharacterPose();
            currentPoseState = clonePoseState(defaultPoseState);
            applyPoseState(currentPoseState);
            setGeneratedAnimationName();
            updateSmoothingLabel();
            updateCaptureModeUi();
            updateRecordingUi();
            updateStats();
            updateManualAssistKeyUi();
            handleResize();
            warmupMediaPipe();
            window.addEventListener('resize', handleResize);
        }

        function cacheUi() {
            ui.shareScreenBtn = document.getElementById('share-screen-btn');
            ui.uploadVideoBtn = document.getElementById('upload-video-btn');
            ui.videoFileInput = document.getElementById('video-file-input');
            ui.stopShareBtn = document.getElementById('stop-share-btn');
            ui.neutralBtn = document.getElementById('neutral-btn');
            ui.recordBtn = document.getElementById('record-btn');
            ui.clearBtn = document.getElementById('clear-btn');
            ui.saveLibraryBtn = document.getElementById('save-library-btn');
            ui.exportHtmlBtn = document.getElementById('export-html-btn');
            ui.exportGlbBtn = document.getElementById('export-glb-btn');
            ui.animationName = document.getElementById('animation-name');
            ui.characterColor = document.getElementById('character-color');
            ui.secondCharacterColor = document.getElementById('second-character-color');
            ui.secondCharacterColorField = document.getElementById('second-character-color-field');
            ui.sampleRate = document.getElementById('sample-rate');
            ui.smoothing = document.getElementById('smoothing');
            ui.smoothingValue = document.getElementById('smoothing-value');
            ui.poseBackend = document.getElementById('pose-backend');
            ui.manualAssistPanel = document.getElementById('manual-assist-panel');
            ui.assistCharacter = document.getElementById('assist-character');
            ui.resetAssistBtn = document.getElementById('reset-assist-btn');
            ui.addAssistKeyBtn = document.getElementById('add-assist-key-btn');
            ui.clearAssistKeysBtn = document.getElementById('clear-assist-keys-btn');
            ui.assistKeyCount = document.getElementById('assist-key-count');
            ui.rootMotion = document.getElementById('root-motion');
            ui.rootMotionField = document.getElementById('root-motion-field');
            ui.upperBodyOnly = document.getElementById('upper-body-only');
            ui.focusCrop = document.getElementById('focus-crop');
            ui.resetCropBtn = document.getElementById('reset-crop-btn');
            ui.multiCharacter = document.getElementById('multi-character');
            ui.recordingBadge = document.getElementById('recording-badge');
            ui.trackedState = document.getElementById('tracked-state');
            ui.confidenceValue = document.getElementById('confidence-value');
            ui.frameCount = document.getElementById('frame-count');
            ui.durationValue = document.getElementById('duration-value');
            ui.statusText = document.getElementById('status-text');
            ui.videoMeta = document.getElementById('video-meta');
            ui.sourceVideo = document.getElementById('source-video');
            ui.poseOverlay = document.getElementById('pose-overlay');
            ui.previewCanvas = document.getElementById('preview-canvas');
            ui.characterModel = document.getElementById('character-model');
        }

        function bindUi() {
            ui.shareScreenBtn?.addEventListener('click', startScreenShare);
            ui.uploadVideoBtn?.addEventListener('click', () => ui.videoFileInput.click());
            ui.videoFileInput?.addEventListener('change', handleVideoFileSelected);
            ui.stopShareBtn?.addEventListener('click', stopScreenShare);
            ui.neutralBtn?.addEventListener('click', captureNeutralPose);
            ui.recordBtn?.addEventListener('click', toggleRecording);
            ui.clearBtn?.addEventListener('click', clearRecording);
            ui.saveLibraryBtn?.addEventListener('click', saveAnimationToLibrary);
            ui.exportHtmlBtn?.addEventListener('click', exportAnimationAsHtml);
            ui.exportGlbBtn?.addEventListener('click', exportAnimationAsGlb);
            ui.characterColor.addEventListener('input', () => setCharacterColorByIndex(0, ui.characterColor.value));
            ui.secondCharacterColor?.addEventListener('input', () => setCharacterColorByIndex(1, ui.secondCharacterColor.value));
            ui.multiCharacter?.addEventListener('change', handleMultiCharacterChanged);
            ui.upperBodyOnly.addEventListener('change', () => handleCaptureModeChanged());
            ui.poseBackend.addEventListener('change', handlePoseBackendChanged);
            ui.assistCharacter?.addEventListener('change', () => drawOverlay(latestPosePackets));
            ui.resetAssistBtn?.addEventListener('click', resetManualHandles);
            ui.addAssistKeyBtn?.addEventListener('click', () => addManualHandleKeyframe());
            ui.clearAssistKeysBtn?.addEventListener('click', clearManualHandleKeyframes);
            ui.focusCrop.addEventListener('change', handleFocusCropChanged);
            ui.resetCropBtn.addEventListener('click', resetSourceCrop);
            ui.poseOverlay.addEventListener('pointerdown', handleCropPointerDown);
            ui.poseOverlay.addEventListener('pointermove', handleCropPointerMove);
            ui.poseOverlay.addEventListener('pointerup', handleCropPointerUp);
            ui.poseOverlay.addEventListener('pointercancel', cancelCropDrag);
            ui.smoothing.addEventListener('input', updateSmoothingLabel);
            ui.characterModel.addEventListener('change', handleCharacterModelChanged);
            ui.sourceVideo.addEventListener('loadedmetadata', () => {
                ui.videoMeta.textContent = `${ui.sourceVideo.videoWidth}x${ui.sourceVideo.videoHeight}`;
                sourceCropRect = null;
                manualHandleKeyframes = [];
                clearManualTrackTemplates();
                updateManualAssistKeyUi();
                resizeOverlayCanvas();
            });
            ui.sourceVideo.addEventListener('emptied', () => {
                ui.videoMeta.textContent = 'No stream';
                clearManualTrackTemplates();
                clearOverlay();
            });
        }

        function initScene() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050816);
            scene.fog = new THREE.Fog(0x050816, 10, 26);

            camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
            camera.position.set(0, 4.8, 10.5);

            renderer = new THREE.WebGLRenderer({
                canvas: ui.previewCanvas,
                antialias: true,
                alpha: true
            });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit to 2 for performance
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            // Use ResizeObserver for crisp resolution
            resizeObserver = new ResizeObserver(() => handleResize());
            resizeObserver.observe(ui.previewCanvas);

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
            scene.add(ambientLight);

            const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
            keyLight.position.set(6, 10, 5);
            keyLight.castShadow = true;
            keyLight.shadow.mapSize.width = 2048;
            keyLight.shadow.mapSize.height = 2048;
            scene.add(keyLight);

            const rimLight = new THREE.DirectionalLight(0x67e8f9, 0.7);
            rimLight.position.set(-6, 6, -4);
            scene.add(rimLight);

            const fillLight = new THREE.PointLight(0x34d399, 0.8, 20);
            fillLight.position.set(0, 5, 4);
            scene.add(fillLight);

            const grid = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
            scene.add(grid);

            const floor = new THREE.Mesh(
                new THREE.PlaneGeometry(50, 50),
                new THREE.MeshStandardMaterial({ color: 0x07111f, roughness: 0.9, metalness: 0.05 })
            );
            floor.rotation.x = -Math.PI / 2;
            floor.receiveShadow = true;
            scene.add(floor);

            orbitControls = new OrbitControls(camera, renderer.domElement);
            orbitControls.enableDamping = true;
            orbitControls.dampingFactor = 0.08;
            orbitControls.target.set(0, 2.6, 0);
            orbitControls.maxPolarAngle = Math.PI / 2 - 0.08;

            renderer.setAnimationLoop(renderScene);
        }

        function getCaptureCharacterCount() {
            return ui.multiCharacter?.checked ? MAX_CAPTURE_CHARACTERS : 1;
        }

        function isUpperBodyOnly() {
            return !!ui.upperBodyOnly?.checked;
        }

        function isFocusCropEnabled() {
            return !!ui.focusCrop?.checked;
        }

        function getPoseBackend() {
            return ui.poseBackend?.value || POSE_BACKENDS.MEDIAPIPE_VIDEO;
        }

        function getPoseBackendLabel() {
            const labels = {
                [POSE_BACKENDS.MEDIAPIPE_VIDEO]: 'MediaPipe Video',
                [POSE_BACKENDS.MEDIAPIPE_IMAGE]: 'MediaPipe Image Search',
                [POSE_BACKENDS.MOVENET]: 'MoveNet Search',
                [POSE_BACKENDS.MANUAL]: 'Manual Assist'
            };
            return labels[getPoseBackend()] || 'Pose Backend';
        }

        function usesMediaPipeBackend() {
            return getPoseBackend() !== POSE_BACKENDS.MOVENET && getPoseBackend() !== POSE_BACKENDS.MANUAL;
        }

        function isManualAssistBackend() {
            return getPoseBackend() === POSE_BACKENDS.MANUAL;
        }

        function getMediaPipeRunningMode() {
            return getPoseBackend() === POSE_BACKENDS.MEDIAPIPE_IMAGE ? 'IMAGE' : 'VIDEO';
        }

        function getJointName(baseName, characterIndex) {
            return `${baseName}_${characterIndex}`;
        }

        function parseJointName(jointName) {
            const match = String(jointName || '').match(/^(.+)_(\d+)$/);
            if (!match || !BASE_JOINT_SET.has(match[1])) return null;
            return {
                baseName: match[1],
                characterIndex: Number.parseInt(match[2], 10)
            };
        }

        function isKnownJointName(jointName) {
            return !!parseJointName(jointName);
        }

        function getJointOrder(characterCount = getCaptureCharacterCount()) {
            const order = [];
            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                BASE_JOINT_ORDER.forEach(baseName => {
                    order.push(getJointName(baseName, characterIndex));
                });
            }
            return order;
        }

        function getCharacterColorValue(characterIndex) {
            const colorInput = characterIndex === 1 ? ui.secondCharacterColor : ui.characterColor;
            return colorInput?.value || DEFAULT_COLORS[characterIndex] || DEFAULT_COLOR;
        }

        function getCharacterColors() {
            return Array.from({ length: getCaptureCharacterCount() }, (_, characterIndex) => {
                return previewRoots[characterIndex]?.userData.characterColor || getCharacterColorValue(characterIndex);
            });
        }

        function getCharacterBasePosition(characterIndex, characterCount = getCaptureCharacterCount()) {
            const xOffset = characterCount > 1
                ? (characterIndex - (characterCount - 1) / 2) * MULTI_CHARACTER_SPACING
                : 0;
            return new THREE.Vector3(
                ROOT_BASE_POSITION.x + xOffset,
                ROOT_BASE_POSITION.y,
                ROOT_BASE_POSITION.z
            );
        }

        function isGlbModelMode() {
            return ui.characterModel?.value === 'lady-x-bot';
        }

        function createPreviewCharacters() {
            previewRoots.forEach(root => scene.remove(root));
            previewRoots = [];

            const characterCount = getCaptureCharacterCount();
            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                previewRoots.push(createPreviewCharacter(characterIndex, characterCount));
            }

            previewRoot = previewRoots[0] || null;
            syncPreviewVisibility();
        }

        function syncPreviewVisibility() {
            const glbMode = isGlbModelMode();
            previewRoots.forEach(root => { root.visible = !glbMode; });
            glbModels.forEach(model => { if (model) model.visible = glbMode; });
        }

        async function handleCharacterModelChanged() {
            syncPreviewVisibility();
            if (isGlbModelMode()) {
                await ensureGlbCharacters();
                if (currentPoseState) applyPoseState(currentPoseState);
            }
        }

        async function ensureGlbCharacters() {
            const characterCount = getCaptureCharacterCount();
            const loader = new GLTFLoader();

            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                if (glbModels[characterIndex]) continue;
                try {
                    const gltf = await loader.loadAsync('/models/lady-x-bot.glb');
                    const model = gltf.scene;

                    const box = new THREE.Box3().setFromObject(model);
                    const modelHeight = box.max.y - box.min.y;
                    const targetHeight = 5.2;
                    if (modelHeight > 0) model.scale.setScalar(targetHeight / modelHeight);

                    const scaledBox = new THREE.Box3().setFromObject(model);
                    const basePos = getCharacterBasePosition(characterIndex, characterCount);
                    model.position.set(basePos.x, -scaledBox.min.y, 0);

                    const rawBones = {};
                    model.traverse(obj => {
                        if (obj.isBone || obj.type === 'Bone') rawBones[obj.name] = obj;
                    });
                    const boneMap = {};
                    const restQuat = {};
                    Object.entries(MIXAMO_BONE_MAP).forEach(([jointName, mixamoName]) => {
                        const bone = rawBones[mixamoName];
                        if (!bone) return;
                        boneMap[jointName] = bone;
                        restQuat[jointName] = bone.quaternion.clone();
                    });

                    glbModels[characterIndex] = model;
                    glbBonesMaps[characterIndex] = boneMap;
                    glbRestQuaternions[characterIndex] = restQuat;
                    scene.add(model);
                    model.visible = isGlbModelMode();

                    setCharacterColorOnGlb(model, getCharacterColorValue(characterIndex));
                } catch (err) {
                    console.error('Failed to load lady-x-bot.glb', err);
                    setStatus('Could not load Lady-X Bot model.', 'error');
                }
            }
        }

        function setCharacterColorOnGlb(model, colorValue) {
            if (!model) return;
            const color = new THREE.Color(colorValue || DEFAULT_COLOR);
            model.traverse(obj => {
                if (obj.isMesh && obj.material?.color) {
                    obj.material = obj.material.clone();
                    obj.material.color.copy(color);
                }
            });
        }

        function createPreviewCharacter(characterIndex, characterCount) {
            const color = new THREE.Color(getCharacterColorValue(characterIndex));
            const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.08 });

            // pivot at origin; geometry offset by pivotY (negative = pivot at top, positive = pivot at bottom)
            function caps(r, len, pivotY, name) {
                const group = new THREE.Group();
                group.name = `${name}_${characterIndex}`;
                const geo = new THREE.CapsuleGeometry(r, len, 4, 10);
                geo.translate(0, pivotY, 0);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }

            function cyl(rTop, rBot, h, pivotY, name) {
                const group = new THREE.Group();
                group.name = `${name}_${characterIndex}`;
                const geo = new THREE.CylinderGeometry(rTop, rBot, h, 10);
                geo.translate(0, pivotY, 0);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }

            // Dimensions: r=radius, l=cylinder length, H=total height (l + 2r)
            // Proportions matched to reference: thick thighs, wide chest, defined hips
            const UARM_R = 0.13,  UARM_L = 0.52, UARM_H = UARM_L + 2 * UARM_R;
            const LARM_R = 0.105, LARM_L = 0.48, LARM_H = LARM_L + 2 * LARM_R;
            const HAND_R = 0.085, HAND_L = 0.06, HAND_H = HAND_L + 2 * HAND_R;
            const ULEG_R = 0.175, ULEG_L = 0.72, ULEG_H = ULEG_L + 2 * ULEG_R;
            const LLEG_R = 0.125, LLEG_L = 0.68, LLEG_H = LLEG_L + 2 * LLEG_R;
            const CHEST_H = 0.85;
            const NECK_R = 0.10,  NECK_L = 0.13, NECK_H = NECK_L + 2 * NECK_R;
            const HEAD_R = 0.25,  HEAD_L = 0.18, HEAD_H = HEAD_L + 2 * HEAD_R;
            const HIP_R  = 0.195, HIP_L  = 0.10, HIP_H  = HIP_L  + 2 * HIP_R;

            // Hips — wide oval, pivot at center
            const root = caps(HIP_R, HIP_L, 0, 'Hips');
            root.position.copy(getCharacterBasePosition(characterIndex, characterCount));

            // Chest — strongly tapered: broad shoulders, narrow waist, pivot at bottom
            const spine = cyl(0.32, 0.20, CHEST_H, CHEST_H / 2, 'Spine');
            spine.position.set(0, HIP_H * 0.35, 0);
            root.add(spine);

            // Neck — short and thick, pivot at bottom
            const neck = caps(NECK_R, NECK_L, NECK_H / 2, 'Neck');
            neck.position.set(0, CHEST_H, 0);
            spine.add(neck);

            // Head — oval capsule, pivot at bottom (chin)
            const head = caps(HEAD_R, HEAD_L, HEAD_H / 2, 'Head');
            head.position.set(0, NECK_H, 0);
            neck.add(head);

            // Arms — pivot at shoulder, full T-pose (rotation.z ±π/2)
            const leftUpperArm = caps(UARM_R, UARM_L, -UARM_H / 2, 'Left_Upper_Arm');
            leftUpperArm.position.set(0.33, CHEST_H - 0.08, 0);
            leftUpperArm.rotation.z = Math.PI / 2;
            spine.add(leftUpperArm);

            const leftLowerArm = caps(LARM_R, LARM_L, -LARM_H / 2, 'Left_Lower_Arm');
            leftLowerArm.position.set(0, -UARM_H, 0);
            leftUpperArm.add(leftLowerArm);

            const leftHand = caps(HAND_R, HAND_L, -HAND_H / 2, 'Left_Hand');
            leftHand.position.set(0, -LARM_H, 0);
            leftLowerArm.add(leftHand);

            const rightUpperArm = caps(UARM_R, UARM_L, -UARM_H / 2, 'Right_Upper_Arm');
            rightUpperArm.position.set(-0.33, CHEST_H - 0.08, 0);
            rightUpperArm.rotation.z = -Math.PI / 2;
            spine.add(rightUpperArm);

            const rightLowerArm = caps(LARM_R, LARM_L, -LARM_H / 2, 'Right_Lower_Arm');
            rightLowerArm.position.set(0, -UARM_H, 0);
            rightUpperArm.add(rightLowerArm);

            const rightHand = caps(HAND_R, HAND_L, -HAND_H / 2, 'Right_Hand');
            rightHand.position.set(0, -LARM_H, 0);
            rightLowerArm.add(rightHand);

            // Legs — thick thighs, pivot at hip joint
            const leftUpperLeg = caps(ULEG_R, ULEG_L, -ULEG_H / 2, 'Left_Upper_Leg');
            leftUpperLeg.position.set(0.185, -HIP_H * 0.32, 0);
            root.add(leftUpperLeg);

            const leftLowerLeg = caps(LLEG_R, LLEG_L, -LLEG_H / 2, 'Left_Lower_Leg');
            leftLowerLeg.position.set(0, -ULEG_H, 0);
            leftUpperLeg.add(leftLowerLeg);

            // Foot — horizontal capsule pointing forward (+Z)
            const leftFoot = caps(0.09, 0.28, -0.09, 'Left_Foot');
            leftFoot.position.set(0.02, -LLEG_H, 0.05);
            leftFoot.rotation.x = -Math.PI / 2;
            leftLowerLeg.add(leftFoot);

            const rightUpperLeg = caps(ULEG_R, ULEG_L, -ULEG_H / 2, 'Right_Upper_Leg');
            rightUpperLeg.position.set(-0.185, -HIP_H * 0.32, 0);
            root.add(rightUpperLeg);

            const rightLowerLeg = caps(LLEG_R, LLEG_L, -LLEG_H / 2, 'Right_Lower_Leg');
            rightLowerLeg.position.set(0, -ULEG_H, 0);
            rightUpperLeg.add(rightLowerLeg);

            const rightFoot = caps(0.09, 0.28, -0.09, 'Right_Foot');
            rightFoot.position.set(-0.02, -LLEG_H, 0.05);
            rightFoot.rotation.x = -Math.PI / 2;
            rightLowerLeg.add(rightFoot);

            root.userData.characterColor = `#${color.getHexString()}`;
            scene.add(root);
            return root;
        }

        function createLimb(width, height, depth, pivotYOffset, material, name, charId) {
            const group = new THREE.Group();
            group.name = `${name}_${charId}`;

            const geometry = new THREE.BoxGeometry(width, height, depth);
            geometry.translate(0, pivotYOffset, 0);

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            group.add(mesh);

            return group;
        }

        function setCharacterColor(root, value) {
            if (!root) return;

            const color = new THREE.Color(value || DEFAULT_COLOR);
            root.traverse(object => {
                if (!object.isMesh || !object.material?.color) return;
                object.material.color.copy(color);
            });
            root.userData.characterColor = `#${color.getHexString()}`;
        }

        function setCharacterColorByIndex(characterIndex, value) {
            setCharacterColor(previewRoots[characterIndex], value);
            setCharacterColorOnGlb(glbModels[characterIndex], value);
        }

        function captureCharacterPose() {
            const pose = {};
            previewRoots.forEach(root => {
                root.traverse(object => {
                    if (!object.isGroup || !object.name || !isKnownJointName(object.name)) return;
                    pose[object.name] = {
                        position: object.position.clone(),
                        quaternion: object.quaternion.clone()
                    };
                });
            });
            return pose;
        }

        function clonePoseState(source) {
            const pose = {};
            Object.entries(source || {}).forEach(([name, transform]) => {
                if (!transform?.position || !transform?.quaternion) return;
                pose[name] = {
                    position: transform.position.clone(),
                    quaternion: transform.quaternion.clone()
                };
            });
            return pose;
        }

        function applyPoseState(pose) {
            if (!pose) return;

            previewRoots.forEach(root => {
                root.traverse(object => {
                    if (!object.isGroup || !object.name || !pose[object.name]) return;
                    object.position.copy(pose[object.name].position);
                    object.quaternion.copy(pose[object.name].quaternion);
                });
            });

            if (isGlbModelMode()) {
                applyPoseStateToGlb(pose);
            }
        }

        function applyPoseStateToGlb(pose) {
            const characterCount = getCaptureCharacterCount();
            glbModels.forEach((model, characterIndex) => {
                if (!model) return;
                const boneMap = glbBonesMaps[characterIndex] || {};
                const restQuats = glbRestQuaternions[characterIndex] || {};

                Object.entries(pose).forEach(([jointName, transform]) => {
                    const parsed = parseJointName(jointName);
                    if (!parsed || parsed.characterIndex !== characterIndex) return;

                    if (parsed.baseName === 'Hips') {
                        const baseX = getCharacterBasePosition(characterIndex, characterCount).x;
                        model.position.x = baseX + (transform.position.x - getCharacterBasePosition(0, characterCount).x);
                        model.position.z = transform.position.z;
                    }

                    const bone = boneMap[parsed.baseName];
                    if (!bone) return;

                    const restQuat = restQuats[parsed.baseName] || new THREE.Quaternion();
                    const defaultJoint = defaultPoseState?.[jointName];
                    const defaultQuat = defaultJoint?.quaternion ?? new THREE.Quaternion();

                    // delta = how much box bone moved from its T-pose
                    const delta = transform.quaternion.clone().multiply(defaultQuat.clone().invert());
                    bone.quaternion.copy(restQuat.clone().multiply(delta));
                });
            });
        }

        async function warmupMediaPipe() {
            try {
                await ensurePoseBackend();
                setStatus('Pose backend is ready. Share your screen or upload a video file.', 'success');
            } catch (error) {
                console.error(error);
                setStatus('Pose backend could not load. Check your internet connection and reload this page.', 'error');
            }
        }

        async function ensurePoseBackend() {
            if (isManualAssistBackend()) {
                ensureManualHandles();
                return null;
            }

            if (usesMediaPipeBackend()) {
                return ensurePoseLandmarker(getMediaPipeRunningMode());
            }

            return ensureMoveNetDetector();
        }

        async function ensurePoseLandmarker(runningMode = 'VIDEO') {
            const requestedPoseCount = getCaptureCharacterCount();

            if (poseLandmarker && poseLandmarkerPoseCount === requestedPoseCount && poseLandmarkerMode === runningMode) {
                return poseLandmarker;
            }

            if (poseLandmarker) {
                poseLandmarker.close?.();
                poseLandmarker = null;
                poseLandmarkerPoseCount = 0;
                poseLandmarkerMode = '';
            }

            setStatus(`Loading MediaPipe ${runningMode === 'IMAGE' ? 'image search' : 'video tracker'}...`, 'info');
            const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);

            try {
                poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: MEDIAPIPE_MODEL_PATH,
                        delegate: 'GPU'
                    },
                    runningMode,
                    numPoses: requestedPoseCount,
                    minPoseDetectionConfidence: MIN_POSE_DETECTION_CONFIDENCE,
                    minPosePresenceConfidence: MIN_POSE_PRESENCE_CONFIDENCE,
                    minTrackingConfidence: MIN_POSE_TRACKING_CONFIDENCE
                });
            } catch (error) {
                poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: MEDIAPIPE_MODEL_PATH,
                        delegate: 'CPU'
                    },
                    runningMode,
                    numPoses: requestedPoseCount,
                    minPoseDetectionConfidence: MIN_POSE_DETECTION_CONFIDENCE,
                    minPosePresenceConfidence: MIN_POSE_PRESENCE_CONFIDENCE,
                    minTrackingConfidence: MIN_POSE_TRACKING_CONFIDENCE
                });
            }

            poseLandmarkerPoseCount = requestedPoseCount;
            poseLandmarkerMode = runningMode;
            return poseLandmarker;
        }

        async function ensureMoveNetDetector() {
            if (moveNetDetector) {
                return moveNetDetector;
            }

            const tf = window.tf;
            const poseDetection = window.poseDetection;
            if (!tf || !poseDetection) {
                throw new Error('TensorFlow.js MoveNet scripts are not available.');
            }

            setStatus('Loading MoveNet pose detector...', 'info');
            try {
                await tf.setBackend('webgl');
            } catch (error) {
                console.warn('Could not switch TensorFlow.js to WebGL.', error);
            }
            await tf.ready();

            const modelTypes = poseDetection.movenet?.modelType || {};
            moveNetModelType = modelTypes.MULTIPOSE_LIGHTNING
                || modelTypes.SINGLEPOSE_THUNDER
                || modelTypes.SINGLEPOSE_LIGHTNING
                || '';

            const detectorConfig = {
                scoreThreshold: MOVENET_KEYPOINT_SCORE
            };
            if (moveNetModelType) {
                detectorConfig.modelType = moveNetModelType;
            }
            if (moveNetModelType === modelTypes.MULTIPOSE_LIGHTNING) {
                detectorConfig.maxPoses = MAX_CAPTURE_CHARACTERS;
                detectorConfig.enableTracking = true;
            }

            moveNetDetector = await poseDetection.createDetector(
                poseDetection.SupportedModels.MoveNet,
                detectorConfig
            );
            return moveNetDetector;
        }

        async function startScreenShare() {
            try {
                await ensurePoseBackend();

                if (mediaStream) {
                    stopScreenShare({ preservePose: true, skipStatus: true });
                }

                mediaStream = await navigator.mediaDevices.getDisplayMedia({
                    video: {
                        frameRate: { ideal: 30, max: 30 }
                    },
                    audio: false
                });

                const [videoTrack] = mediaStream.getVideoTracks();
                if (videoTrack) {
                    videoTrack.addEventListener('ended', () => {
                        const activeTrack = mediaStream?.getVideoTracks?.()[0];
                        if (activeTrack !== videoTrack) return;
                        stopScreenShare({ preservePose: true });
                    });
                    ui.videoMeta.textContent = videoTrack.label || 'Shared stream';
                }

                ui.sourceVideo.srcObject = mediaStream;
                await ui.sourceVideo.play();
                rootBaselines = [];
                resizeOverlayCanvas();

                lastProcessedVideoTime = -1;
                processFrame();
                setStatus('Screen share connected. Play the reference video, wait for the skeleton, then record.', 'success');
            } catch (error) {
                console.error(error);
                const message = error?.name === 'NotAllowedError'
                    ? 'Screen sharing was canceled or blocked.'
                    : 'Could not start screen sharing.';
                setStatus(message, 'error');
            }
        }

        function stopScreenShare(options = {}) {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = 0;
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }

            ui.sourceVideo.pause();
            ui.sourceVideo.srcObject = null;
            isPlayingUploadedVideo = false;
            lastProcessedVideoTime = -1;
            latestPosePackets = [];
            if (ui.trackedState) ui.trackedState.textContent = 'Waiting';
            if (ui.confidenceValue) ui.confidenceValue.textContent = '0%';
            clearOverlay();

            if (!options.preservePose) {
                currentPoseState = clonePoseState(defaultPoseState);
                applyPoseState(currentPoseState);
            }

            if (!options.skipStatus) {
                setStatus('Video source stopped. You can still export the captured frames, or load a new source.', 'info');
            }
        }

        async function processFrame() {
            animationFrameId = requestAnimationFrame(processFrame);

            if (isProcessingFrame) return;
            if (!mediaStream && !isPlayingUploadedVideo) return;
            if (ui.sourceVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
            if (ui.sourceVideo.currentTime === lastProcessedVideoTime && !(isManualAssistBackend() && isRecording)) return;

            lastProcessedVideoTime = ui.sourceVideo.currentTime;
            isProcessingFrame = true;

            try {
                await ensurePoseBackend();
                const nowMs = performance.now();
                const detectionFrame = getPrimaryDetectionFrame();
                let posePackets = await detectPosePackets(detectionFrame.source, nowMs, { region: detectionFrame.region });

                if (getCaptureCharacterCount() > 1 && getPoseBackend() !== POSE_BACKENDS.MEDIAPIPE_VIDEO && !isManualAssistBackend()) {
                    const regionalPackets = await detectRegionalPosePackets(nowMs);
                    posePackets = mergePosePackets(posePackets, regionalPackets);
                }

                if (posePackets.length === 0) {
                    latestPosePackets = [];
                    if (ui.trackedState) ui.trackedState.textContent = 'Searching';
                    if (ui.confidenceValue) ui.confidenceValue.textContent = '0%';
                    clearOverlay();
                    return;
                }

                latestPosePackets = posePackets;
                if (ui.trackedState) ui.trackedState.textContent = getTrackedStateLabel(posePackets.length);
                if (ui.confidenceValue) ui.confidenceValue.textContent = `${Math.round(getAveragePoseConfidence(posePackets) * 100)}%`;
                drawOverlay(posePackets);

                const targetPose = clonePoseState(currentPoseState || defaultPoseState);
                let hasMappedPose = false;
                posePackets.forEach(packet => {
                    hasMappedPose = applyLandmarksToPose(targetPose, packet.landmarks, packet.worldLandmarks, packet.characterIndex) || hasMappedPose;
                });

                if (!hasMappedPose) return;

                currentPoseState = smoothPoseState(currentPoseState, targetPose, Number.parseFloat(ui.smoothing.value) || 0);
                applyPoseState(currentPoseState);
                samplePoseIfRecording(performance.now());
            } catch (error) {
                console.error(error);
                setStatus('Pose detection hit an error while reading the shared video.', 'error');
            } finally {
                isProcessingFrame = false;
            }
        }

        async function handleVideoFileSelected(event) {
            const file = event.target.files?.[0];
            if (!file) return;

            try {
                await ensurePoseBackend();

                if (mediaStream) {
                    stopScreenShare({ preservePose: true, skipStatus: true });
                }

                const videoUrl = URL.createObjectURL(file);
                ui.sourceVideo.src = videoUrl;
                ui.sourceVideo.controls = true;
                ui.videoMeta.textContent = file.name;
                isPlayingUploadedVideo = true;
                rootBaselines = [];
                lastProcessedVideoTime = -1;

                ui.sourceVideo.play().catch(err => {
                    console.error('Video playback error:', err);
                    setStatus('Could not play the video file. Try a different format (MP4, WebM, etc.).', 'error');
                });

                processFrame();
                setStatus('Video loaded. Play the video and let the skeleton lock onto the performer, then record.', 'success');

                ui.videoFileInput.value = '';
            } catch (error) {
                console.error(error);
                setStatus('Could not load the file.', 'error');
                ui.videoFileInput.value = '';
            }
        }

        function getPrimaryDetectionFrame() {
            const region = getActiveSourceRegion();
            if (!region) {
                return { source: ui.sourceVideo, region: null };
            }

            const sourceWidth = ui.sourceVideo.videoWidth || 0;
            const sourceHeight = ui.sourceVideo.videoHeight || 0;
            if (!sourceWidth || !sourceHeight || !drawRegionToScanCanvas(region, sourceWidth, sourceHeight)) {
                return { source: ui.sourceVideo, region: null };
            }

            return { source: regionScanCanvas, region };
        }

        function getActiveSourceRegion() {
            return isFocusCropEnabled() && sourceCropRect ? sourceCropRect : null;
        }

        async function detectPosePackets(source, nowMs, options = {}) {
            if (isManualAssistBackend()) {
                updateManualTrackedHandles();
                return getManualPosePackets({ skipTimelineSync: hasManualTrackTemplates() });
            }

            if (getPoseBackend() === POSE_BACKENDS.MOVENET) {
                const detector = await ensureMoveNetDetector();
                const poses = await detector.estimatePoses(source, {
                    maxPoses: getCaptureCharacterCount(),
                    flipHorizontal: false
                });
                return collectMoveNetPosePackets(poses, source, options);
            }

            const landmarker = await ensurePoseLandmarker(getMediaPipeRunningMode());
            const result = getMediaPipeRunningMode() === 'IMAGE'
                ? landmarker.detect(source)
                : landmarker.detectForVideo(source, nowMs);
            return collectPosePackets(result, options);
        }

        function collectPosePackets(result, options = {}) {
            const landmarksList = Array.isArray(result?.landmarks) ? result.landmarks : [];
            const worldLandmarksList = Array.isArray(result?.worldLandmarks) ? result.worldLandmarks : [];
            const maxCount = Number.isInteger(options.maxCount) ? options.maxCount : getCaptureCharacterCount();
            const assignCharacterIndices = options.assignCharacterIndices !== false;

            return landmarksList
                .map((landmarks, sourceIndex) => {
                    const mappedLandmarks = options.region ? mapRegionLandmarksToSource(landmarks, options.region) : landmarks;
                    return {
                        landmarks: mappedLandmarks,
                        worldLandmarks: worldLandmarksList[sourceIndex],
                        sortX: getPoseSortX(mappedLandmarks),
                        source: options.source || 'full'
                    };
                })
                .filter(packet => {
                    return Array.isArray(packet.landmarks)
                        && Array.isArray(packet.worldLandmarks)
                        && Number.isFinite(packet.sortX)
                        && hasUsablePoseLandmarks(packet.landmarks);
                })
                .sort((a, b) => a.sortX - b.sortX)
                .slice(0, maxCount)
                .map((packet, characterIndex) => ({
                    ...packet,
                    characterIndex: assignCharacterIndices ? characterIndex : -1
                }));
        }

        function collectMoveNetPosePackets(poses, source, options = {}) {
            const maxCount = Number.isInteger(options.maxCount) ? options.maxCount : getCaptureCharacterCount();
            const assignCharacterIndices = options.assignCharacterIndices !== false;
            const sourceSize = getDetectionSourceSize(source);

            return (Array.isArray(poses) ? poses : [])
                .map(pose => {
                    const localLandmarks = moveNetKeypointsToLandmarks(pose?.keypoints, sourceSize);
                    const landmarks = options.region ? mapRegionLandmarksToSource(localLandmarks, options.region) : localLandmarks;
                    return {
                        landmarks,
                        worldLandmarks: createWorldLandmarksFromImageLandmarks(landmarks),
                        sortX: getPoseSortX(landmarks),
                        source: 'movenet',
                        score: Number.isFinite(pose?.score) ? pose.score : getPoseConfidence(landmarks)
                    };
                })
                .filter(packet => Number.isFinite(packet.sortX) && hasUsablePoseLandmarks(packet.landmarks))
                .sort((a, b) => b.score - a.score)
                .slice(0, maxCount)
                .sort((a, b) => a.sortX - b.sortX)
                .map((packet, characterIndex) => ({
                    ...packet,
                    characterIndex: assignCharacterIndices ? characterIndex : -1
                }));
        }

        function moveNetKeypointsToLandmarks(keypoints, sourceSize) {
            const landmarks = Array.from({ length: 33 }, () => null);
            const width = sourceSize.width || 1;
            const height = sourceSize.height || 1;

            (Array.isArray(keypoints) ? keypoints : []).forEach((keypoint, index) => {
                const keypointName = keypoint?.name || getMoveNetKeypointName(index);
                const landmarkIndex = MOVENET_TO_MEDIAPIPE_LANDMARKS[keypointName];
                if (!Number.isInteger(landmarkIndex)) return;

                const score = Number.isFinite(keypoint.score) ? keypoint.score : 0;
                landmarks[landmarkIndex] = {
                    x: THREE.MathUtils.clamp((Number(keypoint.x) || 0) / width, 0, 1),
                    y: THREE.MathUtils.clamp((Number(keypoint.y) || 0) / height, 0, 1),
                    z: 0,
                    visibility: score,
                    presence: score
                };
            });

            return landmarks;
        }

        function getMoveNetKeypointName(index) {
            return [
                'nose',
                'left_eye',
                'right_eye',
                'left_ear',
                'right_ear',
                'left_shoulder',
                'right_shoulder',
                'left_elbow',
                'right_elbow',
                'left_wrist',
                'right_wrist',
                'left_hip',
                'right_hip',
                'left_knee',
                'right_knee',
                'left_ankle',
                'right_ankle'
            ][index];
        }

        function getDetectionSourceSize(source) {
            return {
                width: source?.videoWidth || source?.naturalWidth || source?.width || 1,
                height: source?.videoHeight || source?.naturalHeight || source?.height || 1
            };
        }

        function createWorldLandmarksFromImageLandmarks(landmarks) {
            return (Array.isArray(landmarks) ? landmarks : []).map(landmark => {
                if (!landmark) return null;
                return {
                    x: landmark.x,
                    y: landmark.y,
                    z: landmark.z || 0,
                    visibility: landmark.visibility ?? 0,
                    presence: landmark.presence ?? landmark.visibility ?? 0
                };
            });
        }

        function ensureManualHandles() {
            const characterCount = getCaptureCharacterCount();
            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                if (!manualHandles[characterIndex]) {
                    manualHandles[characterIndex] = createDefaultManualHandles(characterIndex, characterCount);
                }
            }
            manualHandles.length = characterCount;
        }

        function createDefaultManualHandles(characterIndex, characterCount = getCaptureCharacterCount()) {
            const centerX = characterCount > 1
                ? 0.34 + characterIndex * 0.32
                : 0.5;
            const scale = characterCount > 1 ? 0.86 : 1;

            return {
                head: { x: centerX, y: 0.24 },
                leftShoulder: { x: centerX - 0.09 * scale, y: 0.38 },
                rightShoulder: { x: centerX + 0.09 * scale, y: 0.38 },
                leftElbow: { x: centerX - 0.15 * scale, y: 0.52 },
                rightElbow: { x: centerX + 0.15 * scale, y: 0.52 },
                leftWrist: { x: centerX - 0.19 * scale, y: 0.66 },
                rightWrist: { x: centerX + 0.19 * scale, y: 0.66 }
            };
        }

        function cloneManualHandles(source = manualHandles) {
            return (Array.isArray(source) ? source : []).map(handles => {
                const cloned = {};
                Object.entries(handles || {}).forEach(([handleId, point]) => {
                    cloned[handleId] = {
                        x: Number(point?.x) || 0,
                        y: Number(point?.y) || 0
                    };
                });
                return cloned;
            });
        }

        function getManualTimelineTime() {
            const time = Number.parseFloat(ui.sourceVideo.currentTime);
            return roundTime(Number.isFinite(time) ? time : 0);
        }

        function addManualHandleKeyframe(options = {}) {
            if (!isManualAssistBackend()) return;

            ensureManualHandles();
            const time = getManualTimelineTime();
            const existingIndex = manualHandleKeyframes.findIndex(keyframe => Math.abs(keyframe.time - time) < 1e-6);
            const keyframe = {
                time,
                handles: cloneManualHandles()
            };

            if (existingIndex >= 0) {
                manualHandleKeyframes.splice(existingIndex, 1, keyframe);
            } else {
                manualHandleKeyframes.push(keyframe);
            }

            manualHandleKeyframes.sort((a, b) => a.time - b.time);
            updateManualAssistKeyUi();
            applyManualPoseFromHandles({ skipTimelineSync: true });
            if (!options.quiet) {
                setStatus(`Manual assist key saved at ${time.toFixed(1)}s. Add more keys later in the clip so handles can follow the motion.`, 'success');
            }
        }

        function clearManualHandleKeyframes() {
            manualHandleKeyframes = [];
            updateManualAssistKeyUi();
            setStatus('Manual assist keys cleared. Locked handles will keep tracking until you reset them.', 'info');
        }

        function updateManualAssistKeyUi() {
            if (!ui.assistKeyCount) return;
            ui.assistKeyCount.textContent = String(manualHandleKeyframes.length);
        }

        function syncManualHandlesToTimeline() {
            if (!isManualAssistBackend() || manualDragState || manualHandleKeyframes.length === 0) return;
            manualHandles = getManualHandlesAtTime(getManualTimelineTime());
        }

        function getManualHandlesAtTime(time) {
            ensureManualHandles();
            if (manualHandleKeyframes.length === 0) {
                return cloneManualHandles();
            }

            if (manualHandleKeyframes.length === 1 || time <= manualHandleKeyframes[0].time) {
                return cloneManualHandles(manualHandleKeyframes[0].handles);
            }

            const lastKeyframe = manualHandleKeyframes[manualHandleKeyframes.length - 1];
            if (time >= lastKeyframe.time) {
                return cloneManualHandles(lastKeyframe.handles);
            }

            for (let index = 1; index < manualHandleKeyframes.length; index += 1) {
                const next = manualHandleKeyframes[index];
                if (time > next.time) continue;

                const previous = manualHandleKeyframes[index - 1];
                const span = Math.max(0.0001, next.time - previous.time);
                const alpha = THREE.MathUtils.clamp((time - previous.time) / span, 0, 1);
                return interpolateManualHandles(previous.handles, next.handles, alpha);
            }

            return cloneManualHandles(lastKeyframe.handles);
        }

        function interpolateManualHandles(startHandles, endHandles, alpha) {
            const characterCount = getCaptureCharacterCount();
            const interpolated = [];

            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                const characterHandles = {};
                MANUAL_HANDLE_DEFS.forEach(def => {
                    const startPoint = startHandles?.[characterIndex]?.[def.id]
                        || manualHandles?.[characterIndex]?.[def.id]
                        || createDefaultManualHandles(characterIndex, characterCount)[def.id];
                    const endPoint = endHandles?.[characterIndex]?.[def.id] || startPoint;
                    characterHandles[def.id] = {
                        x: THREE.MathUtils.lerp(startPoint.x, endPoint.x, alpha),
                        y: THREE.MathUtils.lerp(startPoint.y, endPoint.y, alpha)
                    };
                });
                interpolated.push(characterHandles);
            }

            return interpolated;
        }

        function ensureManualTrackTemplates() {
            const characterCount = getCaptureCharacterCount();
            for (let characterIndex = 0; characterIndex < characterCount; characterIndex += 1) {
                if (!manualTrackTemplates[characterIndex]) {
                    manualTrackTemplates[characterIndex] = {};
                }
            }
            manualTrackTemplates.length = characterCount;
        }

        function hasManualTrackTemplates() {
            return manualTrackTemplates.some(characterTemplates =>
                Object.values(characterTemplates || {}).some(Boolean)
            );
        }

        function clearManualTrackTemplates() {
            manualTrackTemplates = [];
            ensureManualTrackTemplates();
        }

        function getManualTrackFrame() {
            const sourceWidth = ui.sourceVideo.videoWidth || 0;
            const sourceHeight = ui.sourceVideo.videoHeight || 0;
            if (!sourceWidth || !sourceHeight || ui.sourceVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
                return null;
            }

            if (!manualTrackCanvas) {
                manualTrackCanvas = document.createElement('canvas');
                manualTrackContext = manualTrackCanvas.getContext('2d', { willReadFrequently: true });
            }
            if (!manualTrackContext) return null;

            const scale = Math.min(1, MANUAL_TRACK_MAX_WIDTH / sourceWidth);
            const width = Math.max(1, Math.round(sourceWidth * scale));
            const height = Math.max(1, Math.round(sourceHeight * scale));

            if (manualTrackCanvas.width !== width || manualTrackCanvas.height !== height) {
                manualTrackCanvas.width = width;
                manualTrackCanvas.height = height;
            }

            try {
                manualTrackContext.drawImage(ui.sourceVideo, 0, 0, width, height);
                return {
                    width,
                    height,
                    data: manualTrackContext.getImageData(0, 0, width, height).data
                };
            } catch (error) {
                console.warn('Manual tracker could not read the current video frame.', error);
                return null;
            }
        }

        function sampleManualTrackPatch(frame, centerX, centerY, radius = MANUAL_TRACK_PATCH_RADIUS) {
            const diameter = radius * 2 + 1;
            const pixels = new Float32Array(diameter * diameter);
            let writeIndex = 0;

            for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
                const y = THREE.MathUtils.clamp(Math.round(centerY + offsetY), 0, frame.height - 1);
                for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
                    const x = THREE.MathUtils.clamp(Math.round(centerX + offsetX), 0, frame.width - 1);
                    const sourceIndex = (y * frame.width + x) * 4;
                    pixels[writeIndex] = (
                        frame.data[sourceIndex] * 0.299
                        + frame.data[sourceIndex + 1] * 0.587
                        + frame.data[sourceIndex + 2] * 0.114
                    );
                    writeIndex += 1;
                }
            }

            return pixels;
        }

        function scoreManualTrackPatch(frame, template, centerX, centerY) {
            const radius = template.radius || MANUAL_TRACK_PATCH_RADIUS;
            const candidate = sampleManualTrackPatch(frame, centerX, centerY, radius);
            let totalDelta = 0;

            for (let index = 0; index < candidate.length; index += 1) {
                totalDelta += Math.abs(candidate[index] - template.pixels[index]);
            }

            return {
                score: totalDelta / candidate.length,
                pixels: candidate
            };
        }

        function captureManualTrackTemplate(characterIndex, handleId) {
            ensureManualHandles();
            ensureManualTrackTemplates();

            const point = manualHandles?.[characterIndex]?.[handleId];
            const frame = point ? getManualTrackFrame() : null;
            if (!point || !frame) return false;

            const centerX = point.x * (frame.width - 1);
            const centerY = point.y * (frame.height - 1);
            manualTrackTemplates[characterIndex][handleId] = {
                radius: MANUAL_TRACK_PATCH_RADIUS,
                pixels: sampleManualTrackPatch(frame, centerX, centerY),
                lastScore: 0
            };
            return true;
        }

        function updateManualTrackedHandles() {
            if (!isManualAssistBackend() || manualDragState || !hasManualTrackTemplates()) return false;

            ensureManualHandles();
            ensureManualTrackTemplates();

            const frame = getManualTrackFrame();
            if (!frame) return false;

            let moved = false;
            manualTrackTemplates.forEach((characterTemplates, characterIndex) => {
                Object.entries(characterTemplates || {}).forEach(([handleId, template]) => {
                    const point = manualHandles?.[characterIndex]?.[handleId];
                    if (!point || !template?.pixels) return;

                    const centerX = point.x * (frame.width - 1);
                    const centerY = point.y * (frame.height - 1);
                    const searchRadius = Math.max(8, MANUAL_TRACK_SEARCH_RADIUS);
                    let best = {
                        x: centerX,
                        y: centerY,
                        score: Number.POSITIVE_INFINITY,
                        pixels: null
                    };

                    for (let offsetY = -searchRadius; offsetY <= searchRadius; offsetY += MANUAL_TRACK_SEARCH_STEP) {
                        const candidateY = THREE.MathUtils.clamp(centerY + offsetY, 0, frame.height - 1);
                        for (let offsetX = -searchRadius; offsetX <= searchRadius; offsetX += MANUAL_TRACK_SEARCH_STEP) {
                            const candidateX = THREE.MathUtils.clamp(centerX + offsetX, 0, frame.width - 1);
                            const match = scoreManualTrackPatch(frame, template, candidateX, candidateY);
                            if (match.score >= best.score) continue;

                            best = {
                                x: candidateX,
                                y: candidateY,
                                score: match.score,
                                pixels: match.pixels
                            };
                        }
                    }

                    template.lastScore = best.score;
                    if (best.score > MANUAL_TRACK_MAX_SCORE || !best.pixels) return;

                    const nextPoint = {
                        x: THREE.MathUtils.clamp(best.x / Math.max(1, frame.width - 1), 0, 1),
                        y: THREE.MathUtils.clamp(best.y / Math.max(1, frame.height - 1), 0, 1)
                    };

                    if (Math.hypot(nextPoint.x - point.x, nextPoint.y - point.y) > 0.0005) {
                        manualHandles[characterIndex][handleId] = nextPoint;
                        moved = true;
                    }

                    for (let index = 0; index < template.pixels.length; index += 1) {
                        template.pixels[index] = THREE.MathUtils.lerp(
                            template.pixels[index],
                            best.pixels[index],
                            MANUAL_TRACK_TEMPLATE_BLEND
                        );
                    }
                });
            });

            return moved;
        }

        function resetManualHandles() {
            manualHandles = [];
            manualHandleKeyframes = [];
            clearManualTrackTemplates();
            ensureManualHandles();
            updateManualAssistKeyUi();
            lastProcessedVideoTime = -1;
            applyManualPoseFromHandles({ skipTimelineSync: true });
            setStatus('Manual handles reset. Drag the upper-body points over the character silhouette to lock fresh tracking handles.', 'info');
        }

        function getActiveManualCharacterIndex() {
            const value = Number.parseInt(ui.assistCharacter?.value, 10);
            return THREE.MathUtils.clamp(Number.isFinite(value) ? value : 0, 0, getCaptureCharacterCount() - 1);
        }

        function getManualPosePackets(options = {}) {
            ensureManualHandles();
            if (!options.skipTimelineSync && !hasManualTrackTemplates()) {
                syncManualHandlesToTimeline();
            }
            return manualHandles.map((handles, characterIndex) => {
                const landmarks = manualHandlesToLandmarks(handles);
                return {
                    landmarks,
                    worldLandmarks: createWorldLandmarksFromImageLandmarks(landmarks),
                    sortX: getPoseSortX(landmarks),
                    source: 'manual',
                    score: 1,
                    characterIndex
                };
            });
        }

        function manualHandlesToLandmarks(handles) {
            const landmarks = Array.from({ length: 33 }, () => null);

            MANUAL_HANDLE_DEFS.forEach(def => {
                const point = handles?.[def.id];
                if (!point) return;
                landmarks[def.landmark] = {
                    x: THREE.MathUtils.clamp(point.x, 0, 1),
                    y: THREE.MathUtils.clamp(point.y, 0, 1),
                    z: 0,
                    visibility: 1,
                    presence: 1
                };
            });

            const shoulderCenter = midpointLandmark(landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]);
            if (shoulderCenter) {
                landmarks[LM.LEFT_HIP] = {
                    x: THREE.MathUtils.clamp(shoulderCenter.x - 0.06, 0, 1),
                    y: THREE.MathUtils.clamp(shoulderCenter.y + 0.34, 0, 1),
                    z: 0,
                    visibility: 1,
                    presence: 1
                };
                landmarks[LM.RIGHT_HIP] = {
                    x: THREE.MathUtils.clamp(shoulderCenter.x + 0.06, 0, 1),
                    y: THREE.MathUtils.clamp(shoulderCenter.y + 0.34, 0, 1),
                    z: 0,
                    visibility: 1,
                    presence: 1
                };
            }

            return landmarks;
        }

        async function detectRegionalPosePackets(nowMs) {
            const sourceWidth = ui.sourceVideo.videoWidth || 0;
            const sourceHeight = ui.sourceVideo.videoHeight || 0;
            if (!sourceWidth || !sourceHeight) return [];

            const packets = [];
            const regions = getRegionalScanRegions(getActiveSourceRegion());
            for (const [regionIndex, region] of regions.entries()) {
                if (!drawRegionToScanCanvas(region, sourceWidth, sourceHeight)) continue;

                const regionPackets = await detectPosePackets(regionScanCanvas, nowMs + (regionIndex + 1) * 0.1, {
                    region,
                    source: 'region',
                    maxCount: MAX_CAPTURE_CHARACTERS,
                    assignCharacterIndices: false
                });
                const expectedX = region.characterIndex === 0 ? 0.32 : 0.68;
                const selected = regionPackets
                    .sort((a, b) => Math.abs(a.sortX - expectedX) - Math.abs(b.sortX - expectedX))[0];

                if (selected) {
                    packets.push({ ...selected, characterIndex: region.characterIndex });
                }
            }

            return packets;
        }

        function getRegionalScanRegions(baseRegion = null) {
            const sourceRegion = baseRegion || { x: 0, y: 0, width: 1, height: 1 };
            return [
                {
                    characterIndex: 0,
                    x: sourceRegion.x,
                    y: sourceRegion.y,
                    width: sourceRegion.width * 0.62,
                    height: sourceRegion.height
                },
                {
                    characterIndex: 1,
                    x: sourceRegion.x + sourceRegion.width * 0.38,
                    y: sourceRegion.y,
                    width: sourceRegion.width * 0.62,
                    height: sourceRegion.height
                }
            ];
        }

        function drawRegionToScanCanvas(region, sourceWidth, sourceHeight) {
            if (!regionScanCanvas) {
                regionScanCanvas = document.createElement('canvas');
                regionScanContext = regionScanCanvas.getContext('2d', { willReadFrequently: true });
            }

            if (!regionScanContext) return false;

            const sourceX = Math.max(0, Math.floor(region.x * sourceWidth));
            const sourceY = Math.max(0, Math.floor(region.y * sourceHeight));
            const sourceRegionWidth = Math.min(sourceWidth - sourceX, Math.ceil(region.width * sourceWidth));
            const sourceRegionHeight = Math.min(sourceHeight - sourceY, Math.ceil(region.height * sourceHeight));
            if (sourceRegionWidth <= 0 || sourceRegionHeight <= 0) return false;
            const targetWidth = Math.round(THREE.MathUtils.clamp(sourceRegionWidth, REGION_SCAN_MIN_WIDTH, REGION_SCAN_MAX_WIDTH));
            const targetHeight = Math.max(1, Math.round(sourceRegionHeight * (targetWidth / sourceRegionWidth)));

            if (regionScanCanvas.width !== targetWidth || regionScanCanvas.height !== targetHeight) {
                regionScanCanvas.width = targetWidth;
                regionScanCanvas.height = targetHeight;
            }

            regionScanContext.clearRect(0, 0, targetWidth, targetHeight);
            regionScanContext.drawImage(
                ui.sourceVideo,
                sourceX,
                sourceY,
                sourceRegionWidth,
                sourceRegionHeight,
                0,
                0,
                targetWidth,
                targetHeight
            );
            return true;
        }

        function mapRegionLandmarksToSource(landmarks, region) {
            return (Array.isArray(landmarks) ? landmarks : []).map(landmark => {
                if (!landmark) return landmark;
                return {
                    ...landmark,
                    x: region.x + landmark.x * region.width,
                    y: region.y + landmark.y * region.height
                };
            });
        }

        function mergePosePackets(fullPackets, regionalPackets) {
            const slots = new Array(getCaptureCharacterCount()).fill(null);

            regionalPackets.forEach(packet => {
                if (!Number.isInteger(packet.characterIndex)) return;
                if (packet.characterIndex < 0 || packet.characterIndex >= slots.length) return;
                if (hasDuplicatePosePacket(packet, slots)) return;
                slots[packet.characterIndex] = packet;
            });

            fullPackets.forEach(packet => {
                if (hasDuplicatePosePacket(packet, slots)) return;
                const characterIndex = getBestOpenCharacterSlot(packet, slots);
                if (characterIndex < 0) return;
                slots[characterIndex] = { ...packet, characterIndex };
            });

            return slots.filter(Boolean).sort((a, b) => a.characterIndex - b.characterIndex);
        }

        function hasDuplicatePosePacket(packet, slots) {
            return slots.some(existing => {
                return !!existing && Math.abs(existing.sortX - packet.sortX) < DUPLICATE_POSE_X_THRESHOLD;
            });
        }

        function getBestOpenCharacterSlot(packet, slots) {
            const preferredSlot = slots.length > 1 && packet.sortX >= 0.5 ? 1 : 0;
            if (preferredSlot < slots.length && !slots[preferredSlot]) return preferredSlot;
            return slots.findIndex(slot => !slot);
        }

        function getPoseSortX(landmarks) {
            const coreLandmarks = [
                landmarks?.[LM.LEFT_SHOULDER],
                landmarks?.[LM.RIGHT_SHOULDER],
                isUpperBodyOnly() ? landmarks?.[LM.NOSE] : landmarks?.[LM.LEFT_HIP],
                isUpperBodyOnly() ? landmarks?.[LM.LEFT_EAR] : landmarks?.[LM.RIGHT_HIP],
                isUpperBodyOnly() ? landmarks?.[LM.RIGHT_EAR] : null
            ];
            const values = coreLandmarks
                .filter(isReliableLandmark)
                .map(landmark => landmark?.x)
                .filter(Number.isFinite);

            if (values.length === 0) return Number.NaN;
            return values.reduce((sum, value) => sum + value, 0) / values.length;
        }

        function hasUsablePoseLandmarks(landmarks) {
            if (!isUpperBodyOnly()) {
                return hasReliablePair(landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
                    && hasReliablePair(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP);
            }

            const hasShoulderPair = hasReliablePair(landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER);
            const hasAnyShoulder = hasAnyReliableLandmark(landmarks, [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER]);
            const hasHeadAnchor = hasAnyReliableLandmark(landmarks, [LM.NOSE, LM.LEFT_EAR, LM.RIGHT_EAR]);
            const hasArmChain = hasReliablePair(landmarks, LM.LEFT_SHOULDER, LM.LEFT_ELBOW)
                || hasReliablePair(landmarks, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW);

            return hasShoulderPair || (hasAnyShoulder && hasHeadAnchor) || hasArmChain;
        }

        function hasAnyReliableLandmark(landmarks, indices) {
            return indices.some(index => isReliableLandmark(landmarks[index]));
        }

        function getTrackedStateLabel(trackedCount) {
            const requestedCount = getCaptureCharacterCount();
            return requestedCount > 1 ? `${trackedCount}/${requestedCount} Locked` : 'Locked';
        }

        function getAveragePoseConfidence(posePackets) {
            if (!posePackets.length) return 0;
            const total = posePackets.reduce((sum, packet) => sum + getPoseConfidence(packet.landmarks), 0);
            return total / posePackets.length;
        }

        function applyLandmarksToPose(pose, landmarks, worldLandmarks, characterIndex) {
            const upperBodyOnly = isUpperBodyOnly();
            const hasShoulders = hasReliablePair(landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER);
            const hasHips = hasReliablePair(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP);

            if (!hasUsablePoseLandmarks(landmarks) || (!upperBodyOnly && !hasHips)) {
                return false;
            }

            if (upperBodyOnly) {
                resetStandingLowerBody(pose, characterIndex);
            }

            const worldQuaternionMap = getWorldQuaternionMap(pose, characterIndex);
            const world = worldLandmarks.map(toWorldVector);
            const joint = baseName => getJointName(baseName, characterIndex);

            const hipsCenter = hasHips ? getReliableWorldCenter(world, landmarks, [LM.LEFT_HIP, LM.RIGHT_HIP]) : null;
            const shouldersCenter = getReliableWorldCenter(world, landmarks, [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER])
                || getReliableWorldCenter(world, landmarks, [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW, LM.RIGHT_ELBOW])
                || new THREE.Vector3(0, 1, 0);
            const torsoUp = hasHips
                ? directionBetween(hipsCenter, shouldersCenter)
                : estimateUpperBodyUp(world, shouldersCenter);
            const bodyLeft = averageDirection([
                hasHips ? directionBetween(world[LM.RIGHT_HIP], world[LM.LEFT_HIP]) : null,
                hasShoulders ? directionBetween(world[LM.RIGHT_SHOULDER], world[LM.LEFT_SHOULDER]) : null,
                new THREE.Vector3(1, 0, 0)
            ]);

            if (torsoUp && bodyLeft) {
                const hipsWorldQuaternion = quaternionFromBasis(bodyLeft, torsoUp);
                if (!upperBodyOnly) {
                    setWorldQuaternionOnPose(pose, worldQuaternionMap, joint('Hips'), hipsWorldQuaternion);
                }
                setWorldQuaternionOnPose(pose, worldQuaternionMap, joint('Spine'), hipsWorldQuaternion);
            }

            const headLeft = averageDirection([
                directionBetween(world[LM.RIGHT_EAR], world[LM.LEFT_EAR]),
                directionBetween(world[LM.RIGHT_SHOULDER], world[LM.LEFT_SHOULDER])
            ]);
            const headUp = averageDirection([
                directionBetween(shouldersCenter, world[LM.NOSE]),
                torsoUp
            ]);
            if (headLeft && headUp) {
                const headWorldQuaternion = quaternionFromBasis(headLeft, headUp);
                setWorldQuaternionOnPose(pose, worldQuaternionMap, joint('Head'), headWorldQuaternion);
            }

            applyLimbDirection(pose, worldQuaternionMap, joint('Left_Upper_Arm'), landmarkDirection(world, landmarks, LM.LEFT_SHOULDER, LM.LEFT_ELBOW));
            applyLimbDirection(pose, worldQuaternionMap, joint('Left_Lower_Arm'), landmarkDirection(world, landmarks, LM.LEFT_ELBOW, LM.LEFT_WRIST));
            applyLimbDirection(pose, worldQuaternionMap, joint('Right_Upper_Arm'), landmarkDirection(world, landmarks, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW));
            applyLimbDirection(pose, worldQuaternionMap, joint('Right_Lower_Arm'), landmarkDirection(world, landmarks, LM.RIGHT_ELBOW, LM.RIGHT_WRIST));

            if (!upperBodyOnly) {
                applyLimbDirection(pose, worldQuaternionMap, joint('Left_Upper_Leg'), landmarkDirection(world, landmarks, LM.LEFT_HIP, LM.LEFT_KNEE));
                applyLimbDirection(pose, worldQuaternionMap, joint('Left_Lower_Leg'), landmarkDirection(world, landmarks, LM.LEFT_KNEE, LM.LEFT_ANKLE));
                applyLimbDirection(pose, worldQuaternionMap, joint('Right_Upper_Leg'), landmarkDirection(world, landmarks, LM.RIGHT_HIP, LM.RIGHT_KNEE));
                applyLimbDirection(pose, worldQuaternionMap, joint('Right_Lower_Leg'), landmarkDirection(world, landmarks, LM.RIGHT_KNEE, LM.RIGHT_ANKLE));
            }

            pose[joint('Hips')].position.copy(upperBodyOnly ? getCharacterBasePosition(characterIndex) : computeRootPosition(landmarks, characterIndex));
            return true;
        }

        function resetStandingLowerBody(pose, characterIndex) {
            STANDING_LOWER_BODY_JOINTS.forEach(baseName => {
                const name = getJointName(baseName, characterIndex);
                const standingTransform = defaultPoseState?.[name];
                if (!pose[name] || !standingTransform) return;
                pose[name].position.copy(standingTransform.position);
                pose[name].quaternion.copy(standingTransform.quaternion);
            });
            const hipsName = getJointName('Hips', characterIndex);
            pose[hipsName]?.position.copy(getCharacterBasePosition(characterIndex));
        }

        function getReliableWorldCenter(world, landmarks, indices) {
            const center = new THREE.Vector3();
            let count = 0;

            indices.forEach(index => {
                if (!isReliableLandmark(landmarks[index]) || !world[index]) return;
                center.add(world[index]);
                count += 1;
            });

            return count > 0 ? center.multiplyScalar(1 / count) : null;
        }

        function estimateUpperBodyUp(world, shouldersCenter) {
            return averageDirection([
                directionBetween(shouldersCenter, world[LM.NOSE]),
                directionBetween(shouldersCenter, midpointVector(world[LM.LEFT_EAR], world[LM.RIGHT_EAR])),
                new THREE.Vector3(0, 1, 0)
            ]);
        }

        function landmarkDirection(world, landmarks, startIndex, endIndex) {
            if (!isReliableLandmark(landmarks[startIndex]) || !isReliableLandmark(landmarks[endIndex])) return null;
            return directionBetween(world[startIndex], world[endIndex]);
        }

        function applyLimbDirection(pose, worldQuaternionMap, jointName, direction) {
            if (!direction) return;
            const worldQuaternion = new THREE.Quaternion().setFromUnitVectors(DOWN_AXIS, direction);
            setWorldQuaternionOnPose(pose, worldQuaternionMap, jointName, worldQuaternion);
        }

        function computeRootPosition(landmarks, characterIndex) {
            const rootPosition = getCharacterBasePosition(characterIndex);

            if (!ui.rootMotion.checked || isUpperBodyOnly()) {
                return rootPosition;
            }

            const leftHip = landmarks[LM.LEFT_HIP];
            const rightHip = landmarks[LM.RIGHT_HIP];
            const leftShoulder = landmarks[LM.LEFT_SHOULDER];
            const rightShoulder = landmarks[LM.RIGHT_SHOULDER];

            if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) {
                return rootPosition;
            }

            const hipCenter = midpointLandmark(leftHip, rightHip);
            const shoulderSpan = distance2D(leftShoulder, rightShoulder);

            if (!rootBaselines[characterIndex]) {
                rootBaselines[characterIndex] = {
                    x: hipCenter.x,
                    y: hipCenter.y,
                    shoulderSpan: shoulderSpan || 0.2
                };
            }

            const rootBaseline = rootBaselines[characterIndex];
            const deltaX = (hipCenter.x - rootBaseline.x) * 8;
            const deltaY = (rootBaseline.y - hipCenter.y) * 10;
            const depthDelta = (shoulderSpan - rootBaseline.shoulderSpan) * 9;

            rootPosition.x = THREE.MathUtils.clamp(rootPosition.x + deltaX, -4.5, 4.5);
            rootPosition.y = THREE.MathUtils.clamp(ROOT_BASE_POSITION.y + deltaY, 1.1, 6.4);
            rootPosition.z = THREE.MathUtils.clamp(ROOT_BASE_POSITION.z + depthDelta, -3.25, 3.25);

            return rootPosition;
        }

        function smoothPoseState(previousPose, nextPose, smoothing) {
            if (!previousPose) {
                return clonePoseState(nextPose);
            }

            const alpha = THREE.MathUtils.clamp(1 - smoothing, 0.05, 1);
            const pose = clonePoseState(previousPose);

            Object.entries(nextPose).forEach(([name, transform]) => {
                if (!transform?.position || !transform?.quaternion) return;
                if (!pose[name]) {
                    pose[name] = {
                        position: transform.position.clone(),
                        quaternion: transform.quaternion.clone()
                    };
                    return;
                }

                pose[name].position.lerp(transform.position, alpha);
                pose[name].quaternion.slerp(transform.quaternion, alpha).normalize();
            });

            return pose;
        }

        function getWorldQuaternionMap(pose, characterIndex) {
            const map = {};

            BASE_JOINT_ORDER.forEach(baseName => {
                const name = getJointName(baseName, characterIndex);
                const parentBaseName = BASE_JOINT_PARENTS[baseName];
                const parentName = parentBaseName ? getJointName(parentBaseName, characterIndex) : null;
                if (!parentName) {
                    map[name] = pose[name].quaternion.clone();
                    return;
                }

                map[name] = map[parentName].clone().multiply(pose[name].quaternion);
            });

            return map;
        }

        function setWorldQuaternionOnPose(pose, worldQuaternionMap, jointName, worldQuaternion) {
            const parsedJointName = parseJointName(jointName);
            const parentBaseName = parsedJointName ? BASE_JOINT_PARENTS[parsedJointName.baseName] : null;
            const parentName = parentBaseName ? getJointName(parentBaseName, parsedJointName.characterIndex) : null;

            if (parentName) {
                pose[jointName].quaternion.copy(worldQuaternionMap[parentName].clone().invert().multiply(worldQuaternion)).normalize();
            } else {
                pose[jointName].quaternion.copy(worldQuaternion).normalize();
            }

            worldQuaternionMap[jointName] = worldQuaternion.clone();
        }

        function quaternionFromBasis(leftAxis, upAxis) {
            const yAxis = upAxis.clone().normalize();
            let xAxis = leftAxis.clone();
            xAxis.sub(yAxis.clone().multiplyScalar(xAxis.dot(yAxis)));

            if (xAxis.lengthSq() < 1e-8) {
                xAxis = Math.abs(yAxis.y) < 0.95
                    ? new THREE.Vector3(0, 1, 0)
                    : new THREE.Vector3(0, 0, 1);
                xAxis.sub(yAxis.clone().multiplyScalar(xAxis.dot(yAxis)));
            }

            if (xAxis.lengthSq() < 1e-8) {
                return new THREE.Quaternion();
            }

            xAxis.normalize();
            const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
            const correctedXAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();
            const matrix = new THREE.Matrix4().makeBasis(correctedXAxis, yAxis, zAxis);
            return new THREE.Quaternion().setFromRotationMatrix(matrix);
        }

        function samplePoseIfRecording(nowMs) {
            if (!isRecording || !currentPoseState) return;

            const interval = 1 / (Number.parseInt(ui.sampleRate.value, 10) || 10);
            const elapsedSeconds = (nowMs - recordingStartedAt) / 1000;

            if (elapsedSeconds + 1e-6 < lastSampledAt + interval) {
                return;
            }

            const roundedTime = roundTime(elapsedSeconds);
            const serializedPose = serializePose(currentPoseState);
            const lastFrame = recordedFrames[recordedFrames.length - 1];

            if (lastFrame && Math.abs(lastFrame.time - roundedTime) < 1e-6) {
                lastFrame.pose = serializedPose;
            } else {
                recordedFrames.push({
                    time: roundedTime,
                    pose: serializedPose
                });
            }

            lastSampledAt = roundedTime;
            updateStats();
        }

        function toggleRecording() {
            if (isRecording) {
                stopRecording();
                return;
            }

            if (isManualAssistBackend()) {
                applyManualPoseFromHandles();
            }

            const requiredCharacterCount = getCaptureCharacterCount();
            if (latestPosePackets.length < requiredCharacterCount) {
                setStatus(
                    requiredCharacterCount > 1
                        ? `Wait for ${requiredCharacterCount} tracked performers before recording.`
                        : 'Wait for a tracked performer before recording.',
                    'error'
                );
                return;
            }

            if (!ui.animationName.value.trim()) {
                setGeneratedAnimationName();
            }

            recordedFrames = [];
            isRecording = true;
            recordingStartedAt = performance.now();
            lastSampledAt = -Infinity;
            if (ui.rootMotion.checked) {
                captureNeutralPose({ quiet: true });
            }
            samplePoseIfRecording(recordingStartedAt);
            updateRecordingUi();
            setStatus('Recording started. Let the reference motion play, then click Stop.', 'success');
        }

        function stopRecording() {
            isRecording = false;
            updateRecordingUi();
            updateStats();
            setStatus(
                recordedFrames.length > 0
                    ? `Recording stopped with ${recordedFrames.length} captured keyframes.`
                    : 'Recording stopped before any keyframes were captured.',
                recordedFrames.length > 0 ? 'success' : 'info'
            );
        }

        function clearRecording() {
            isRecording = false;
            recordedFrames = [];
            lastSampledAt = -Infinity;
            rootBaselines = [];
            currentPoseState = clonePoseState(defaultPoseState);
            applyPoseState(currentPoseState);
            updateRecordingUi();
            updateStats();
            setStatus('Capture cleared. Share a source again or keep the current stream and record a new take.', 'info');
        }

        function captureNeutralPose(options = {}) {
            if (isUpperBodyOnly()) {
                rootBaselines = [];
                if (!options.quiet) {
                    setStatus('Upper-body source mode keeps hips and legs planted, so neutral root motion is skipped.', 'info');
                }
                return;
            }

            if (latestPosePackets.length === 0) {
                if (!options.quiet) {
                    setStatus('A tracked performer is required before setting the neutral pose.', 'error');
                }
                return;
            }

            rootBaselines = [];
            latestPosePackets.forEach(packet => {
                const { landmarks, characterIndex } = packet;
                const hipCenter = midpointLandmark(landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]);
                const shoulderSpan = distance2D(landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]);

                rootBaselines[characterIndex] = {
                    x: hipCenter.x,
                    y: hipCenter.y,
                    shoulderSpan: shoulderSpan || 0.2
                };
            });

            if (!options.quiet) {
                setStatus('Neutral pose captured. Root motion will now be measured relative to this frame.', 'success');
            }
        }

        function createAnimationAsset() {
            if (recordedFrames.length === 0) {
                throw new Error('Record at least one keyframe before saving or exporting.');
            }

            const name = ensureAnimationName();
            const uniqueFrames = new Map();
            recordedFrames.forEach(frame => {
                uniqueFrames.set(frame.time.toFixed(1), {
                    time: roundTime(frame.time),
                    pose: frame.pose
                });
            });

            return {
                format: ASSET_FORMAT,
                version: ASSET_VERSION,
                type: 'animation',
                name,
                savedAt: new Date().toISOString(),
                scene: {
                    characterCount: getCaptureCharacterCount(),
                    characterColors: getCharacterColors()
                },
                playbackSpeed: 1,
                keyframes: Array.from(uniqueFrames.values()).sort((a, b) => a.time - b.time)
            };
        }



        function saveAnimationToLibrary() {
            try {
                const asset = createAnimationAsset();
                const raw = window.localStorage.getItem(MAIN_ANIMATION_STORAGE_KEY);
                const library = JSON.parse(raw || '[]');
                const safeLibrary = Array.isArray(library) ? library : [];
                const existingIndex = safeLibrary.findIndex(entry => entry?.name?.toLowerCase?.() === asset.name.toLowerCase());

                if (existingIndex >= 0) {
                    safeLibrary.splice(existingIndex, 1);
                }

                safeLibrary.unshift(asset);
                window.localStorage.setItem(MAIN_ANIMATION_STORAGE_KEY, JSON.stringify(safeLibrary));
                setStatus(`Animation "${asset.name}" was saved into the shared browser library. Open Index.html and load it from Animation Library.`, 'success');
            } catch (error) {
                console.error(error);
                setStatus('Could not save into the shared browser library.', 'error');
            }
        }

        function downloadAssetFile(asset) {
            const blob = new Blob([JSON.stringify(asset, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const safeName = asset.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'motion-rip';

            link.href = url;
            link.download = `${safeName}.animation.json`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        }

        function exportAnimationAsHtml() {
            try {
                const asset = createAnimationAsset();
                const htmlContent = generateStandaloneHtml(asset);

                const blob = new Blob([htmlContent], { type: 'text/html' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                const safeName = asset.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'motion-rip';

                link.href = url;
                link.download = `${safeName}.html`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);

                setStatus(`Animation "${asset.name}" exported successfully as standalone HTML.`, 'success');
            } catch (error) {
                console.error(error);
                setStatus(error.message || 'Could not export as standalone HTML.', 'error');
            }
        }

        function exportAnimationAsGlb() {
            try {
                const asset = createAnimationAsset();
                const name = ensureAnimationName();
                const keyframes = asset.keyframes;
                const times = keyframes.map(kf => kf.time);
                const duration = times.length > 0 ? times[times.length - 1] : 0;

                const characterCount = getCaptureCharacterCount();
                const glbMode = isGlbModelMode();
                
                const activeModels = [];
                for (let i = 0; i < characterCount; i++) {
                    const model = glbMode ? glbModels[i] : previewRoots[i];
                    if (model) activeModels.push(model);
                }

                if (activeModels.length === 0) {
                    throw new Error('No active character model to export.');
                }

                // If multiple characters, wrap them in a Group. Otherwise just export the active model.
                let exportRoot;
                if (activeModels.length === 1) {
                    exportRoot = activeModels[0];
                } else {
                    exportRoot = new THREE.Group();
                    exportRoot.name = "SceneRoot";
                    activeModels.forEach(m => exportRoot.add(m));
                }

                // Build tracks
                const tracks = [];
                for (let characterIndex = 0; characterIndex < characterCount; characterIndex++) {
                    let restHipsPos = new THREE.Vector3(0, 0, 0);
                    let scale = 1;
                    if (glbMode) {
                        const boneMap = glbBonesMaps[characterIndex] || {};
                        const hipsBone = boneMap['Hips'];
                        if (hipsBone) {
                            restHipsPos.copy(hipsBone.position);
                        }
                        const model = glbModels[characterIndex];
                        if (model) {
                            scale = model.scale.x;
                        }
                    }

                    BASE_JOINT_ORDER.forEach(baseName => {
                        const jointName = getJointName(baseName, characterIndex);
                        let targetNodeName;
                        if (glbMode) {
                            targetNodeName = MIXAMO_BONE_MAP[baseName];
                        } else {
                            targetNodeName = `${baseName}_${characterIndex}`;
                        }
                        if (!targetNodeName) return;

                        const positionValues = [];
                        const quaternionValues = [];
                        let hasPosition = false;

                        keyframes.forEach(kf => {
                            const pose = kf.pose;
                            const jointPose = pose[jointName];
                            if (jointPose) {
                                if (baseName === 'Hips') {
                                    if (glbMode) {
                                        const defaultHips = defaultPoseState?.[jointName];
                                        const defaultHipsPos = defaultHips?.position ? new THREE.Vector3().fromArray(defaultHips.position) : getCharacterBasePosition(characterIndex);
                                        const currentHipsPos = new THREE.Vector3().fromArray(jointPose.position);
                                        const deltaPos = currentHipsPos.clone().sub(defaultHipsPos);
                                        
                                        const boneLocalPos = restHipsPos.clone().add(deltaPos.divideScalar(scale));
                                        positionValues.push(boneLocalPos.x, boneLocalPos.y, boneLocalPos.z);
                                    } else {
                                        positionValues.push(jointPose.position[0], jointPose.position[1], jointPose.position[2]);
                                    }
                                    hasPosition = true;
                                }

                                quaternionValues.push(
                                    jointPose.quaternion[0],
                                    jointPose.quaternion[1],
                                    jointPose.quaternion[2],
                                    jointPose.quaternion[3]
                                );
                            } else {
                                if (baseName === 'Hips') {
                                    if (glbMode) {
                                        positionValues.push(restHipsPos.x, restHipsPos.y, restHipsPos.z);
                                    } else {
                                        const basePos = getCharacterBasePosition(characterIndex);
                                        positionValues.push(basePos.x, basePos.y, basePos.z);
                                    }
                                    hasPosition = true;
                                }
                                quaternionValues.push(0, 0, 0, 1);
                            }
                        });

                        if (hasPosition) {
                            const posTrack = new THREE.Vector3KeyframeTrack(
                                `${targetNodeName}.position`,
                                times,
                                positionValues
                            );
                            tracks.push(posTrack);
                        }

                        const rotTrack = new THREE.QuaternionKeyframeTrack(
                            `${targetNodeName}.quaternion`,
                            times,
                            quaternionValues
                        );
                        tracks.push(rotTrack);
                    });
                }

                const clip = new THREE.AnimationClip(name, duration, tracks);
                const exporter = new GLTFExporter();
                
                setStatus('Exporting animation as GLB...', 'info');

                exporter.parse(
                    exportRoot,
                    (gltfBuffer) => {
                        const blob = new Blob([gltfBuffer], { type: 'application/octet-stream' });
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        const safeName = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'motion-rip';

                        link.href = url;
                        link.download = `${safeName}.glb`;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                        URL.revokeObjectURL(url);

                        if (characterCount > 1) {
                            activeModels.forEach(m => scene.add(m));
                        }

                        setStatus(`Animation "${name}" exported successfully as GLB.`, 'success');
                    },
                    (error) => {
                        console.error('GLTF Export error:', error);
                        if (characterCount > 1) {
                            activeModels.forEach(m => scene.add(m));
                        }
                        setStatus('Error occurred during GLB export.', 'error');
                    },
                    { 
                        binary: true,
                        animations: [clip]
                    }
                );

                if (characterCount > 1) {
                    activeModels.forEach(m => scene.add(m));
                }

            } catch (error) {
                console.error(error);
                setStatus(error.message || 'Could not export as GLB.', 'error');
            }
        }

        function generateStandaloneHtml(asset) {
            return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Standalone Player - ${asset.name}</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #050816;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #f1f5f9;
            user-select: none;
            -webkit-user-select: none;
        }
        #viewport {
            width: 100%;
            height: 100%;
            display: block;
        }
        
        .hud-panel {
            position: absolute;
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            padding: 16px 20px;
            z-index: 10;
        }
        
        .hud-header {
            top: 24px;
            left: 24px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            pointer-events: none;
        }
        .hud-header h1 {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: #ffffff;
        }
        .hud-header p {
            margin: 0;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #06b6d4;
        }

        .hud-controls {
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 600px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-sizing: border-box;
        }

        .controls-row {
            display: flex;
            align-items: center;
            gap: 16px;
            width: 100%;
        }

        .timeline-container {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .time-display {
            font-size: 12px;
            font-variant-numeric: tabular-nums;
            color: #94a3b8;
            font-weight: 500;
            min-width: 70px;
            text-align: right;
        }

        input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: rgba(255, 255, 255, 0.15);
            outline: none;
            cursor: pointer;
            margin: 0;
            transition: background 0.2s;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transition: transform 0.1s;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.3);
        }

        .play-btn {
            background: #ffffff;
            color: #0f172a;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s, background 0.2s;
            flex-shrink: 0;
        }
        .play-btn:hover {
            transform: scale(1.08);
            background: #f1f5f9;
        }
        .play-btn:active {
            transform: scale(0.96);
        }
        .play-btn svg {
            width: 14px;
            height: 14px;
            fill: currentColor;
        }

        .secondary-btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #f1f5f9;
            height: 32px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 500;
            padding: 0 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .secondary-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }
        .secondary-btn:active {
            transform: translateY(0);
        }

        select {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #f1f5f9;
            border-radius: 8px;
            height: 32px;
            font-size: 12px;
            font-weight: 500;
            padding: 0 10px;
            outline: none;
            cursor: pointer;
            transition: all 0.2s;
        }
        select:hover {
            background: rgba(15, 23, 42, 0.95);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .flex-spacer {
            flex-grow: 1;
        }
    </style>
</head>
<body>
    <div class="hud-panel hud-header">
        <p>Offline Viewer</p>
        <h1>${asset.name}</h1>
    </div>

    <div class="hud-panel hud-controls">
        <div class="controls-row">
            <button class="play-btn" id="play-btn" title="Play/Pause">
                <svg id="play-icon" viewBox="0 0 24 24" style="display:none;"><path d="M8 5v14l11-7z"/></svg>
                <svg id="pause-icon" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
            </button>
            
            <div class="timeline-container">
                <input type="range" id="scrubber" min="0" max="100" value="0" step="0.1">
                <span class="time-display" id="time-display">0.0s / 0.0s</span>
            </div>
        </div>
        <div class="controls-row" style="margin-top: 4px;">
            <select id="speed-select" title="Playback Speed">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1.0" selected>1.0x (Normal)</option>
                <option value="1.5">1.5x</option>
                <option value="2.0">2.0x</option>
            </select>

            <div class="flex-spacer"></div>

            <button class="secondary-btn" id="reset-cam-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><polyline points="3 3 3 8 8 8"></polyline><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"></path><polyline points="16 16 21 16 21 21"></polyline></svg>
                Reset Camera
            </button>
        </div>
    </div>

    <canvas id="viewport"></canvas>

    <script type="importmap">
        {
            "imports": {
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }
        }
    <\/script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        const ANIMATION_DATA = ${JSON.stringify(asset)};
        const keyframes = ANIMATION_DATA.keyframes || [];
        const totalDuration = keyframes.length > 0 ? keyframes[keyframes.length - 1].time : 0;
        const BASE_JOINT_ORDER = [
            'Hips', 'Spine', 'Head', 'Left_Upper_Arm', 'Left_Lower_Arm',
            'Right_Upper_Arm', 'Right_Lower_Arm', 'Left_Upper_Leg',
            'Left_Lower_Leg', 'Right_Upper_Leg', 'Right_Lower_Leg'
        ];

        let scene, camera, renderer, orbitControls;
        let previewRoots = [];
        let currentTime = 0;
        let isPlaying = true;
        let playbackSpeed = 1.0;
        let clock = new THREE.Clock();

        const playBtn = document.getElementById('play-btn');
        const playIcon = document.getElementById('play-icon');
        const pauseIcon = document.getElementById('pause-icon');
        const scrubber = document.getElementById('scrubber');
        const timeDisplay = document.getElementById('time-display');
        const speedSelect = document.getElementById('speed-select');
        const resetCamBtn = document.getElementById('reset-cam-btn');

        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050816);
            scene.fog = new THREE.Fog(0x050816, 10, 26);

            camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 100);
            camera.position.set(0, 4.8, 10.5);

            renderer = new THREE.WebGLRenderer({
                canvas: document.getElementById('viewport'),
                antialias: true,
                alpha: false
            });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
            scene.add(ambientLight);

            const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
            keyLight.position.set(6, 10, 5);
            keyLight.castShadow = true;
            keyLight.shadow.mapSize.width = 2048;
            keyLight.shadow.mapSize.height = 2048;
            scene.add(keyLight);

            const rimLight = new THREE.DirectionalLight(0x67e8f9, 0.7);
            rimLight.position.set(-6, 6, -4);
            scene.add(rimLight);

            const fillLight = new THREE.PointLight(0x34d399, 0.8, 20);
            fillLight.position.set(0, 5, 4);
            scene.add(fillLight);

            const grid = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
            scene.add(grid);

            const floor = new THREE.Mesh(
                new THREE.PlaneGeometry(50, 50),
                new THREE.MeshStandardMaterial({ color: 0x07111f, roughness: 0.9, metalness: 0.05 })
            );
            floor.rotation.x = -Math.PI / 2;
            floor.receiveShadow = true;
            scene.add(floor);

            orbitControls = new OrbitControls(camera, renderer.domElement);
            orbitControls.enableDamping = true;
            orbitControls.dampingFactor = 0.08;
            orbitControls.target.set(0, 2.6, 0);
            orbitControls.maxPolarAngle = Math.PI / 2 - 0.08;

            const characterCount = ANIMATION_DATA.scene?.characterCount || 1;
            const characterColors = ANIMATION_DATA.scene?.characterColors || ['#5eead4'];
            for (let charIndex = 0; charIndex < characterCount; charIndex++) {
                const colorHex = characterColors[charIndex] || '#5eead4';
                previewRoots.push(createPreviewCharacter(charIndex, characterCount, colorHex));
            }

            playBtn.addEventListener('click', togglePlay);
            scrubber.addEventListener('input', handleScrub);
            speedSelect.addEventListener('change', (e) => {
                playbackSpeed = parseFloat(e.target.value);
            });
            resetCamBtn.addEventListener('click', resetCamera);

            window.addEventListener('resize', handleResize);

            clock.start();
            renderer.setAnimationLoop(update);
        }

        function createPreviewCharacter(characterIndex, characterCount, colorHex) {
            const color = new THREE.Color(colorHex);
            const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.08 });

            function caps(r, len, pivotY, name) {
                const group = new THREE.Group();
                group.name = name + '_' + characterIndex;
                const geo = new THREE.CapsuleGeometry(r, len, 4, 10);
                geo.translate(0, pivotY, 0);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }

            function cyl(rTop, rBot, h, pivotY, name) {
                const group = new THREE.Group();
                group.name = name + '_' + characterIndex;
                const geo = new THREE.CylinderGeometry(rTop, rBot, h, 10);
                geo.translate(0, pivotY, 0);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.castShadow = mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }

            const UARM_R = 0.13,  UARM_L = 0.52, UARM_H = UARM_L + 2 * UARM_R;
            const LARM_R = 0.105, LARM_L = 0.48, LARM_H = LARM_L + 2 * LARM_R;
            const HAND_R = 0.085, HAND_L = 0.06, HAND_H = HAND_L + 2 * HAND_R;
            const ULEG_R = 0.175, ULEG_L = 0.72, ULEG_H = ULEG_L + 2 * ULEG_R;
            const LLEG_R = 0.125, LLEG_L = 0.68, LLEG_H = LLEG_L + 2 * LLEG_R;
            const CHEST_H = 0.85;
            const NECK_R = 0.10,  NECK_L = 0.13, NECK_H = NECK_L + 2 * NECK_R;
            const HEAD_R = 0.25,  HEAD_L = 0.18, HEAD_H = HEAD_L + 2 * HEAD_R;
            const HIP_R  = 0.195, HIP_L  = 0.10, HIP_H  = HIP_L  + 2 * HIP_R;

            const root = caps(HIP_R, HIP_L, 0, 'Hips');
            
            const xOffset = characterCount > 1
                ? (characterIndex - (characterCount - 1) / 2) * 3
                : 0;
            root.position.set(xOffset, 2.6, 0);

            const spine = cyl(0.32, 0.20, CHEST_H, CHEST_H / 2, 'Spine');
            spine.position.set(0, HIP_H * 0.35, 0);
            root.add(spine);

            const neck = caps(NECK_R, NECK_L, NECK_H / 2, 'Neck');
            neck.position.set(0, CHEST_H, 0);
            spine.add(neck);

            const head = caps(HEAD_R, HEAD_L, HEAD_H / 2, 'Head');
            head.position.set(0, NECK_H, 0);
            neck.add(head);

            const leftUpperArm = caps(UARM_R, UARM_L, -UARM_H / 2, 'Left_Upper_Arm');
            leftUpperArm.position.set(0.33, CHEST_H - 0.08, 0);
            leftUpperArm.rotation.z = Math.PI / 2;
            spine.add(leftUpperArm);

            const leftLowerArm = caps(LARM_R, LARM_L, -LARM_H / 2, 'Left_Lower_Arm');
            leftLowerArm.position.set(0, -UARM_H, 0);
            leftUpperArm.add(leftLowerArm);

            const leftHand = caps(HAND_R, HAND_L, -HAND_H / 2, 'Left_Hand');
            leftHand.position.set(0, -LARM_H, 0);
            leftLowerArm.add(leftHand);

            const rightUpperArm = caps(UARM_R, UARM_L, -UARM_H / 2, 'Right_Upper_Arm');
            rightUpperArm.position.set(-0.33, CHEST_H - 0.08, 0);
            rightUpperArm.rotation.z = -Math.PI / 2;
            spine.add(rightUpperArm);

            const rightLowerArm = caps(LARM_R, LARM_L, -LARM_H / 2, 'Right_Lower_Arm');
            rightLowerArm.position.set(0, -UARM_H, 0);
            rightUpperArm.add(rightLowerArm);

            const rightHand = caps(HAND_R, HAND_L, -HAND_H / 2, 'Right_Hand');
            rightHand.position.set(0, -LARM_H, 0);
            rightLowerArm.add(rightHand);

            const leftUpperLeg = caps(ULEG_R, ULEG_L, -ULEG_H / 2, 'Left_Upper_Leg');
            leftUpperLeg.position.set(0.185, -HIP_H * 0.32, 0);
            root.add(leftUpperLeg);

            const leftLowerLeg = caps(LLEG_R, LLEG_L, -LLEG_H / 2, 'Left_Lower_Leg');
            leftLowerLeg.position.set(0, -ULEG_H, 0);
            leftUpperLeg.add(leftLowerLeg);

            const leftFoot = caps(0.09, 0.28, -0.09, 'Left_Foot');
            leftFoot.position.set(0.02, -LLEG_H, 0.05);
            leftFoot.rotation.x = -Math.PI / 2;
            leftLowerLeg.add(leftFoot);

            const rightUpperLeg = caps(ULEG_R, ULEG_L, -ULEG_H / 2, 'Right_Upper_Leg');
            rightUpperLeg.position.set(-0.185, -HIP_H * 0.32, 0);
            root.add(rightUpperLeg);

            const rightLowerLeg = caps(LLEG_R, LLEG_L, -LLEG_H / 2, 'Right_Lower_Leg');
            rightLowerLeg.position.set(0, -ULEG_H, 0);
            rightUpperLeg.add(rightLowerLeg);

            const rightFoot = caps(0.09, 0.28, -0.09, 'Right_Foot');
            rightFoot.position.set(-0.02, -LLEG_H, 0.05);
            rightFoot.rotation.x = -Math.PI / 2;
            rightLowerLeg.add(rightFoot);

            scene.add(root);
            return root;
        }

        function togglePlay() {
            isPlaying = !isPlaying;
            if (isPlaying) {
                playIcon.style.display = 'none';
                pauseIcon.style.display = 'block';
            } else {
                playIcon.style.display = 'block';
                pauseIcon.style.display = 'none';
            }
        }

        function handleScrub(e) {
            const val = parseFloat(e.target.value);
            currentTime = (val / 100) * totalDuration;
            updateTimeText();
            applyKeyframePose(currentTime);
        }

        function resetCamera() {
            camera.position.set(0, 4.8, 10.5);
            orbitControls.target.set(0, 2.6, 0);
            orbitControls.update();
        }

        function handleResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function getInterpolatedPose(time) {
            if (keyframes.length === 0) return null;
            if (keyframes.length === 1) return keyframes[0].pose;

            if (time <= keyframes[0].time) return keyframes[0].pose;
            if (time >= keyframes[keyframes.length - 1].time) return keyframes[keyframes.length - 1].pose;

            let i1 = 0;
            let i2 = 1;
            for (let i = 0; i < keyframes.length - 1; i++) {
                if (time >= keyframes[i].time && time <= keyframes[i + 1].time) {
                    i1 = i;
                    i2 = i + 1;
                    break;
                }
            }

            const k1 = keyframes[i1];
            const k2 = keyframes[i2];

            const span = k2.time - k1.time;
            const alpha = span > 0 ? (time - k1.time) / span : 0;

            const pose = {};
            Object.keys(k1.pose).forEach(jointName => {
                const j1 = k1.pose[jointName];
                const j2 = k2.pose[jointName] || j1;

                const p1 = new THREE.Vector3().fromArray(j1.position);
                const p2 = new THREE.Vector3().fromArray(j2.position);
                const pos = p1.lerp(p2, alpha);

                const q1 = new THREE.Quaternion().fromArray(j1.quaternion);
                const q2 = new THREE.Quaternion().fromArray(j2.quaternion);
                const quat = q1.slerp(q2, alpha);

                pose[jointName] = { position: pos, quaternion: quat };
            });

            return pose;
        }

        function applyKeyframePose(time) {
            const poseState = getInterpolatedPose(time);
            if (!poseState) return;

            previewRoots.forEach(root => {
                root.traverse(object => {
                    if (!object.isGroup || !object.name || !poseState[object.name]) return;
                    object.position.copy(poseState[object.name].position);
                    object.quaternion.copy(poseState[object.name].quaternion);
                });
            });
        }

        function updateTimeText() {
            timeDisplay.textContent = currentTime.toFixed(1) + 's / ' + totalDuration.toFixed(1) + 's';
        }

        function update() {
            const delta = clock.getDelta();
            if (isPlaying && totalDuration > 0) {
                currentTime += delta * playbackSpeed;
                if (currentTime > totalDuration) {
                    currentTime = 0;
                }
                scrubber.value = (currentTime / totalDuration) * 100;
                updateTimeText();
                applyKeyframePose(currentTime);
            }
            orbitControls.update();
            renderer.render(scene, camera);
        }

        init();
    </script>
</body>
</html>`;
        }

        function serializePose(pose) {
            const serialized = {};

            getJointOrder(getCaptureCharacterCount()).forEach(name => {
                const transform = pose[name];
                if (!transform?.position || !transform?.quaternion) return;
                serialized[name] = {
                    position: [transform.position.x, transform.position.y, transform.position.z],
                    quaternion: [
                        transform.quaternion.x,
                        transform.quaternion.y,
                        transform.quaternion.z,
                        transform.quaternion.w
                    ]
                };
            });

            return serialized;
        }

        function setGeneratedAnimationName() {
            if (!ui.animationName || ui.animationName.value.trim()) return;

            const now = new Date();
            const datePart = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
            const timePart = `${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`;
            ui.animationName.value = `rip-${datePart}-${timePart}`;
        }

        function ensureAnimationName() {
            if (!ui.animationName) {
                const now = new Date();
                const datePart = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
                const timePart = `${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`;
                return `rip-${datePart}-${timePart}`;
            }
            const trimmed = ui.animationName.value.trim();
            if (trimmed) return trimmed;
            setGeneratedAnimationName();
            return ui.animationName.value.trim();
        }

        function updateSmoothingLabel() {
            ui.smoothingValue.textContent = (Number.parseFloat(ui.smoothing.value) || 0).toFixed(2);
        }

        function handleMultiCharacterChanged() {
            handleCaptureModeChanged({ reconfigureLandmarker: true });
        }

        function handlePoseBackendChanged() {
            isProcessingFrame = false;
            latestPosePackets = [];
            lastProcessedVideoTime = -1;
            if (isManualAssistBackend()) {
                ui.upperBodyOnly.checked = true;
                ui.focusCrop.checked = false;
                ensureManualHandles();
            }
            clearOverlay();
            handleCaptureModeChanged({ reconfigureLandmarker: true });
        }

        async function handleCaptureModeChanged(options = {}) {
            if (isRecording) {
                stopRecording();
            }

            recordedFrames = [];
            lastSampledAt = -Infinity;
            latestPosePackets = [];
            rootBaselines = [];
            lastProcessedVideoTime = -1;
            if (isManualAssistBackend()) {
                ui.upperBodyOnly.checked = true;
                ensureManualHandles();
            }

            glbModels.forEach(model => { if (model) scene.remove(model); });
            glbModels = [];
            glbBonesMaps = [];
            glbRestQuaternions = [];

            createPreviewCharacters();
            defaultPoseState = captureCharacterPose();
            currentPoseState = clonePoseState(defaultPoseState);

            if (isGlbModelMode()) {
                await ensureGlbCharacters();
            }
            applyPoseState(currentPoseState);
            updateCaptureModeUi();
            updateStats();
            clearOverlay();

            if (options.reconfigureLandmarker && poseLandmarker) {
                poseLandmarker.close?.();
                poseLandmarker = null;
                poseLandmarkerPoseCount = 0;
                poseLandmarkerMode = '';
            }

            try {
                await ensurePoseBackend();
                const modeLabel = getCaptureCharacterCount() > 1 ? 'Multi-character' : 'Single-character';
                const bodyLabel = isUpperBodyOnly() ? ' Upper-body source mode will keep hips and legs planted.' : '';
                setStatus(`${modeLabel} capture is ready with ${getPoseBackendLabel()}. Recordings will save ${getCaptureCharacterCount()} character${getCaptureCharacterCount() > 1 ? 's' : ''} in the shared animation format.${bodyLabel}`, 'success');
            } catch (error) {
                console.error(error);
                setStatus('Pose backend could not switch capture modes. Check your connection and try again.', 'error');
            }
        }

        function updateCaptureModeUi() {
            const isMultiCharacter = getCaptureCharacterCount() > 1;
            const manualMode = isManualAssistBackend();
            if (manualMode && !ui.upperBodyOnly.checked) {
                ui.upperBodyOnly.checked = true;
            }
            const upperBodyOnly = isUpperBodyOnly();
            const focusCrop = isFocusCropEnabled();

            syncAssistCharacterOptions();
            ui.manualAssistPanel.classList.toggle('hidden', !manualMode);
            ui.secondCharacterColorField.classList.toggle('hidden', !isMultiCharacter);
            ui.secondCharacterColor.disabled = !isMultiCharacter;
            ui.upperBodyOnly.disabled = manualMode;
            ui.rootMotion.disabled = upperBodyOnly;
            ui.rootMotionField.classList.toggle('opacity-50', upperBodyOnly);
            ui.poseOverlay.classList.toggle('crop-active', focusCrop);
            ui.poseOverlay.classList.toggle('manual-active', manualMode);
            ui.resetCropBtn.disabled = !focusCrop;
            ui.resetCropBtn.classList.toggle('opacity-50', !focusCrop);

            if (upperBodyOnly) {
                ui.rootMotion.checked = false;
            }
        }

        function syncAssistCharacterOptions() {
            const characterCount = getCaptureCharacterCount();
            Array.from(ui.assistCharacter.options).forEach(option => {
                option.disabled = Number.parseInt(option.value, 10) >= characterCount;
            });
            if (Number.parseInt(ui.assistCharacter.value, 10) >= characterCount) {
                ui.assistCharacter.value = '0';
            }
        }

        function handleFocusCropChanged() {
            cropDragState = null;
            updateCaptureModeUi();
            clearOverlay();

            if (isFocusCropEnabled()) {
                setStatus('Focus crop enabled. Drag a rectangle over the movie frame in the Pose Feed, then wait for the skeleton to relock.', 'info');
            } else {
                sourceCropRect = null;
                setStatus('Focus crop disabled. Tracking will use the full shared video again.', 'info');
            }
        }

        function resetSourceCrop() {
            sourceCropRect = null;
            cropDragState = null;
            clearOverlay();
            setStatus('Crop reset. Drag over the Pose Feed to set a tighter tracking area.', 'info');
        }

        function applyManualPoseFromHandles(options = {}) {
            if (!isManualAssistBackend()) return;
            if (!options.skipTimelineSync && !hasManualTrackTemplates()) {
                syncManualHandlesToTimeline();
            }

            const posePackets = getManualPosePackets({ skipTimelineSync: options.skipTimelineSync });
            latestPosePackets = posePackets;
            if (ui.trackedState) ui.trackedState.textContent = getTrackedStateLabel(posePackets.length);
            if (ui.confidenceValue) ui.confidenceValue.textContent = '100%';
            drawOverlay(posePackets);

            const targetPose = clonePoseState(currentPoseState || defaultPoseState);
            let hasMappedPose = false;
            posePackets.forEach(packet => {
                hasMappedPose = applyLandmarksToPose(targetPose, packet.landmarks, packet.worldLandmarks, packet.characterIndex) || hasMappedPose;
            });

            if (!hasMappedPose) return;

            currentPoseState = targetPose;
            applyPoseState(currentPoseState);
            if (isRecording) {
                samplePoseIfRecording(performance.now());
            }
        }

        function updateRecordingUi() {
            if (ui.recordBtn) {
                ui.recordBtn.textContent = isRecording ? 'Stop Record' : 'Start Record';
                ui.recordBtn.className = isRecording
                    ? 'rounded-2xl bg-rose-500 px-4 py-3 text-sm font-semibold text-rose-950 transition hover:bg-rose-400'
                    : 'rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400';
            }
            ui.recordingBadge.textContent = isRecording ? 'Recording' : 'Idle';
            ui.recordingBadge.className = isRecording
                ? 'rounded-full border border-rose-400/25 bg-rose-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-rose-200'
                : 'rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-200';
        }

        function updateStats() {
            if (ui.frameCount) ui.frameCount.textContent = String(recordedFrames.length);
            const lastTime = recordedFrames.length > 0 ? recordedFrames[recordedFrames.length - 1].time : 0;
            if (ui.durationValue) ui.durationValue.textContent = `${lastTime.toFixed(1)}s`;

            const disabled = recordedFrames.length === 0;
            if (ui.saveLibraryBtn) ui.saveLibraryBtn.disabled = disabled;
            if (ui.exportHtmlBtn) ui.exportHtmlBtn.disabled = disabled;
        }

        function setStatus(message, tone = 'info') {
            if (!ui.statusText) return;
            const classes = {
                info: 'text-sm leading-6 text-slate-300',
                success: 'text-sm leading-6 text-emerald-200',
                error: 'text-sm leading-6 text-rose-200'
            };

            ui.statusText.className = classes[tone] || classes.info;
            ui.statusText.textContent = message;
        }

        function renderScene() {
            orbitControls.update();
            renderer.render(scene, camera);
        }

        function handleResize() {
            if (!ui.previewCanvas || !renderer) return;
            const width = ui.previewCanvas.clientWidth || 1;
            const height = ui.previewCanvas.clientHeight || 1;
            
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height, false);
            resizeOverlayCanvas();
        }

        function resizeOverlayCanvas() {
            const width = ui.poseOverlay.clientWidth || 1;
            const height = ui.poseOverlay.clientHeight || 1;

            if (ui.poseOverlay.width !== width || ui.poseOverlay.height !== height) {
                ui.poseOverlay.width = width;
                ui.poseOverlay.height = height;
            }
        }

        function handleCropPointerDown(event) {
            if (isManualAssistBackend() && handleManualPointerDown(event)) return;
            if (!isFocusCropEnabled()) return;
            const point = getSourcePointFromOverlayEvent(event);
            if (!point) return;

            event.preventDefault();
            ui.poseOverlay.setPointerCapture?.(event.pointerId);
            cropDragState = { start: point, current: point };
            drawOverlay(latestPosePackets);
        }

        function handleCropPointerMove(event) {
            if (manualDragState) {
                handleManualPointerMove(event);
                return;
            }
            if (!cropDragState || !isFocusCropEnabled()) return;
            const point = getSourcePointFromOverlayEvent(event);
            if (!point) return;

            event.preventDefault();
            cropDragState.current = point;
            drawOverlay(latestPosePackets);
        }

        function handleCropPointerUp(event) {
            if (manualDragState) {
                handleManualPointerUp(event);
                return;
            }
            if (!cropDragState || !isFocusCropEnabled()) return;
            const point = getSourcePointFromOverlayEvent(event);
            event.preventDefault();
            ui.poseOverlay.releasePointerCapture?.(event.pointerId);

            if (point) {
                cropDragState.current = point;
            }

            const cropRect = normalizeCropRect(cropDragState.start, cropDragState.current);
            cropDragState = null;

            if (cropRect && cropRect.width >= MIN_CROP_SIZE && cropRect.height >= MIN_CROP_SIZE) {
                sourceCropRect = cropRect;
                lastProcessedVideoTime = -1;
                setStatus('Focus crop set. Tracking now ignores the browser UI outside that rectangle.', 'success');
            } else {
                setStatus('Crop was too small. Drag a larger rectangle around the movie frame.', 'error');
            }

            drawOverlay(latestPosePackets);
        }

        function cancelCropDrag(event) {
            if (manualDragState) {
                ui.poseOverlay.releasePointerCapture?.(event.pointerId);
                manualDragState = null;
                drawOverlay(latestPosePackets);
                return;
            }
            if (!cropDragState) return;
            ui.poseOverlay.releasePointerCapture?.(event.pointerId);
            cropDragState = null;
            drawOverlay(latestPosePackets);
        }

        function handleManualPointerDown(event) {
            const point = getSourcePointFromOverlayEvent(event);
            if (!point) return false;

            ensureManualHandles();
            const hit = findManualHandleAtPoint(point);
            if (!hit) return false;

            event.preventDefault();
            ui.poseOverlay.setPointerCapture?.(event.pointerId);
            manualDragState = hit;
            ui.assistCharacter.value = String(hit.characterIndex);
            drawOverlay(latestPosePackets);
            return true;
        }

        function handleManualPointerMove(event) {
            if (!manualDragState) return;
            const point = getSourcePointFromOverlayEvent(event);
            if (!point) return;

            event.preventDefault();
            manualHandles[manualDragState.characterIndex][manualDragState.handleId] = point;
            applyManualPoseFromHandles();
        }

        function handleManualPointerUp(event) {
            if (!manualDragState) return;
            handleManualPointerMove(event);
            const capturedTrack = captureManualTrackTemplate(manualDragState.characterIndex, manualDragState.handleId);
            addManualHandleKeyframe({ quiet: true });
            ui.poseOverlay.releasePointerCapture?.(event.pointerId);
            manualDragState = null;
            if (capturedTrack) {
                setStatus('Manual handle locked. As the source moves, the handle will track that local image patch; drag it again if it drifts.', 'success');
            }
            drawOverlay(latestPosePackets);
        }

        function findManualHandleAtPoint(point) {
            const rect = getContainedVideoRect(
                ui.poseOverlay.width || 1,
                ui.poseOverlay.height || 1,
                ui.sourceVideo.videoWidth || 1,
                ui.sourceVideo.videoHeight || 1
            );
            const activeCharacterIndex = getActiveManualCharacterIndex();
            const handles = manualHandles[activeCharacterIndex] || {};
            let bestHit = null;

            MANUAL_HANDLE_DEFS.forEach(def => {
                const handle = handles[def.id];
                if (!handle) return;
                const dx = (handle.x - point.x) * rect.width;
                const dy = (handle.y - point.y) * rect.height;
                const distance = Math.hypot(dx, dy);
                if (distance > 22 || (bestHit && distance >= bestHit.distance)) return;
                bestHit = {
                    characterIndex: activeCharacterIndex,
                    handleId: def.id,
                    distance
                };
            });

            return bestHit;
        }

        function getSourcePointFromOverlayEvent(event) {
            const bounds = ui.poseOverlay.getBoundingClientRect();
            const x = event.clientX - bounds.left;
            const y = event.clientY - bounds.top;
            const displayRect = getContainedVideoRect(
                ui.poseOverlay.clientWidth || 1,
                ui.poseOverlay.clientHeight || 1,
                ui.sourceVideo.videoWidth || 1,
                ui.sourceVideo.videoHeight || 1
            );

            if (displayRect.width <= 0 || displayRect.height <= 0) return null;

            return {
                x: THREE.MathUtils.clamp((x - displayRect.x) / displayRect.width, 0, 1),
                y: THREE.MathUtils.clamp((y - displayRect.y) / displayRect.height, 0, 1)
            };
        }

        function normalizeCropRect(start, end) {
            if (!start || !end) return null;
            const x = Math.min(start.x, end.x);
            const y = Math.min(start.y, end.y);
            const width = Math.abs(start.x - end.x);
            const height = Math.abs(start.y - end.y);
            return { x, y, width, height };
        }

        function drawOverlay(posePackets) {
            resizeOverlayCanvas();

            const context = ui.poseOverlay.getContext('2d');
            const width = ui.poseOverlay.width;
            const height = ui.poseOverlay.height;
            context.clearRect(0, 0, width, height);

            const rect = getContainedVideoRect(width, height, ui.sourceVideo.videoWidth || 1, ui.sourceVideo.videoHeight || 1);
            drawSourceCropOverlay(context, rect);

            context.lineWidth = 3;
            context.lineCap = 'round';
            context.shadowBlur = 16;

            posePackets.forEach(packet => {
                const color = new THREE.Color(getCharacterColorValue(packet.characterIndex));
                const stroke = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.95)`;
                const shadow = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.45)`;
                context.strokeStyle = stroke;
                context.shadowColor = shadow;

                CONNECTIONS.forEach(([startIndex, endIndex]) => {
                    const start = packet.landmarks[startIndex];
                    const end = packet.landmarks[endIndex];
                    if (!isReliableLandmark(start) || !isReliableLandmark(end)) return;

                    const startPoint = projectLandmark(start, rect);
                    const endPoint = projectLandmark(end, rect);

                    context.beginPath();
                    context.moveTo(startPoint.x, startPoint.y);
                    context.lineTo(endPoint.x, endPoint.y);
                    context.stroke();
                });

                context.shadowBlur = 0;
                packet.landmarks.forEach(landmark => {
                    if (!isReliableLandmark(landmark)) return;
                    const point = projectLandmark(landmark, rect);
                    context.beginPath();
                    context.fillStyle = stroke;
                    context.arc(point.x, point.y, 4, 0, Math.PI * 2);
                    context.fill();
                });

                context.shadowBlur = 16;
            });

            drawManualHandles(context, rect);
        }

        function clearOverlay() {
            resizeOverlayCanvas();
            const context = ui.poseOverlay.getContext('2d');
            context.clearRect(0, 0, ui.poseOverlay.width, ui.poseOverlay.height);

            const rect = getContainedVideoRect(
                ui.poseOverlay.width || 1,
                ui.poseOverlay.height || 1,
                ui.sourceVideo.videoWidth || 1,
                ui.sourceVideo.videoHeight || 1
            );
            drawSourceCropOverlay(context, rect);
            drawManualHandles(context, rect);
        }

        function drawManualHandles(context, displayRect) {
            if (!isManualAssistBackend()) return;

            ensureManualHandles();
            const activeCharacterIndex = getActiveManualCharacterIndex();
            const handles = manualHandles[activeCharacterIndex] || {};
            const color = new THREE.Color(getCharacterColorValue(activeCharacterIndex));
            const stroke = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.95)`;
            const fill = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.18)`;

            context.save();
            context.lineWidth = 2;
            context.strokeStyle = stroke;
            context.fillStyle = fill;
            context.shadowBlur = 10;
            context.shadowColor = stroke;

            [
                ['head', 'leftShoulder'],
                ['head', 'rightShoulder'],
                ['leftShoulder', 'rightShoulder'],
                ['leftShoulder', 'leftElbow'],
                ['leftElbow', 'leftWrist'],
                ['rightShoulder', 'rightElbow'],
                ['rightElbow', 'rightWrist']
            ].forEach(([startId, endId]) => {
                const start = handles[startId];
                const end = handles[endId];
                if (!start || !end) return;
                context.beginPath();
                context.moveTo(displayRect.x + start.x * displayRect.width, displayRect.y + start.y * displayRect.height);
                context.lineTo(displayRect.x + end.x * displayRect.width, displayRect.y + end.y * displayRect.height);
                context.stroke();
            });

            MANUAL_HANDLE_DEFS.forEach(def => {
                const point = handles[def.id];
                if (!point) return;
                const isLocked = Boolean(manualTrackTemplates?.[activeCharacterIndex]?.[def.id]);
                const x = displayRect.x + point.x * displayRect.width;
                const y = displayRect.y + point.y * displayRect.height;
                context.beginPath();
                context.fillStyle = isLocked ? stroke : fill;
                context.arc(x, y, 8, 0, Math.PI * 2);
                context.fill();
                context.stroke();
                if (isLocked) {
                    context.beginPath();
                    context.fillStyle = 'rgba(15, 23, 42, 0.92)';
                    context.arc(x, y, 3, 0, Math.PI * 2);
                    context.fill();
                }

                context.shadowBlur = 0;
                context.fillStyle = 'rgba(226, 232, 240, 0.96)';
                context.font = '11px system-ui, sans-serif';
                context.fillText(def.label, x + 11, y - 9);
                context.shadowBlur = 10;
                context.fillStyle = fill;
            });

            context.restore();
        }

        function drawSourceCropOverlay(context, displayRect) {
            const cropRect = cropDragState
                ? normalizeCropRect(cropDragState.start, cropDragState.current)
                : isFocusCropEnabled() ? sourceCropRect : null;
            if (!cropRect) return;

            const x = displayRect.x + cropRect.x * displayRect.width;
            const y = displayRect.y + cropRect.y * displayRect.height;
            const width = cropRect.width * displayRect.width;
            const height = cropRect.height * displayRect.height;

            context.save();
            context.lineWidth = 2;
            context.setLineDash([8, 6]);
            context.strokeStyle = 'rgba(34, 211, 238, 0.95)';
            context.fillStyle = 'rgba(34, 211, 238, 0.08)';
            context.fillRect(x, y, width, height);
            context.strokeRect(x, y, width, height);
            context.restore();
        }

        function projectLandmark(landmark, rect) {
            return {
                x: rect.x + landmark.x * rect.width,
                y: rect.y + landmark.y * rect.height
            };
        }

        function getContainedVideoRect(canvasWidth, canvasHeight, videoWidth, videoHeight) {
            const canvasAspect = canvasWidth / canvasHeight;
            const videoAspect = videoWidth / videoHeight;

            if (!Number.isFinite(videoAspect) || videoAspect <= 0) {
                return { x: 0, y: 0, width: canvasWidth, height: canvasHeight };
            }

            if (videoAspect > canvasAspect) {
                const width = canvasWidth;
                const height = width / videoAspect;
                return { x: 0, y: (canvasHeight - height) / 2, width, height };
            }

            const height = canvasHeight;
            const width = height * videoAspect;
            return { x: (canvasWidth - width) / 2, y: 0, width, height };
        }

        function hasReliablePair(landmarks, firstIndex, secondIndex) {
            return isReliableLandmark(landmarks[firstIndex]) && isReliableLandmark(landmarks[secondIndex]);
        }

        function isReliableLandmark(landmark) {
            const threshold = isUpperBodyOnly() ? UPPER_BODY_LANDMARK_VISIBILITY : DEFAULT_LANDMARK_VISIBILITY;
            return !!landmark && (landmark.visibility ?? 1) >= threshold;
        }

        function getPoseConfidence(landmarks) {
            const trackedIndices = isUpperBodyOnly()
                ? [
                    LM.NOSE,
                    LM.LEFT_SHOULDER,
                    LM.RIGHT_SHOULDER,
                    LM.LEFT_ELBOW,
                    LM.RIGHT_ELBOW,
                    LM.LEFT_WRIST,
                    LM.RIGHT_WRIST
                ]
                : [
                    LM.LEFT_SHOULDER,
                    LM.RIGHT_SHOULDER,
                    LM.LEFT_ELBOW,
                    LM.RIGHT_ELBOW,
                    LM.LEFT_HIP,
                    LM.RIGHT_HIP,
                    LM.LEFT_KNEE,
                    LM.RIGHT_KNEE
                ];

            const scores = trackedIndices
                .map(index => landmarks[index]?.visibility)
                .filter(score => Number.isFinite(score));

            if (scores.length === 0) return 0;
            return scores.reduce((sum, score) => sum + score, 0) / scores.length;
        }

        function toWorldVector(landmark) {
            if (!landmark) return null;
            return new THREE.Vector3(landmark.x, -landmark.y, -landmark.z);
        }

        function midpointVector(a, b) {
            if (!a || !b) return null;
            return a.clone().add(b).multiplyScalar(0.5);
        }

        function midpointLandmark(a, b) {
            return {
                x: ((a?.x ?? 0) + (b?.x ?? 0)) * 0.5,
                y: ((a?.y ?? 0) + (b?.y ?? 0)) * 0.5
            };
        }

        function directionBetween(start, end) {
            if (!start || !end) return null;
            const direction = end.clone().sub(start);
            if (direction.lengthSq() < 1e-8) return null;
            return direction.normalize();
        }

        function averageDirection(vectors) {
            const sum = new THREE.Vector3();
            let count = 0;

            vectors.forEach(vector => {
                if (!vector || vector.lengthSq() < 1e-8) return;
                sum.add(vector);
                count += 1;
            });

            if (count === 0 || sum.lengthSq() < 1e-8) {
                return null;
            }

            return sum.normalize();
        }

        function distance2D(a, b) {
            if (!a || !b) return 0;
            return Math.hypot((a.x ?? 0) - (b.x ?? 0), (a.y ?? 0) - (b.y ?? 0));
        }

        function roundTime(value) {
            return Math.round((value + Number.EPSILON) / KEYFRAME_TIME_STEP) * KEYFRAME_TIME_STEP;
        }


        window.__motionRipperCleanup = function cleanupMotionRipper() {
            try {
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = 0;
                }
                if (renderer) {
                    renderer.setAnimationLoop(null);
                    renderer.dispose();
                }
                orbitControls?.dispose?.();
                resizeObserver?.disconnect?.();
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                if (ui.sourceVideo) {
                    ui.sourceVideo.pause();
                    ui.sourceVideo.srcObject = null;
                    ui.sourceVideo.removeAttribute('src');
                    ui.sourceVideo.load?.();
                }
                window.removeEventListener('resize', handleResize);
                poseLandmarker?.close?.();
                moveNetDetector?.dispose?.();
            } catch (error) {
                console.warn('Motion Ripper cleanup failed', error);
            }
        };

