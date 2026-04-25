import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import './PoseEditor.css';

const MP_LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
];

const PREFERRED_CHILD = {
    mixamorig_Hips: 'mixamorig_Spine',
    mixamorig_Spine: 'mixamorig_Spine1',
    mixamorig_Spine1: 'mixamorig_Spine2',
    mixamorig_Spine2: 'mixamorig_Neck',
    mixamorig_Neck: 'mixamorig_Head',
    mixamorig_Head: 'mixamorig_HeadTop_End',
    mixamorig_LeftShoulder: 'mixamorig_LeftArm',
    mixamorig_LeftArm: 'mixamorig_LeftForeArm',
    mixamorig_LeftForeArm: 'mixamorig_LeftHand',
    mixamorig_RightShoulder: 'mixamorig_RightArm',
    mixamorig_RightArm: 'mixamorig_RightForeArm',
    mixamorig_RightForeArm: 'mixamorig_RightHand',
    mixamorig_LeftUpLeg: 'mixamorig_LeftLeg',
    mixamorig_LeftLeg: 'mixamorig_LeftFoot',
    mixamorig_LeftFoot: 'mixamorig_LeftToeBase',
    mixamorig_RightUpLeg: 'mixamorig_RightLeg',
    mixamorig_RightLeg: 'mixamorig_RightFoot',
    mixamorig_RightFoot: 'mixamorig_RightToeBase',
};

const JOINT_LIMITS = {
    mixamorig_Hips: { x: [-0.6, 0.6], y: [-0.8, 0.8], z: [-0.5, 0.5] },
    mixamorig_Spine: { x: [-0.5, 0.3], y: [-0.3, 0.3], z: [-0.3, 0.3] },
    mixamorig_Spine1: { x: [-0.3, 0.2], y: [-0.2, 0.2], z: [-0.2, 0.2] },
    mixamorig_Spine2: { x: [-0.3, 0.2], y: [-0.2, 0.2], z: [-0.2, 0.2] },
    mixamorig_Neck: { x: [-0.6, 0.6], y: [-0.8, 0.8], z: [-0.4, 0.4] },
    mixamorig_Head: { x: [-0.5, 0.5], y: [-0.7, 0.7], z: [-0.3, 0.3] },
    mixamorig_LeftArm: { x: [-1.8, 1.8], y: [-1.5, 1.5], z: [-1.0, 2.8] },
    mixamorig_RightArm: { x: [-1.8, 1.8], y: [-1.5, 1.5], z: [-2.8, 1.0] },
    mixamorig_LeftForeArm: { x: [-0.3, 0.3], y: [-2.6, 0.1], z: [-0.3, 0.3] },
    mixamorig_RightForeArm: { x: [-0.3, 0.3], y: [-0.1, 2.6], z: [-0.3, 0.3] },
    mixamorig_LeftUpLeg: { x: [-2.2, 0.5], y: [-0.8, 0.8], z: [2.0, 4.3] },
    mixamorig_RightUpLeg: { x: [-2.2, 0.5], y: [-0.8, 0.8], z: [2.0, 4.3] },
    mixamorig_LeftLeg: { x: [-2.6, 0.0], y: [-0.1, 0.1], z: [-0.1, 0.1] },
    mixamorig_RightLeg: { x: [-2.6, 0.0], y: [-0.1, 0.1], z: [-0.1, 0.1] },
    mixamorig_LeftFoot: { x: [-0.2, 1.5], y: [-0.5, 0.5], z: [-0.3, 0.3] },
    mixamorig_RightFoot: { x: [-0.2, 1.5], y: [-0.5, 0.5], z: [-0.3, 0.3] },
};

const POSE_PRESETS = [
    {
        id: 't-pose',
        name: 'T Pose',
        description: 'Straight neutral stance with arms extended.',
        rotations: {},
    },
    {
        id: 'a-pose',
        name: 'A Pose',
        description: 'Relaxed neutral stance with arms lowered.',
        rotations: {
            mixamorig_LeftArm: { z: 0.95 },
            mixamorig_RightArm: { z: -0.95 },
            mixamorig_LeftForeArm: { y: -0.18 },
            mixamorig_RightForeArm: { y: 0.18 },
        },
    },
    {
        id: 'sitting-pose',
        name: 'Sitting Pose',
        description: 'Seated lower body with relaxed arms.',
        rotations: {
            mixamorig_Hips: { x: -0.18 },
            mixamorig_Spine: { x: 0.12 },
            mixamorig_LeftArm: { z: 1.1, x: 0.2 },
            mixamorig_RightArm: { z: -1.1, x: 0.2 },
            mixamorig_LeftForeArm: { y: -1.3 },
            mixamorig_RightForeArm: { y: 1.3 },
            mixamorig_LeftHand: { z: 0.25 },
            mixamorig_RightHand: { z: -0.25 },
            mixamorig_LeftUpLeg: { x: -1.35 },
            mixamorig_RightUpLeg: { x: -1.35 },
            mixamorig_LeftLeg: { x: -1.45 },
            mixamorig_RightLeg: { x: -1.45 },
            mixamorig_LeftFoot: { x: 0.35 },
            mixamorig_RightFoot: { x: 0.35 },
        },
    },
    {
        id: 'sitting-on-floor',
        name: 'Sitting On Floor',
        description: 'Sitting cross-legged on the floor.',
        rotations: {
            mixamorig_Hips: { x: 0.05 },
            mixamorig_Spine: { x: -0.1 },
            mixamorig_Spine2: { x: -0.08 },
            mixamorig_RightShoulder: { x: 1.6926, y: 0.0495, z: 1.7782 },
            mixamorig_RightArm: { x: -0.14, y: -0.8, z: -1.4 },
            mixamorig_RightForeArm: { x: -0.91, y: 0.6 },
            mixamorig_RightHandThumb1: { x: 0.5274, z: -0.4773 },
            mixamorig_RightHandThumb2: { x: -0.0046, y: -0.0024, z: 0.0186 },
            mixamorig_RightHandThumb3: { x: -0.0033, y: -0.0016, z: 0.0138 },
            mixamorig_RightHandThumb4: { y: -0.2358, z: -0.0982 },
            mixamorig_RightHandIndex1: { x: 0.18, z: -0.05 },
            mixamorig_RightHandIndex2: { x: 0.22 },
            mixamorig_RightHandIndex3: { x: 0.15 },
            mixamorig_RightHandIndex4: { y: 0.002, z: 0.0002 },
            mixamorig_RightHandMiddle1: { x: 0.18, z: -0.01 },
            mixamorig_RightHandMiddle2: { x: 0.22 },
            mixamorig_RightHandMiddle3: { x: 0.15 },
            mixamorig_RightHandMiddle4: { y: 0.0019, z: 0.0003 },
            mixamorig_RightHandRing1: { x: 0.16 },
            mixamorig_RightHandRing2: { x: 0.2 },
            mixamorig_RightHandRing3: { x: 0.14 },
            mixamorig_RightHandRing4: { y: 0.0003, z: -0.0016 },
            mixamorig_RightHandPinky1: { x: 0.15, z: 0.01 },
            mixamorig_RightHandPinky2: { x: 0.18 },
            mixamorig_RightHandPinky3: { x: 0.12 },
            mixamorig_RightHandPinky4: { y: 0.0031, z: -0.0001 },
            mixamorig_LeftShoulder: { x: 1.6926, y: -0.0495, z: -1.7782 },
            mixamorig_LeftArm: { x: -0.17, y: 1.08, z: 1.4 },
            mixamorig_LeftForeArm: { x: -1.18, y: -0.22 },
            mixamorig_LeftHandThumb1: { x: 0.5269, z: 0.4754 },
            mixamorig_LeftHandThumb2: { x: -0.0041, y: 0.0021, z: -0.0164 },
            mixamorig_LeftHandThumb3: { x: -0.0031, y: 0.0015, z: -0.0121 },
            mixamorig_LeftHandThumb4: { y: 0.2462, z: 0.084 },
            mixamorig_LeftHandIndex1: { x: 0.18, z: 0.05 },
            mixamorig_LeftHandIndex2: { x: 0.22 },
            mixamorig_LeftHandIndex3: { x: 0.15 },
            mixamorig_LeftHandIndex4: { y: -0.0008 },
            mixamorig_LeftHandMiddle1: { x: 0.18, z: 0.01 },
            mixamorig_LeftHandMiddle2: { x: 0.22 },
            mixamorig_LeftHandMiddle3: { x: 0.15 },
            mixamorig_LeftHandMiddle4: { y: -0.002, z: 0.0001 },
            mixamorig_LeftHandRing1: { x: 0.16 },
            mixamorig_LeftHandRing2: { x: 0.2 },
            mixamorig_LeftHandRing3: { x: 0.14 },
            mixamorig_LeftHandRing4: { y: 0.001 },
            mixamorig_LeftHandPinky1: { x: 0.15, z: 0.004 },
            mixamorig_LeftHandPinky2: { x: 0.18 },
            mixamorig_LeftHandPinky3: { x: 0.12 },
            mixamorig_LeftHandPinky4: { y: -0.0016, z: -0.0006 },
            mixamorig_RightUpLeg: { x: -1.65, y: 0.05, z: 3.1416 },
            mixamorig_RightLeg: { x: -1.5 },
            mixamorig_RightFoot: { x: 0.91 },
            mixamorig_RightToeBase: { x: 0.25 },
            mixamorig_RightToe_End: { y: 0.0232 },
            mixamorig_LeftUpLeg: { x: -1.65, y: -0.05, z: 3.1416 },
            mixamorig_LeftLeg: { x: -1.5 },
            mixamorig_LeftFoot: { x: 0.92 },
            mixamorig_LeftToeBase: { x: 0.25 },
            mixamorig_LeftToe_End: { y: -0.0237 },
        },
    },
];

function mp(landmark) {
    return new THREE.Vector3(landmark.x, -landmark.y, -landmark.z);
}

function mpMid(landmarks, a, b) {
    return mp(landmarks[a]).add(mp(landmarks[b])).multiplyScalar(0.5);
}

function indexedToNamed(landmarks) {
    const named = {};
    landmarks.forEach((landmark, index) => {
        if (index < MP_LANDMARK_NAMES.length) {
            named[MP_LANDMARK_NAMES[index]] = landmark;
        }
    });
    return named;
}

function clampEuler(euler, boneName) {
    const limits = JOINT_LIMITS[boneName];
    if (!limits) return euler;
    return new THREE.Euler(
        Math.max(limits.x[0], Math.min(limits.x[1], euler.x)),
        Math.max(limits.y[0], Math.min(limits.y[1], euler.y)),
        Math.max(limits.z[0], Math.min(limits.z[1], euler.z)),
        euler.order
    );
}

function getBoneWorldDirection(bone, boneMap) {
    let child = null;
    const preferred = PREFERRED_CHILD[bone.name];

    if (preferred && boneMap[preferred]) {
        child = boneMap[preferred];
    } else if (bone.name === 'mixamorig_Spine' && boneMap.mixamorig_Spine2) {
        child = boneMap.mixamorig_Spine2;
    } else {
        for (const candidate of bone.children) {
            if (candidate.isBone) {
                child = candidate;
                break;
            }
        }
    }

    if (!child) return null;

    const bonePos = new THREE.Vector3();
    const childPos = new THREE.Vector3();
    bone.getWorldPosition(bonePos);
    child.getWorldPosition(childPos);

    const direction = childPos.sub(bonePos);
    if (direction.lengthSq() < 0.000001) return null;
    return direction.normalize();
}

function computePoleConstrainedRotation(boneName, currentDir, targetDir, worldLandmarks) {
    const baseQ = new THREE.Quaternion().setFromUnitVectors(currentDir, targetDir);

    if (boneName === 'mixamorig_LeftForeArm' || boneName === 'mixamorig_RightForeArm') {
        const isLeft = boneName.includes('Left');
        const shoulder = mp(worldLandmarks[isLeft ? 'left_shoulder' : 'right_shoulder']);
        const elbow = mp(worldLandmarks[isLeft ? 'left_elbow' : 'right_elbow']);
        const wrist = mp(worldLandmarks[isLeft ? 'left_wrist' : 'right_wrist']);
        const upperArm = elbow.clone().sub(shoulder);
        const foreArm = wrist.clone().sub(elbow);
        const armPlaneNormal = new THREE.Vector3().crossVectors(upperArm, foreArm);

        if (armPlaneNormal.lengthSq() > 0.0001) {
            armPlaneNormal.normalize();
            const currentNormal = new THREE.Vector3(0, 0, isLeft ? -1 : 1).applyQuaternion(baseQ);
            const twistAngle = Math.atan2(
                currentNormal.dot(armPlaneNormal),
                currentNormal.clone().cross(armPlaneNormal).dot(targetDir)
            );

            if (Math.abs(twistAngle) > 0.1) {
                const twistQ = new THREE.Quaternion().setFromAxisAngle(targetDir, -twistAngle * 0.3);
                baseQ.premultiply(twistQ);
            }
        }
    }

    if (boneName === 'mixamorig_LeftLeg' || boneName === 'mixamorig_RightLeg') {
        const isLeft = boneName.includes('Left');
        const hip = mp(worldLandmarks[isLeft ? 'left_hip' : 'right_hip']);
        const knee = mp(worldLandmarks[isLeft ? 'left_knee' : 'right_knee']);
        const ankle = mp(worldLandmarks[isLeft ? 'left_ankle' : 'right_ankle']);
        const thigh = knee.clone().sub(hip);
        const shin = ankle.clone().sub(knee);
        const legPlaneNormal = new THREE.Vector3().crossVectors(thigh, shin);

        if (legPlaneNormal.lengthSq() > 0.0001) {
            legPlaneNormal.normalize();
            const forward = new THREE.Vector3(0, 0, 1);
            if (legPlaneNormal.dot(forward) < 0) {
                const flipQ = new THREE.Quaternion().setFromAxisAngle(targetDir, Math.PI);
                baseQ.premultiply(flipQ);
            }
        }
    }

    return baseQ;
}

export default function PoseEditor() {
    const navigate = useNavigate();
    const viewportRef = useRef(null);
    const imageInputRef = useRef(null);
    const previewImgRef = useRef(null);
    const overlayCanvasRef = useRef(null);
    const toastTimerRef = useRef(null);
    const previewUrlRef = useRef('');
    const poseLandmarkerRef = useRef(null);
    const initializedRef = useRef(false);
    const sceneStateRef = useRef({
        scene: null,
        camera: null,
        renderer: null,
        model: null,
        modelContainer: null,
        boneMap: {},
        baseBonePositions: {},
        currentPoseRotations: {},
        currentMixamoOutput: null,
        tPoseData: {},
        pointsGroup: null,
        allBoneIndicators: {},
        boneIndicator: null,
        boneIndicatorRing: null,
        skeletonHelper: null,
        raycaster: new THREE.Raycaster(),
        mouse: new THREE.Vector2(),
        isDragging: false,
        lastMouse: { x: 0, y: 0 },
        spherical: { theta: 0.3, phi: 1.1, radius: 2.8 },
        target: new THREE.Vector3(0, 1, 0),
        animationFrame: null,
        selectedBone: null,
    });

    const [step, setStep] = useState(1);
    const [badge, setBadge] = useState({ text: 'Waiting', tone: 'info' });
    const [showImagePanel, setShowImagePanel] = useState(true);
    const [showBonePanel, setShowBonePanel] = useState(true);
    const [showAllPoints, setShowAllPoints] = useState(false);
    const [activePanel, setActivePanel] = useState('view-bones');
    const [imageReady, setImageReady] = useState(false);
    const [boneCategories, setBoneCategories] = useState({});
    const [searchQuery, setSearchQuery] = useState('');
    const [imageSrc, setImageSrc] = useState('');
    const [imageStats, setImageStats] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [loadingText, setLoadingText] = useState('Initializing MediaPipe…');
    const [isPoseReady, setIsPoseReady] = useState(false);
    const [isModelReady, setIsModelReady] = useState(false);
    const [boneNames, setBoneNames] = useState([]);
    const [boneFilter, setBoneFilter] = useState('');
    const [selectedBone, setSelectedBone] = useState(null);
    const [boneRotation, setBoneRotation] = useState({ x: 0, y: 0, z: 0 });
    const [modelPosition, setModelPosition] = useState({ x: 0, y: 0, z: 0 });
    const [hasPoseData, setHasPoseData] = useState(false);
    const [exportModalOpen, setExportModalOpen] = useState(false);
    const [exportText, setExportText] = useState('');
    const [toast, setToast] = useState({ visible: false, icon: '', text: '' });
    const [leftPanelMode, setLeftPanelMode] = useState('image');
    const [jsonInputText, setJsonInputText] = useState('');
    const [jsonError, setJsonError] = useState('');

    const applyJsonPose = (poseData) => {
        if (!isModelReady) {
            showToast('⚠️', 'Model not loaded yet');
            return;
        }

        applyTPose();

        const { boneMap, currentPoseRotations } = sceneStateRef.current;
        let appliedCount = 0;

        Object.entries(poseData).forEach(([boneName, boneData]) => {
            const bone = boneMap[boneName];
            if (!bone || typeof boneData !== 'object') return;

            const rx = Number(boneData.x) || 0;
            const ry = Number(boneData.y) || 0;
            const rz = Number(boneData.z) || 0;

            bone.rotation.set(rx, ry, rz);
            currentPoseRotations[boneName] = { x: rx, y: ry, z: rz };
            appliedCount++;

            if (boneData.position) {
                bone.position.x = Number(boneData.position.x) || bone.position.x;
                bone.position.y = Number(boneData.position.y) || bone.position.y;
                bone.position.z = Number(boneData.position.z) || bone.position.z;
            }
        });

        sceneStateRef.current.currentMixamoOutput = poseData;
        setHasPoseData(appliedCount > 0);
        setBadge({ text: `${appliedCount} bones applied`, tone: 'success' });
        setStep(3);

        if (sceneStateRef.current.selectedBone) {
            refreshBoneEditorState(sceneStateRef.current.selectedBone);
        }

        showToast('✨', `Pose applied — ${appliedCount} bones`);
    };

    const handleApplyJsonText = () => {
        if (!jsonInputText.trim()) {
            setJsonError('Paste JSON pose data first');
            return;
        }
        try {
            const parsed = JSON.parse(jsonInputText);
            if (typeof parsed !== 'object' || Array.isArray(parsed)) {
                throw new Error('Expected an object with bone names as keys');
            }
            applyJsonPose(parsed);
            setJsonError('');
        } catch (err) {
            setJsonError(err.message);
        }
    };

    const handleJsonFile = (file) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            setJsonInputText(text);
            try {
                const parsed = JSON.parse(text);
                if (typeof parsed !== 'object' || Array.isArray(parsed)) {
                    throw new Error('Expected an object with bone names as keys');
                }
                applyJsonPose(parsed);
                setJsonError('');
            } catch (err) {
                setJsonError(err.message);
            }
        };
        reader.readAsText(file);
    };

    const handleResetAll = () => {
        if (!sceneStateRef.current.boneMap) return;
        applyTPose();
        sceneStateRef.current.currentMixamoOutput = { ...sceneStateRef.current.tPoseData };
        setHasPoseData(false);
        setBadge({ text: 'T-pose reset', tone: 'info' });
        if (selectedBone) {
            refreshBoneEditorState(selectedBone);
        }
        showToast('🔄', 'Pose reset to T-Pose');
    };

    const showToast = (icon, text) => {
        setToast({ visible: true, icon, text });
        if (toastTimerRef.current) {
            clearTimeout(toastTimerRef.current);
        }
        toastTimerRef.current = window.setTimeout(() => {
            setToast((current) => ({ ...current, visible: false }));
        }, 2500);
    };

    const setLoading = (text) => {
        setLoadingText(text);
        setIsLoading(true);
    };

    const clearLoading = () => {
        setIsLoading(false);
    };

    const applyBoneIndicator = (boneName) => {
        const bone = sceneStateRef.current.boneMap[boneName];
        const indicator = sceneStateRef.current.boneIndicator;
        if (!bone || !indicator) return;

        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);
        indicator.position.copy(worldPos);
        indicator.visible = true;

        if (sceneStateRef.current.boneIndicatorRing) {
            sceneStateRef.current.boneIndicatorRing.position.copy(worldPos);
            sceneStateRef.current.boneIndicatorRing.lookAt(sceneStateRef.current.camera.position);
            sceneStateRef.current.boneIndicatorRing.visible = true;
        }
    };

    const hideBoneIndicator = () => {
        if (sceneStateRef.current.boneIndicator) {
            sceneStateRef.current.boneIndicator.visible = false;
        }
        if (sceneStateRef.current.boneIndicatorRing) {
            sceneStateRef.current.boneIndicatorRing.visible = false;
        }
    };

    const syncSelectedBone = (boneName) => {
        const rotation = sceneStateRef.current.currentPoseRotations[boneName] || { x: 0, y: 0, z: 0 };
        sceneStateRef.current.selectedBone = boneName;
        setSelectedBone(boneName);
        setBoneRotation(rotation);
        applyBoneIndicator(boneName);
    };

    const refreshBoneEditorState = (boneName) => {
        const rotation = sceneStateRef.current.currentPoseRotations[boneName] || { x: 0, y: 0, z: 0 };
        setBoneRotation(rotation);
    };

    const selectBone = (boneName) => {
        syncSelectedBone(boneName);
        setShowBonePanel(true);
        setActivePanel('view-bones');
    };

    const setBoneRotationValue = (axis, value) => {
        if (!selectedBone) return;
        const currentRotation = sceneStateRef.current.currentPoseRotations[selectedBone] || { x: 0, y: 0, z: 0 };
        const nextRotation = { ...currentRotation, [axis]: value };
        const bone = sceneStateRef.current.boneMap[selectedBone];
        if (!bone) return;

        bone.rotation.set(nextRotation.x, nextRotation.y, nextRotation.z);
        sceneStateRef.current.currentPoseRotations[selectedBone] = nextRotation;
        setBoneRotation(nextRotation);

        if (sceneStateRef.current.currentMixamoOutput) {
            sceneStateRef.current.currentMixamoOutput[selectedBone] = {
                x: Number(nextRotation.x.toFixed(4)),
                y: Number(nextRotation.y.toFixed(4)),
                z: Number(nextRotation.z.toFixed(4)),
            };
        }
    };

    const setModelTranslation = (axis, value) => {
        const next = { ...modelPosition, [axis]: value };
        setModelPosition(next);
        if (sceneStateRef.current.modelContainer) {
            sceneStateRef.current.modelContainer.position.set(next.x, next.y, next.z);
        }
    };

    const buildCurrentPoseOutput = () => {
        const output = {};
        Object.entries(sceneStateRef.current.currentPoseRotations).forEach(([name, rotation]) => {
            output[name] = {
                x: Number(rotation.x.toFixed(4)),
                y: Number(rotation.y.toFixed(4)),
                z: Number(rotation.z.toFixed(4)),
            };
        });
        return output;
    };

    const applyTPose = () => {
        const { boneMap, tPoseData, baseBonePositions, currentPoseRotations, model } = sceneStateRef.current;
        Object.keys(boneMap).forEach((name) => {
            boneMap[name].rotation.set(0, 0, 0);
            currentPoseRotations[name] = { x: 0, y: 0, z: 0 };

            if (baseBonePositions[name]) {
                boneMap[name].position.copy(baseBonePositions[name]);
            }
        });

        Object.entries(tPoseData).forEach(([name, rotation]) => {
            if (boneMap[name]) {
                boneMap[name].rotation.set(rotation.x, rotation.y, rotation.z);
                currentPoseRotations[name] = { x: rotation.x, y: rotation.y, z: rotation.z };
            }
        });

        if (sceneStateRef.current.selectedBone) {
            refreshBoneEditorState(sceneStateRef.current.selectedBone);
        }

        if (model) {
            model.updateMatrixWorld(true);
        }
    };

    const applyPosePreset = (preset) => {
        if (!isModelReady) {
            showToast('⚠️', 'Model not loaded yet');
            return;
        }

        applyTPose();

        const { boneMap, currentPoseRotations, model } = sceneStateRef.current;
        Object.entries(preset.rotations).forEach(([boneName, override]) => {
            const bone = boneMap[boneName];
            if (!bone) return;

            const current = currentPoseRotations[boneName] || {
                x: bone.rotation.x,
                y: bone.rotation.y,
                z: bone.rotation.z,
            };
            const next = {
                x: override.x ?? current.x,
                y: override.y ?? current.y,
                z: override.z ?? current.z,
            };

            bone.rotation.set(next.x, next.y, next.z);
            currentPoseRotations[boneName] = next;
        });

        model?.updateMatrixWorld(true);
        sceneStateRef.current.currentMixamoOutput = buildCurrentPoseOutput();
        setHasPoseData(true);
        setBadge({ text: preset.name, tone: 'success' });
        setStep(3);

        if (sceneStateRef.current.selectedBone) {
            refreshBoneEditorState(sceneStateRef.current.selectedBone);
        }

        showToast('✨', `${preset.name} applied`);
    };

    const createBoneIndicators = () => {
        const { scene, boneMap, pointsGroup, boneIndicator, boneIndicatorRing } = sceneStateRef.current;

        if (pointsGroup) scene.remove(pointsGroup);
        if (boneIndicator) scene.remove(boneIndicator);
        if (boneIndicatorRing) scene.remove(boneIndicatorRing);

        const group = new THREE.Group();
        group.visible = showAllPoints;
        scene.add(group);
        sceneStateRef.current.pointsGroup = group;
        sceneStateRef.current.allBoneIndicators = {};

        const pointGeometry = new THREE.SphereGeometry(0.014, 12, 12);
        const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x30d158, transparent: true, opacity: 0.65, depthTest: false });

        Object.keys(boneMap).forEach((name) => {
            const point = new THREE.Mesh(pointGeometry, pointMaterial);
            point.renderOrder = 997;
            point.userData.boneName = name;
            group.add(point);
            sceneStateRef.current.allBoneIndicators[name] = point;
        });

        const highlightGeo = new THREE.SphereGeometry(0.018, 16, 16);
        const highlightMat = new THREE.MeshBasicMaterial({ color: 0x0a84ff, transparent: true, opacity: 0.9, depthTest: false });
        const highlight = new THREE.Mesh(highlightGeo, highlightMat);
        highlight.renderOrder = 999;
        highlight.visible = false;
        scene.add(highlight);
        sceneStateRef.current.boneIndicator = highlight;

        const ringGeo = new THREE.RingGeometry(0.025, 0.04, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x0a84ff, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthTest: false });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.renderOrder = 998;
        ring.visible = false;
        scene.add(ring);
        sceneStateRef.current.boneIndicatorRing = ring;
    };

    const createSkeletonHelper = () => {
        const { scene, model, skeletonHelper } = sceneStateRef.current;
        if (skeletonHelper) scene.remove(skeletonHelper);
        const helper = new THREE.SkeletonHelper(model);
        helper.material.linewidth = 1;
        helper.material.color.setHex(0x0a84ff);
        helper.material.transparent = true;
        helper.material.opacity = 0.25;
        helper.material.depthTest = false;
        helper.renderOrder = 900;
        scene.add(helper);
        sceneStateRef.current.skeletonHelper = helper;
    };

    const loadModelAndTPose = async () => {
        setLoading('Loading 3D model…');
        try {
            const [glbRes, tPoseRes] = await Promise.all([
                fetch('/pose-temp/lady-x-bot.glb'),
                fetch('/pose-temp/extract-t-pose.json'),
            ]);

            if (!glbRes.ok || !tPoseRes.ok) {
                throw new Error('Model assets were not found');
            }

            const [glbBuffer, tPoseData] = await Promise.all([
                glbRes.arrayBuffer(),
                tPoseRes.json(),
            ]);

            const loader = new GLTFLoader();
            await new Promise((resolve, reject) => {
                loader.parse(glbBuffer, '', (gltf) => {
                    const { scene, modelContainer } = sceneStateRef.current;
                    
                    if (sceneStateRef.current.model) {
                        scene.remove(sceneStateRef.current.model);
                    }

                    if (!modelContainer) {
                        sceneStateRef.current.modelContainer = new THREE.Group();
                        scene.add(sceneStateRef.current.modelContainer);
                    } else {
                        modelContainer.clear();
                    }

                    const model = gltf.scene;
                    sceneStateRef.current.model = model;
                    sceneStateRef.current.modelContainer.add(model);

                    // Normalize: scale + center the *scaled* bounds (GLBs can have baked offsets/scales).
                    model.position.set(0, 0, 0);
                    model.updateMatrixWorld(true);

                    const initialBox = new THREE.Box3().setFromObject(model);
                    const initialSize = initialBox.getSize(new THREE.Vector3());
                    const scaleFactor = 1.8 / Math.max(initialSize.x, initialSize.y, initialSize.z, 0.000001);

                    // Preserve any existing model scale by multiplying.
                    model.scale.multiplyScalar(scaleFactor);
                    model.updateMatrixWorld(true);

                    const scaledBox = new THREE.Box3().setFromObject(model);
                    const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
                    model.position.x -= scaledCenter.x;
                    model.position.z -= scaledCenter.z;
                    model.updateMatrixWorld(true);

                    // Put feet on the "floor".
                    const groundedBox = new THREE.Box3().setFromObject(model);
                    model.position.y -= groundedBox.min.y;
                    model.updateMatrixWorld(true);

                    // Fit camera to the model's *actual* post-transform bounds.
                    const camera = sceneStateRef.current.camera;
                    const objectForFit = sceneStateRef.current.modelContainer || model;
                    const fittedBox = new THREE.Box3().setFromObject(objectForFit);

                    if (!fittedBox.isEmpty() && camera) {
                        const fittedSize = fittedBox.getSize(new THREE.Vector3());
                        const fittedCenter = fittedBox.getCenter(new THREE.Vector3());
                        sceneStateRef.current.target.copy(fittedCenter);

                        const viewportW = viewportRef.current?.clientWidth || width;
                        const viewportH = viewportRef.current?.clientHeight || height;
                        const aspect = viewportW / Math.max(1, viewportH);
                        const fov = (camera.fov * Math.PI) / 180;
                        const hFit = fittedSize.y / (2 * Math.tan(fov / 2));
                        const wFit = fittedSize.x / (2 * Math.tan(fov / 2) * aspect);
                        sceneStateRef.current.spherical.radius = Math.max(hFit, wFit) * 2.0;
                    }

                    if (sceneStateRef.current.updateCamera) sceneStateRef.current.updateCamera();

                    sceneStateRef.current.boneMap = {};
                    sceneStateRef.current.baseBonePositions = {};
                    sceneStateRef.current.currentPoseRotations = {};
                    model.traverse((object) => {
                        if (object.isBone) {
                            sceneStateRef.current.boneMap[object.name] = object;
                            sceneStateRef.current.baseBonePositions[object.name] = object.position.clone();
                            sceneStateRef.current.currentPoseRotations[object.name] = {
                                x: object.rotation.x,
                                y: object.rotation.y,
                                z: object.rotation.z,
                            };
                        }
                    });

                    sceneStateRef.current.tPoseData = tPoseData;
                    createBoneIndicators();
                    createSkeletonHelper();
                    
                    const bones = Object.keys(sceneStateRef.current.boneMap);
                    const categories = {
                        'Core': bones.filter(b => /Hips|Spine|Neck|Head/i.test(b) && !/Hand|Foot|Toe/i.test(b)),
                        'Left Arm': bones.filter(b => /Left/i.test(b) && /Arm|Shoulder|ForeArm|Hand/i.test(b)),
                        'Right Arm': bones.filter(b => /Right/i.test(b) && /Arm|Shoulder|ForeArm|Hand/i.test(b)),
                        'Left Leg': bones.filter(b => /Left/i.test(b) && /Leg|Foot|Toe/i.test(b)),
                        'Right Leg': bones.filter(b => /Right/i.test(b) && /Leg|Foot|Toe/i.test(b)),
                        'Other': bones.filter(b => !/Hips|Spine|Neck|Head|Arm|Shoulder|ForeArm|Hand|Leg|Foot|Toe/i.test(b))
                    };
                    setBoneCategories(categories);
                    setBoneNames(bones);
                    setIsModelReady(true);
                    showToast('✅', `Model loaded`);
                    resolve();
                }, undefined, reject);
            });

            applyTPose();
            setModelPosition({ x: 0, y: 0, z: 0 });
            clearLoading();
        } catch (error) {
            showToast('⚠️', 'Could not load the 3D model');
            clearLoading();
        }
    };

    const initScene = () => {
        if (!viewportRef.current || initializedRef.current) {
            return undefined;
        }
        initializedRef.current = true;

        const viewport = viewportRef.current;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);

        const width = viewport.clientWidth || 800;
        const height = viewport.clientHeight || 600;
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
        const state = sceneStateRef.current;
        state.scene = scene;
        state.camera = camera;

        const updateCamera = () => {
            camera.position.set(
                state.target.x + state.spherical.radius * Math.sin(state.spherical.phi) * Math.sin(state.spherical.theta),
                state.target.y + state.spherical.radius * Math.cos(state.spherical.phi),
                state.target.z + state.spherical.radius * Math.sin(state.spherical.phi) * Math.cos(state.spherical.theta)
            );
            camera.lookAt(state.target);
        };
        state.updateCamera = updateCamera;

        updateCamera();

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(width, height, false);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        viewport.appendChild(renderer.domElement);
        state.renderer = renderer;

        const floorGeo = new THREE.PlaneGeometry(20, 20);
        const floorMat = new THREE.MeshBasicMaterial({ color: 0x0d0d0d, transparent: true, opacity: 0.5 });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.01;
        scene.add(floor);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const directional = new THREE.DirectionalLight(0xffffff, 1.0);
        directional.position.set(2, 4, 3);
        scene.add(directional);
        const fill = new THREE.DirectionalLight(0xffffff, 0.25);
        fill.position.set(-2, 1, -3);
        scene.add(fill);
        const rim = new THREE.DirectionalLight(0xeeeeff, 0.15);
        rim.position.set(0, 0, 4);
        scene.add(rim);
        scene.add(new THREE.AxesHelper(1));

        const handlePointerDown = (event) => {
            state.isDragging = true;
            state.lastMouse = { x: event.clientX, y: event.clientY };
        };

        const handlePointerMove = (event) => {
            if (!state.isDragging) return;
            state.spherical.theta -= (event.clientX - state.lastMouse.x) * 0.01;
            state.spherical.phi = Math.max(0.2, Math.min(Math.PI - 0.2, state.spherical.phi - (event.clientY - state.lastMouse.y) * 0.01));
            state.lastMouse = { x: event.clientX, y: event.clientY };
            updateCamera();
        };

        const handlePointerUp = () => {
            state.isDragging = false;
        };

        const handleWheel = (event) => {
            state.spherical.radius = Math.max(0.5, Math.min(10, state.spherical.radius + event.deltaY * 0.005));
            updateCamera();
            event.preventDefault();
        };

        const raycastBonePoints = (event) => {
            if (!state.pointsGroup || !state.pointsGroup.visible) return;

            const rect = renderer.domElement.getBoundingClientRect();
            state.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            state.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            state.raycaster.setFromCamera(state.mouse, camera);
            const intersects = state.raycaster.intersectObjects(state.pointsGroup.children);

            if (intersects.length > 0) {
                const boneName = intersects[0].object.userData.boneName;
                if (boneName) {
                    selectBone(boneName);
                }
            }
        };

        renderer.domElement.addEventListener('pointerdown', handlePointerDown);
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);
        renderer.domElement.addEventListener('click', raycastBonePoints);
        renderer.domElement.addEventListener('wheel', handleWheel, { passive: false });

        const resizeObserver = new ResizeObserver(() => {
            if (!viewportRef.current || !state.camera || !state.renderer) return;
            const nextWidth = viewportRef.current.clientWidth || 800;
            const nextHeight = viewportRef.current.clientHeight || 600;
            state.camera.aspect = nextWidth / nextHeight;
            state.camera.updateProjectionMatrix();
            state.renderer.setSize(nextWidth, nextHeight, false);
            
            // Adjust radius to fit model on resize
            if (state.model) {
                const objectForFit = state.modelContainer || state.model;
                const box = new THREE.Box3().setFromObject(objectForFit);
                const size = box.getSize(new THREE.Vector3());
                const aspect = nextWidth / nextHeight;
                const fov = 45 * Math.PI / 180;
                const hFit = (size.y) / (2 * Math.tan(fov / 2));
                const wFit = (size.x) / (2 * Math.tan(fov / 2) * aspect);
                state.spherical.radius = Math.max(hFit, wFit) * 2.0;
                updateCamera();
            }
        });
        resizeObserver.observe(viewport);

        const renderLoop = () => {
            state.animationFrame = window.requestAnimationFrame(renderLoop);

            if (state.pointsGroup && state.pointsGroup.visible) {
                Object.entries(state.allBoneIndicators).forEach(([name, indicator]) => {
                    const bone = state.boneMap[name];
                    if (bone) {
                        const worldPos = new THREE.Vector3();
                        bone.getWorldPosition(worldPos);
                        indicator.position.copy(worldPos);
                    }
                });
            }

            if (state.boneIndicator && state.boneIndicator.visible && state.selectedBone && state.boneMap[state.selectedBone]) {
                const worldPos = new THREE.Vector3();
                state.boneMap[state.selectedBone].getWorldPosition(worldPos);
                state.boneIndicator.position.copy(worldPos);
                if (state.boneIndicatorRing) {
                    state.boneIndicatorRing.position.copy(worldPos);
                }
            }

            state.renderer.render(state.scene, state.camera);
        };

        renderLoop();

        return () => {
            resizeObserver.disconnect();
            renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
            window.removeEventListener('pointermove', handlePointerMove);
            window.removeEventListener('pointerup', handlePointerUp);
            renderer.domElement.removeEventListener('click', raycastBonePoints);
            renderer.domElement.removeEventListener('wheel', handleWheel);
            if (state.animationFrame) {
                window.cancelAnimationFrame(state.animationFrame);
            }
            renderer.dispose();
            if (viewport.contains(renderer.domElement)) {
                viewport.removeChild(renderer.domElement);
            }
        };
    };

    const initMediaPipe = async () => {
        setLoading('Loading MediaPipe Pose…');
        try {
            const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm');
            poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
                    delegate: 'GPU',
                },
                runningMode: 'IMAGE',
                numPoses: 1,
            });
            setIsPoseReady(true);
            showToast('🤖', 'MediaPipe Pose ready');
        } catch (error) {
            showToast('⚠️', 'MediaPipe failed to load');
        }
        clearLoading();
    };

    const drawLandmarks = (landmarks, imgEl) => {
        const canvas = overlayCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = imgEl.naturalWidth;
        canvas.height = imgEl.naturalHeight;
        canvas.style.width = `${imgEl.clientWidth}px`;
        canvas.style.height = `${imgEl.clientHeight}px`;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const connections = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
            [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32],
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
        ];

        ctx.strokeStyle = 'rgba(10, 132, 255, 0.6)';
        ctx.lineWidth = 3;
        connections.forEach(([a, b]) => {
            if (landmarks[a] && landmarks[b]) {
                const ax = landmarks[a].x * canvas.width;
                const ay = landmarks[a].y * canvas.height;
                const bx = landmarks[b].x * canvas.width;
                const by = landmarks[b].y * canvas.height;
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
            }
        });

        landmarks.forEach((landmark, index) => {
            const x = landmark.x * canvas.width;
            const y = landmark.y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = index >= 11 ? 'rgba(48, 209, 88, 0.9)' : 'rgba(255, 159, 10, 0.9)';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    };

    const convertPoseFromMediaPipe = (worldLandmarks) => {
        const required = [
            'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'nose', 'left_foot_index', 'right_foot_index',
        ];

        required.forEach((name) => {
            if (!worldLandmarks[name]) {
                throw new Error(`Missing landmark: ${name}`);
            }
        });

        applyTPose();

        const { boneMap, tPoseData, currentPoseRotations, model } = sceneStateRef.current;
        const hipCenter = mpMid(worldLandmarks, 'left_hip', 'right_hip');
        const shoulderCenter = mpMid(worldLandmarks, 'left_shoulder', 'right_shoulder');
        const spineDir = shoulderCenter.clone().sub(hipCenter).normalize();

        const targetDirs = {
            mixamorig_Hips: spineDir.clone(),
            mixamorig_Spine: spineDir.clone(),
            mixamorig_Spine1: spineDir.clone(),
            mixamorig_Spine2: spineDir.clone(),
            mixamorig_Neck: mp(worldLandmarks.nose).clone().sub(shoulderCenter).normalize(),
            mixamorig_Head: mp(worldLandmarks.nose).clone().sub(shoulderCenter).normalize(),
            mixamorig_LeftArm: mp(worldLandmarks.left_elbow).clone().sub(mp(worldLandmarks.left_shoulder)).normalize(),
            mixamorig_LeftForeArm: mp(worldLandmarks.left_wrist).clone().sub(mp(worldLandmarks.left_elbow)).normalize(),
            mixamorig_RightArm: mp(worldLandmarks.right_elbow).clone().sub(mp(worldLandmarks.right_shoulder)).normalize(),
            mixamorig_RightForeArm: mp(worldLandmarks.right_wrist).clone().sub(mp(worldLandmarks.right_elbow)).normalize(),
            mixamorig_LeftUpLeg: mp(worldLandmarks.left_knee).clone().sub(mp(worldLandmarks.left_hip)).normalize(),
            mixamorig_LeftLeg: mp(worldLandmarks.left_ankle).clone().sub(mp(worldLandmarks.left_knee)).normalize(),
            mixamorig_LeftFoot: mp(worldLandmarks.left_foot_index).clone().sub(mp(worldLandmarks.left_ankle)).normalize(),
            mixamorig_RightUpLeg: mp(worldLandmarks.right_knee).clone().sub(mp(worldLandmarks.right_hip)).normalize(),
            mixamorig_RightLeg: mp(worldLandmarks.right_ankle).clone().sub(mp(worldLandmarks.right_knee)).normalize(),
            mixamorig_RightFoot: mp(worldLandmarks.right_foot_index).clone().sub(mp(worldLandmarks.right_ankle)).normalize(),
        };

        const result = {};
        const processOrder = [
            'mixamorig_Hips', 'mixamorig_Spine', 'mixamorig_Spine1', 'mixamorig_Spine2',
            'mixamorig_Neck', 'mixamorig_Head',
            'mixamorig_LeftShoulder', 'mixamorig_RightShoulder',
            'mixamorig_LeftArm', 'mixamorig_LeftForeArm',
            'mixamorig_RightArm', 'mixamorig_RightForeArm',
            'mixamorig_LeftUpLeg', 'mixamorig_LeftLeg', 'mixamorig_LeftFoot',
            'mixamorig_RightUpLeg', 'mixamorig_RightLeg', 'mixamorig_RightFoot',
        ];

        processOrder.forEach((boneName) => {
            const bone = boneMap[boneName];
            if (!bone) return;

            if (boneName === 'mixamorig_LeftShoulder' || boneName === 'mixamorig_RightShoulder') {
                const tRotation = tPoseData[boneName];
                if (tRotation) {
                    result[boneName] = { x: tRotation.x, y: tRotation.y, z: tRotation.z };
                }
                return;
            }

            const targetDir = targetDirs[boneName];
            if (!targetDir) return;

            const currentDir = getBoneWorldDirection(bone, boneMap);
            if (!currentDir) return;

            const dot = currentDir.dot(targetDir);
            if (dot > 0.9999) {
                result[boneName] = {
                    x: Math.round(bone.rotation.x * 10000) / 10000,
                    y: Math.round(bone.rotation.y * 10000) / 10000,
                    z: Math.round(bone.rotation.z * 10000) / 10000,
                };
                return;
            }

            let deltaQ;
            if (dot < -0.9999) {
                const perp = Math.abs(currentDir.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
                const axis = new THREE.Vector3().crossVectors(currentDir, perp).normalize();
                deltaQ = new THREE.Quaternion().setFromAxisAngle(axis, Math.PI);
            } else {
                deltaQ = computePoleConstrainedRotation(boneName, currentDir, targetDir, worldLandmarks);
            }

            const parentWorldQ = new THREE.Quaternion();
            if (bone.parent) {
                bone.parent.getWorldQuaternion(parentWorldQ);
            }

            const localDelta = parentWorldQ.clone().invert().multiply(deltaQ).multiply(parentWorldQ);
            const currentLocalQ = bone.quaternion.clone();
            const newLocalQ = localDelta.multiply(currentLocalQ);

            const euler = new THREE.Euler();
            euler.setFromQuaternion(newLocalQ, bone.rotation.order || 'XYZ');
            const clamped = clampEuler(euler, boneName);

            if (boneName.includes('UpLeg')) {
                clamped.z = Math.PI;
            }

            bone.rotation.set(clamped.x, clamped.y, clamped.z);
            model.updateMatrixWorld(true);

            const rotation = {
                x: Math.round(clamped.x * 10000) / 10000,
                y: Math.round(clamped.y * 10000) / 10000,
                z: Math.round(clamped.z * 10000) / 10000,
            };
            result[boneName] = rotation;
            currentPoseRotations[boneName] = rotation;
        });

        const hips = boneMap.mixamorig_Hips;
        if (hips) {
            const ankleY = (mp(worldLandmarks.left_ankle).y + mp(worldLandmarks.right_ankle).y) * 0.5;
            const hipY = hipCenter.y - ankleY;
            hips.position.x = hipCenter.x;
            hips.position.y = hipY;
            hips.position.z = hipCenter.z;

            result.mixamorig_Hips = result.mixamorig_Hips || {};
            result.mixamorig_Hips.position = {
                x: Math.round(hips.position.x * 1000) / 1000,
                y: Math.round(hips.position.y * 1000) / 1000,
                z: Math.round(hips.position.z * 1000) / 1000,
            };
        }

        Object.keys(tPoseData).forEach((name) => {
            if (!result[name] && boneMap[name] && (name.includes('Hand') || name.includes('Thumb') || name.includes('Index') || name.includes('Middle') || name.includes('Ring') || name.includes('Pinky') || name.includes('Toe'))) {
                result[name] = { ...tPoseData[name] };
                currentPoseRotations[name] = { ...tPoseData[name] };
            }
        });

        if (sceneStateRef.current.selectedBone) {
            refreshBoneEditorState(sceneStateRef.current.selectedBone);
        }

        return result;
    };

    const handleImageFile = (file) => {
        if (!file || !file.type.startsWith('image/')) {
            showToast('⚠️', 'Please upload an image file');
            return;
        }

        if (previewUrlRef.current) {
            URL.revokeObjectURL(previewUrlRef.current);
        }

        const nextUrl = URL.createObjectURL(file);
        previewUrlRef.current = nextUrl;
        setImageSrc(nextUrl);
        setImageReady(false);
        setImageStats('');
        setHasPoseData(false);
        setExportModalOpen(false);
        setExportText('');
        setBadge({ text: file.name, tone: 'info' });

        if (overlayCanvasRef.current) {
            const context = overlayCanvasRef.current.getContext('2d');
            if (context) {
                context.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
            }
        }

        setStep(1);
        showToast('🖼️', 'Image loaded — click Extract & Apply');
    };

    const handleExtractAndApply = async () => {
        const img = previewImgRef.current;
        const landmarker = poseLandmarkerRef.current;

        if (!img || !imageReady || !landmarker || !isModelReady) {
            showToast('⚠️', 'Not ready — model or MediaPipe is still loading');
            return;
        }

        setStep(2);
        setLoading('Extracting pose from image…');
        setBadge({ text: 'Processing…', tone: 'info' });

        try {
            const result = landmarker.detect(img);
            if (!result.worldLandmarks || result.worldLandmarks.length === 0) {
                throw new Error('No pose detected in this image');
            }

            drawLandmarks(result.landmarks[0], img);
            const namedWorldLandmarks = indexedToNamed(result.worldLandmarks[0]);

            const stats = [];
            stats.push(`✅ ${result.worldLandmarks[0].length} landmarks detected`);
            stats.push('');
            stats.push('Key points:');
            ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].forEach((key) => {
                const landmark = namedWorldLandmarks[key];
                if (landmark) {
                    stats.push(`  ${key}: (${landmark.x.toFixed(3)}, ${landmark.y.toFixed(3)}, ${landmark.z.toFixed(3)})`);
                }
            });
            setImageStats(stats.join('\n'));

            const mixamo = convertPoseFromMediaPipe(namedWorldLandmarks);
            sceneStateRef.current.currentMixamoOutput = mixamo;
            setHasPoseData(true);
            setBadge({ text: `${Object.keys(mixamo).length} bones mapped`, tone: 'success' });
            setStep(3);
            showToast('✨', `Pose applied — ${Object.keys(mixamo).length} bones`);
        } catch (error) {
            setBadge({ text: 'Error', tone: 'info' });
            setStep(1);
            showToast('⚠️', error.message || 'Failed to extract pose');
        } finally {
            clearLoading();
        }
    };

    const handleExportJson = () => {
        const baseOutput = sceneStateRef.current.currentMixamoOutput || {};
        const output = { ...baseOutput };

        Object.entries(sceneStateRef.current.currentPoseRotations).forEach(([name, rotation]) => {
            if (Math.abs(rotation.x) > 0.001 || Math.abs(rotation.y) > 0.001 || Math.abs(rotation.z) > 0.001) {
                output[name] = {
                    x: Number(rotation.x.toFixed(4)),
                    y: Number(rotation.y.toFixed(4)),
                    z: Number(rotation.z.toFixed(4)),
                };
            }
        });

        setExportText(JSON.stringify(output, null, 2));
        setExportModalOpen(true);
    };

    const handleExportGlb = () => {
        const { modelContainer } = sceneStateRef.current;
        if (!modelContainer) return;
        showToast('⏳', 'Exporting GLB…');

        const exporter = new GLTFExporter();
        exporter.parse(modelContainer, (result) => {
            if (result instanceof ArrayBuffer) {
                const blob = new Blob([result], { type: 'application/octet-stream' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'character-pose.glb';
                link.click();
                URL.revokeObjectURL(link.href);
                showToast('✅', 'GLB exported successfully');
            }
        }, { binary: true });
    };

    useEffect(() => {
        const cleanupScene = initScene();
        initMediaPipe();
        loadModelAndTPose();

        return () => {
            cleanupScene?.();
            initializedRef.current = false;
            if (previewUrlRef.current) {
                URL.revokeObjectURL(previewUrlRef.current);
            }
            if (toastTimerRef.current) {
                clearTimeout(toastTimerRef.current);
            }
            if (poseLandmarkerRef.current?.close) {
                try {
                    poseLandmarkerRef.current.close();
                } catch (error) {}
            }
        };
    }, []);

    useEffect(() => {
        if (sceneStateRef.current.pointsGroup) {
            sceneStateRef.current.pointsGroup.visible = showAllPoints;
        }
    }, [showAllPoints]);

    useEffect(() => {
        if (selectedBone) {
            const rotation = sceneStateRef.current.currentPoseRotations[selectedBone];
            if (rotation) {
                setBoneRotation(rotation);
            }
        }
    }, [selectedBone]);

    return (
        <div className="pose-editor-studio">
            <div className="studio-topbar">
                <div className="studio-logo">
                    <div className="studio-logo-icon">📷</div>
                    Image → Pose
                </div>

                <div className="step-breadcrumbs">
                    <span className={`breadcrumb-item ${step >= 1 ? 'active' : ''}`}>Upload Image</span>
                    <svg className="breadcrumb-separator" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6" /></svg>
                    <span className={`breadcrumb-item ${step >= 2 ? 'active' : ''}`}>Extract Pose</span>
                    <svg className="breadcrumb-separator" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6" /></svg>
                    <span className={`breadcrumb-item ${step >= 3 ? 'active' : ''}`}>Apply to Model</span>
                </div>

                <div className="studio-spacer" />

                <div className="topbar-actions">
                    <div className="panel-toggles">
                        <button className={`topbar-btn ${showImagePanel ? 'active' : ''}`} onClick={() => setShowImagePanel((value) => !value)}>
                            Image
                        </button>
                        <button className={`topbar-btn ${showBonePanel ? 'active' : ''}`} onClick={() => setShowBonePanel((value) => !value)}>
                            Bones
                        </button>
                        <button className={`topbar-btn ${showAllPoints ? 'active' : ''}`} onClick={() => setShowAllPoints((value) => !value)}>
                            Points
                        </button>
                    </div>

                    <div className="export-actions">
                        <button className="topbar-btn" disabled={!hasPoseData} onClick={handleExportJson}>
                            JSON
                        </button>
                        <button className="topbar-btn primary" disabled={!hasPoseData} onClick={handleExportGlb}>
                            GLB
                        </button>
                    </div>
                </div>
            </div>

            <div className="studio-main">
                <aside className={`image-panel ${showImagePanel ? '' : 'hidden'}`}>
                    <div className="panel-header">
                        <span className="panel-title">{leftPanelMode === 'image' ? 'Source Image' : 'Paste JSON'}</span>
                        <span className={`panel-badge ${badge.tone}`}>{badge.text}</span>
                    </div>

                    <div className="segment-control" style={{ margin: '12px 20px 0' }}>
                        <button className={`segment-btn ${leftPanelMode === 'image' ? 'active' : ''}`} onClick={() => setLeftPanelMode('image')}>Image</button>
                        <button className={`segment-btn ${leftPanelMode === 'json' ? 'active' : ''}`} onClick={() => setLeftPanelMode('json')}>JSON</button>
                    </div>

                    {leftPanelMode === 'image' ? (
                        <>
                            <label className={`image-drop ${imageSrc ? 'has-image' : ''}`}>
                                <input
                                    ref={imageInputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={(event) => handleImageFile(event.target.files?.[0])}
                                />

                                {!imageSrc && (
                                    <div className="drop-placeholder">
                                        <div className="drop-icon">🖼️</div>
                                        <div className="drop-title">Drop your image here</div>
                                    </div>
                                )}

                                <div className={`image-preview ${imageSrc ? 'visible' : ''}`}>
                                    {imageSrc && (
                                        <>
                                            <img
                                                ref={previewImgRef}
                                                src={imageSrc}
                                                alt="Uploaded image"
                                                onLoad={() => setImageReady(true)}
                                            />
                                            <canvas ref={overlayCanvasRef} />
                                        </>
                                    )}
                                </div>

                                <div className="landmark-stats" style={{ display: imageStats ? 'block' : 'none' }}>
                                    {imageStats}
                                </div>
                            </label>

                            <div className="image-actions">
                                <button className="action-btn reset-btn" onClick={() => window.location.reload()}>
                                    ↻ Reset
                                </button>
                                <button className="action-btn extract" disabled={!imageReady || !isModelReady} onClick={handleExtractAndApply}>
                                    ✨ Extract & Apply Pose
                                </button>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="json-paste-area">
                                <textarea
                                    className="json-textarea"
                                    placeholder={'Paste Mixamo-compatible JSON here…\n\nExample:\n{\n  "mixamorig_Hips": { "x": 0, "y": 0, "z": 0 },\n  "mixamorig_Spine": { "x": 0.1, "y": 0, "z": 0 }\n}'}
                                    value={jsonInputText}
                                    onChange={(e) => { setJsonInputText(e.target.value); setJsonError(''); }}
                                    spellCheck={false}
                                />
                                {jsonError && <div className="json-error">{jsonError}</div>}
                                <label className="json-file-btn">
                                    <input type="file" accept=".json,application/json" onChange={(e) => handleJsonFile(e.target.files?.[0])} />
                                    Browse JSON file…
                                </label>
                            </div>

                            <div className="image-actions">
                                <button
                                    className="action-btn reset-btn"
                                    onClick={() => { setJsonInputText(''); setJsonError(''); setBadge({ text: 'Waiting', tone: 'info' }); }}
                                >
                                    ↻ Clear
                                </button>
                                <button className="action-btn extract" disabled={!jsonInputText.trim() || !isModelReady} onClick={handleApplyJsonText}>
                                    ✨ Apply Pose
                                </button>
                            </div>
                        </>
                    )}
                </aside>

                <section className="viewport-shell">
                    <div ref={viewportRef} className="three-viewport">
                        <div className="hint">Drag to orbit · Scroll to zoom</div>

                        <div className="axes-legend">
                            <div className="legend-item"><div className="legend-color x" /> X Axis</div>
                            <div className="legend-item"><div className="legend-color y" /> Y Axis</div>
                            <div className="legend-item"><div className="legend-color z" /> Z Axis</div>
                        </div>

                        {isLoading && (
                            <div className="loading-overlay visible">
                                <div className="spinner" />
                                <span className="loading-text">{loadingText}</span>
                            </div>
                        )}
                    </div>
                </section>

                <aside className={`bone-panel ${showBonePanel ? '' : 'hidden'}`}>
                    <div className="segment-control">
                        <button className={`segment-btn ${activePanel === 'view-bones' ? 'active' : ''}`} onClick={() => setActivePanel('view-bones')}>Bones</button>
                        <button className={`segment-btn ${activePanel === 'view-presets' ? 'active' : ''}`} onClick={() => setActivePanel('view-presets')}>Presets</button>
                        <button className={`segment-btn ${activePanel === 'view-model' ? 'active' : ''}`} onClick={() => setActivePanel('view-model')}>Model</button>
                    </div>

                    {activePanel === 'view-bones' ? (
                        <div className="panel-view active">
                            <div className="search-box">
                                <input
                                    type="text"
                                    id="boneSearch"
                                    placeholder="Search bones..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                />
                                {searchQuery && <button className="clear-search" onClick={() => setSearchQuery('')}>✕</button>}
                            </div>

                            <div id="boneList">
                                {Object.entries(boneCategories).map(([catName, bones]) => {
                                    const filtered = bones.filter(name => name.toLowerCase().includes(searchQuery.toLowerCase()));
                                    if (filtered.length === 0) return null;
                                    return (
                                        <div key={catName} className="bone-group">
                                            <div className="bone-group-header">{catName}</div>
                                            {filtered.map((name) => (
                                                <button
                                                    key={name}
                                                    className={`bone-item ${selectedBone === name ? 'selected' : ''}`}
                                                    onClick={() => selectBone(name)}
                                                >
                                                    {name.replace('mixamorig_', '')}
                                                </button>
                                            ))}
                                        </div>
                                    );
                                })}
                            </div>

                            <div className="panel-actions">
                                <button className="panel-action-btn danger" onClick={handleResetAll}>
                                    Reset Full Pose
                                </button>
                            </div>

                            <div id="boneEditor">
                                <div id="boneEditorTitle">{selectedBone ? selectedBone.replace('mixamorig_', '') : 'Select a bone'}</div>
                                {selectedBone ? (
                                    <div id="axisControls">
                                        <div className="axis-row">
                                            <span className="axis-label x">X</span>
                                            <input type="range" min="-3.14" max="3.14" step="0.01" value={boneRotation.x} onChange={(event) => setBoneRotationValue('x', Number(event.target.value))} />
                                            <span className="axis-val">{boneRotation.x.toFixed(2)}</span>
                                        </div>
                                        <div className="axis-row">
                                            <span className="axis-label y">Y</span>
                                            <input type="range" min="-3.14" max="3.14" step="0.01" value={boneRotation.y} onChange={(event) => setBoneRotationValue('y', Number(event.target.value))} />
                                            <span className="axis-val">{boneRotation.y.toFixed(2)}</span>
                                        </div>
                                        <div className="axis-row">
                                            <span className="axis-label z">Z</span>
                                            <input type="range" min="-3.14" max="3.14" step="0.01" value={boneRotation.z} onChange={(event) => setBoneRotationValue('z', Number(event.target.value))} />
                                            <span className="axis-val">{boneRotation.z.toFixed(2)}</span>
                                        </div>
                                        <button
                                            id="resetBoneBtn"
                                            type="button"
                                            onClick={() => {
                                                if (!selectedBone || !sceneStateRef.current.boneMap[selectedBone]) return;
                                                sceneStateRef.current.boneMap[selectedBone].rotation.set(0, 0, 0);
                                                sceneStateRef.current.currentPoseRotations[selectedBone] = { x: 0, y: 0, z: 0 };
                                                setBoneRotation({ x: 0, y: 0, z: 0 });
                                                if (sceneStateRef.current.currentMixamoOutput) {
                                                    delete sceneStateRef.current.currentMixamoOutput[selectedBone];
                                                }
                                            }}
                                        >
                                            Reset Bone
                                        </button>
                                    </div>
                                ) : null}
                            </div>
                        </div>
                    ) : activePanel === 'view-presets' ? (
                        <div className="panel-view active">
                            <div className="preset-list">
                                {POSE_PRESETS.map((preset) => (
                                    <button
                                        key={preset.id}
                                        type="button"
                                        className="preset-card"
                                        disabled={!isModelReady}
                                        onClick={() => applyPosePreset(preset)}
                                    >
                                        <span className="preset-name">{preset.name}</span>
                                        <span className="preset-description">{preset.description}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="panel-view active">
                            <div id="boneEditor" style={{ borderTop: 'none', paddingTop: 4 }}>
                                <div id="boneEditorTitle">Global Translation</div>
                                <div className="axis-row">
                                    <span className="axis-label x">X</span>
                                    <input type="range" min="-5" max="5" step="0.01" value={modelPosition.x} onChange={(event) => setModelTranslation('x', Number(event.target.value))} />
                                    <span className="axis-val">{modelPosition.x.toFixed(2)}</span>
                                </div>
                                <div className="axis-row">
                                    <span className="axis-label y">Y</span>
                                    <input type="range" min="0" max="5" step="0.01" value={modelPosition.y} onChange={(event) => setModelTranslation('y', Number(event.target.value))} />
                                    <span className="axis-val">{modelPosition.y.toFixed(2)}</span>
                                </div>
                                <div className="axis-row">
                                    <span className="axis-label z">Z</span>
                                    <input type="range" min="-5" max="5" step="0.01" value={modelPosition.z} onChange={(event) => setModelTranslation('z', Number(event.target.value))} />
                                    <span className="axis-val">{modelPosition.z.toFixed(2)}</span>
                                </div>
                                <button
                                    id="resetModelBtn"
                                    type="button"
                                    onClick={() => {
                                        setModelPosition({ x: 0, y: 0, z: 0 });
                                        if (sceneStateRef.current.modelContainer) {
                                            sceneStateRef.current.modelContainer.position.set(0, 0, 0);
                                        }
                                    }}
                                >
                                    Reset Position
                                </button>
                            </div>
                        </div>
                    )}
                </aside>
            </div>

            {exportModalOpen && (
                <>
                    <div className="modal-overlay visible" onClick={() => setExportModalOpen(false)} />
                    <div className="modal-box visible">
                        <h3>Pose JSON</h3>
                        <div className="modal-subtitle">Mixamo-compatible bone rotations</div>
                        <textarea value={exportText} readOnly />
                        <div className="modal-actions">
                            <button id="modalCloseBtn" type="button" onClick={() => setExportModalOpen(false)}>Close</button>
                            <button
                                id="modalCopyBtn"
                                type="button"
                                onClick={() => {
                                    if (!navigator.clipboard) {
                                        showToast('⚠️', 'Clipboard API not available');
                                        return;
                                    }
                                    navigator.clipboard.writeText(exportText).then(() => {
                                        showToast('📋', 'Copied to clipboard');
                                    }).catch(() => {
                                        showToast('⚠️', 'Failed to copy');
                                    });
                                }}
                            >
                                Copy to Clipboard
                            </button>
                        </div>
                    </div>
                </>
            )}

            <div className={`toast ${toast.visible ? 'visible' : ''}`}>
                <span>{toast.icon}</span>
                <span>{toast.text}</span>
            </div>
        </div>
    );
}
