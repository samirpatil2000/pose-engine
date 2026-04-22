# Pose Engine

A premium suite of tools for creating, extracting, and exploring 3D poses and 360-degree visuals with precision and ease.

## Features

**Pose Engine** provides three powerful tools accessible from an intuitive landing page:

### 🎯 **Pose Editor**
Create and refine 3D character poses with real-time preview and precise bone control. Adjust joint rotations, preview your changes instantly, and export poses in JSON format.

### 📸 **Pose Extractor**
Extract 3D skeletal poses from images, videos, or live webcam input using advanced computer vision. Supports batch processing of video frames and real-time pose detection.

### 🔄 **360 Viewer**
Explore equirectangular panorama images in an immersive spherical view. Drag to rotate, scroll to zoom, and experience seamless 360-degree panoramic content.

## Getting Started

### Prerequisites
- Node.js 16+
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3002 in your browser

## Available Scripts

- `npm run dev` — Start development server (default port 3002)
- `npm run build` — Build for production
- `npm run preview` — Preview production build
- `npm run lint` — Run TypeScript type checking

## Technology Stack

- **React 19** — UI framework
- **Three.js** — 3D rendering and panorama visualization
- **MediaPipe** — Pose detection and skeletal analysis
- **Vite** — Build tool and dev server
- **TypeScript** — Type safety

## Sample Poses

The app includes pre-built reference poses:
- T-Pose (arms extended horizontally)
- Standing (default upright)
- Action Hero (dynamic lunge pose)
- Sitting (seated pose)
- Walking (mid-stride animation frame)

## Output Formats

Poses can be exported in multiple formats:
- **App Format** — Optimized 18-joint skeleton
- **Raw MediaPipe** — Full 33-point landmark data
- **JSON** — Standard structured format for integration

## Architecture

```
src/
├── pages/
│   ├── Home.jsx           # Landing page with feature tiles
│   ├── PoseEditor.jsx     # 3D pose manipulation interface
│   ├── PoseExtractor.jsx  # Vision-based pose detection
│   └── Viewer360.jsx      # Panoramic image viewer
├── components/
│   ├── TileCard.jsx       # Reusable feature tile component
│   └── Toast.jsx          # Notification system
├── hooks/
│   ├── useModelPreview.js # 3D scene management
│   ├── usePoseEditor.js   # Pose editing logic
│   └── usePoseExtractor.js# Pose detection pipeline
└── utils/
    ├── boneMappings.js    # Skeleton bone configuration
    └── poseConfig.js      # Shared pose utilities
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

WebGL 2.0 support required for 3D rendering.

## License

MIT
