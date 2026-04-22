import React from 'react';
import { Zap, Image, RotateCcw } from 'lucide-react';
import TileCard from '../components/TileCard';
import './Home.css';

export default function Home() {
  const features = [
    {
      icon: Zap,
      title: 'Pose Editor',
      description: 'Transform 3D character poses with intuitive controls and real-time preview.',
      href: '/editor',
    },
    {
      icon: Image,
      title: 'Pose Extractor',
      description: 'Extract 3D poses from images, videos, or live webcam input instantly.',
      href: '/extract',
    },
    {
      icon: RotateCcw,
      title: '360 Viewer',
      description: 'Explore and rotate 360-degree imagery with smooth, immersive controls.',
      href: '/360',
    },
  ];

  return (
    <div className="home-container">
      <div className="home-content">
        {/* Hero Section */}
        <section className="hero">
          <div className="hero-text">
            <h1>Pose Sculpt</h1>
            <p>Create, extract, and explore 3D poses and 360-degree visuals with precision and ease.</p>
          </div>
        </section>

        {/* Tiles Grid */}
        <section className="tiles-grid">
          {features.map((feature, index) => (
            <TileCard
              key={index}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
              href={feature.href}
            />
          ))}
        </section>

        {/* Footer Note */}
        <section className="home-footer">
          <p>Choose a feature to get started. Each tool is built for speed and precision.</p>
        </section>
      </div>
    </div>
  );
}
