import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import PoseEditor from './pages/PoseEditor';
import PoseExtractor from './pages/PoseExtractor';
import Viewer360 from './pages/Viewer360';

function TabBar() {
  return (
    <nav className="tab-bar">
      <NavLink 
        to="/" 
        className={({ isActive }) => isActive ? "active" : ""}
        end
      >
        Home
      </NavLink>
      <NavLink 
        to="/editor" 
        className={({ isActive }) => isActive ? "active" : ""}
      >
        Pose Editor
      </NavLink>
      <NavLink 
        to="/extract" 
        className={({ isActive }) => isActive ? "active" : ""}
      >
        Pose Extractor
      </NavLink>
      <NavLink 
        to="/360" 
        className={({ isActive }) => isActive ? "active" : ""}
      >
        360 Viewer
      </NavLink>
    </nav>
  );
}

function AppRoutes() {
  return (
    <>
      <TabBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/editor" element={<PoseEditor />} />
        <Route path="/extract" element={<PoseExtractor />} />
        <Route path="/360" element={<Viewer360 />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
