import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import './App.css';
import PoseEditor from './pages/PoseEditor';
import PoseExtractor from './pages/PoseExtractor';

function TabBar() {
  return (
    <nav className="tab-bar">
      <NavLink 
        to="/" 
        className={({ isActive }) => isActive ? "active" : ""}
        end
      >
        Pose Editor
      </NavLink>
      <NavLink 
        to="/extract" 
        className={({ isActive }) => isActive ? "active" : ""}
      >
        Pose Extractor
      </NavLink>
    </nav>
  );
}

function AppRoutes() {
  const location = useLocation();
  const showTabBar = location.pathname !== '/';

  return (
    <>
      {showTabBar && <TabBar />}
      <Routes>
        <Route path="/" element={<PoseEditor />} />
        <Route path="/extract" element={<PoseExtractor />} />
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
