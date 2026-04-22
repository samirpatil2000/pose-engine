import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function TileCard({ icon: Icon, title, description, href }) {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(href);
  };

  return (
    <button 
      onClick={handleClick}
      className="tile-card"
      role="link"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          handleClick();
        }
      }}
    >
      <div className="tile-icon">
        {Icon && <Icon size={48} strokeWidth={1.5} />}
      </div>
      <div className="tile-content">
        <h3 className="tile-title">{title}</h3>
        <p className="tile-description">{description}</p>
      </div>
      <div className="tile-arrow">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M7 10h6M12 7l3 3-3 3" />
        </svg>
      </div>
    </button>
  );
}
