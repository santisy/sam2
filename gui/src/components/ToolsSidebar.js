import React from 'react';

function ToolsSidebar({
  cropMode,
  setCropMode,
  cropOrientation,
  setCropOrientation,
  cropSize,
  setCropSize,
  cropData,
  currentCrop,
  onSaveCrop,
  onDeleteCrop
}) {
  return (
    <div className="tools-sidebar">
      <h3>Tools</h3>
      
      <div className="tool-section">
        <h4>Cropping</h4>
        <button 
          className={`tool-btn ${cropMode ? 'active' : ''}`}
          onClick={() => setCropMode(!cropMode)}
        >
          {cropMode ? '✓ Crop Active' : 'Activate Crop'}
        </button>
        
        <div className="tool-control">
          <label>Orientation</label>
          <div className="orientation-toggle">
            <button 
              className={cropOrientation === 'landscape' ? 'active' : ''}
              onClick={() => setCropOrientation('landscape')}
            >
              16:9
            </button>
            <button 
              className={cropOrientation === 'portrait' ? 'active' : ''}
              onClick={() => setCropOrientation('portrait')}
            >
              9:16
            </button>
          </div>
        </div>
        
        <div className="tool-control">
          <label>Size</label>
          <input 
            type="range" 
            min="30" 
            max="100" 
            value={cropSize}
            onChange={(e) => setCropSize(parseInt(e.target.value))}
            className="size-slider"
          />
          <div className="size-value">{cropSize}%</div>
        </div>
        
        <div className="crop-actions">
          <button 
            className="save-crop-btn"
            onClick={onSaveCrop}
            disabled={!currentCrop}
          >
            Save Crop
          </button>
          {cropData && (
            <button 
              className="delete-crop-btn"
              onClick={onDeleteCrop}
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default ToolsSidebar;