import React from 'react';

function Sidebar({ 
  directory, 
  directoryInput, 
  setDirectoryInput, 
  images, 
  currentImage, 
  onChangeDirectory, 
  onSelectImage 
}) {
  return (
    <div className="sidebar">
      <h2>Images</h2>
      <div className="directory-changer">
        <input
          type="text"
          value={directoryInput}
          onChange={(e) => setDirectoryInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && onChangeDirectory()}
        />
        <button onClick={onChangeDirectory}>Load</button>
      </div>
      <div className="directory-info">{directory}</div>
      {images.length === 0 ? (
        <div className="no-images">No images found</div>
      ) : (
        <div className="image-list">
          {images.map(img => (
            <div
              key={img.path}
              className={`image-item ${currentImage?.path === img.path ? 'active' : ''}`}
              onClick={() => onSelectImage(img)}
            >
              {img.filename}
              {img.mask_count > 0 && <span className="count">({img.mask_count})</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Sidebar;