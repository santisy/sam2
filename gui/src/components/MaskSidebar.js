import React from 'react';

const MASK_COLORS = [
  [100, 200, 255],
  [255, 100, 200],
  [255, 230, 100],
  [200, 150, 255],
  [100, 255, 200],
];

function MaskSidebar({ 
  currentImage, 
  masks, 
  activeMask, 
  activeMaskIndex, 
  prompt,
  onStartNewMask, 
  onEditMask, 
  onDeleteMask 
}) {
  return (
    <div className="mask-sidebar">
      <h3>Masks</h3>
      {currentImage && (
        <>
          <button onClick={onStartNewMask} className="new-mask-btn">+ New Mask</button>
          <div className="mask-list">
            {activeMaskIndex === null && activeMask && (
              <div className="mask-item unsaved selected">
                <div className="mask-info">
                  <div className="mask-header">
                    <div 
                      className="mask-color-swatch" 
                      style={{backgroundColor: 'rgb(0, 255, 0)'}}
                    />
                    <div className="mask-name">New Mask (unsaved)</div>
                  </div>
                  <div className="mask-prompt">{prompt || '(no prompt)'}</div>
                </div>
              </div>
            )}
            {masks.map((mask, idx) => {
              const color = MASK_COLORS[idx % MASK_COLORS.length];
              return (
                <div key={idx} className={`mask-item ${activeMaskIndex === idx ? 'selected' : ''}`}>
                  <div className="mask-info">
                    <div className="mask-header">
                      <div 
                        className="mask-color-swatch" 
                        style={{backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`}}
                      />
                      <div className="mask-name">Mask {idx + 1}</div>
                    </div>
                    <div className="mask-prompt">{mask.prompt || '(no prompt)'}</div>
                  </div>
                  <div className="mask-actions">
                    <button onClick={() => onEditMask(idx)}>Edit</button>
                    <button onClick={() => onDeleteMask(idx)} className="delete-btn">Del</button>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export default MaskSidebar;