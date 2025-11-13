import React from 'react';
import './ValidationToolbar.css';

function ValidationToolbar({ 
  currentImage,
  currentMask,
  annotations,
  onUpdateAnnotation,
  onGoToPrev,
  onGoToNext,
  onGoToPrevMask,
  onGoToNextMask,
  hasVideo
}) {
  if (!currentImage || !currentMask) {
    return (
      <div className="validation-toolbar">
        <div className="no-selection-message">Select an image and mask to annotate</div>
      </div>
    );
  }

  const handleToggle = (field, value) => {
    onUpdateAnnotation(field, value);
  };

  return (
    <div className="validation-toolbar">
      <div className="annotation-row">
        <div className="annotation-section">
          <div className="section-label">Localization Correct:</div>
          <div className="button-wrapper">
            <button 
              className={annotations.localization === true ? 'active yes' : ''}
              onClick={() => handleToggle('localization', true)}
              disabled={!hasVideo}
            >
              Yes
            </button>
            <button 
              className={annotations.localization === false ? 'active no' : ''}
              onClick={() => handleToggle('localization', false)}
              disabled={!hasVideo}
            >
              No
            </button>
          </div>
        </div>

        <div className="annotation-section">
          <div className="section-label">Articulation Type Correct:</div>
          <div className="button-wrapper">
            <button 
              className={annotations.articulation_type === true ? 'active yes' : ''}
              onClick={() => handleToggle('articulation_type', true)}
              disabled={!hasVideo}
            >
              Yes
            </button>
            <button 
              className={annotations.articulation_type === false ? 'active no' : ''}
              onClick={() => handleToggle('articulation_type', false)}
              disabled={!hasVideo}
            >
              No
            </button>
          </div>
        </div>

        <div className="annotation-section">
          <div className="section-label">Sequence Plausible:</div>
          <div className="button-wrapper">
            <button 
              className={annotations.sequence_plausible === true ? 'active yes' : ''}
              onClick={() => handleToggle('sequence_plausible', true)}
              disabled={!hasVideo}
            >
              Yes
            </button>
            <button 
              className={annotations.sequence_plausible === false ? 'active no' : ''}
              onClick={() => handleToggle('sequence_plausible', false)}
              disabled={!hasVideo}
            >
              No
            </button>
          </div>
        </div>

        <div className="annotation-section perspective">
          <div className="section-label">Perspective:</div>
          <div className="button-wrapper">
            <div className="placeholder">TBD</div>
          </div>
        </div>
      </div>

      <div className="navigation-section">
        <div className="nav-buttons">
          <button onClick={onGoToPrev}>← Prev Image</button>
          <button onClick={onGoToPrevMask}>← Prev Mask</button>
          <div className="nav-spacer"></div>
          <button onClick={onGoToNextMask}>Next Mask →</button>
          <button onClick={onGoToNext}>Next Image →</button>
        </div>
      </div>
    </div>
  );
}

export default ValidationToolbar;