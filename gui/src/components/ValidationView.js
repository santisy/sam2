import React, { useState, useEffect } from 'react';
import './ValidationView.css';
import ValidationHeader from './ValidationHeader';
import ValidationToolbar from './ValidationToolbar';
import ValidationStatsPanel from './ValidationStatsPanel';

function ValidationView({ 
  apiBase,
  currentImage, 
  masks,
  activeMaskIndex,
  onSelectMask,
  selectedModel,
  onSelectModel,
  onGoToPrev,
  onGoToNext,
  onGoToPrevMask,
  onGoToNextMask,
  validationStats,
  onValidationStatsUpdate
}) {
  const currentMask = activeMaskIndex !== null ? masks[activeMaskIndex] : null;
  
  const [statsPanelOpen, setStatsPanelOpen] = useState(false);
  
  // Temporary local state for annotations (will connect to backend next)
  const [annotations, setAnnotations] = useState({
    localization: null,
    articulation_type: null,
    sequence_plausible: null,
    perspective: null
  });
  
  // Load validation data when image/mask/model changes
  useEffect(() => {
    const loadValidation = async () => {
      if (!currentImage || activeMaskIndex === null) {
        setAnnotations({
          localization: null,
          articulation_type: null,
          sequence_plausible: null,
          perspective: null
        });
        return;
      }
      
      const res = await fetch(
        `${apiBase}/api/validation?image_path=${encodeURIComponent(currentImage.path)}&mask_index=${activeMaskIndex}&model=${selectedModel}`
      );
      const data = await res.json();
      setAnnotations(data);
    };
    
    loadValidation();
  }, [currentImage, activeMaskIndex, selectedModel]);
  
  const handleUpdateAnnotation = async (field, value) => {
    setAnnotations(prev => ({
      ...prev,
      [field]: value
    }));
    
    if (!currentImage || activeMaskIndex === null) return;
    
    const res = await fetch(`${apiBase}/api/validation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: currentImage.path,
        mask_index: activeMaskIndex,
        model: selectedModel,
        field: field,
        value: value
      })
    });
    
    const data = await res.json();
    
    // Update global stats from server response
    if (data.global_stats) {
      onValidationStatsUpdate(data.global_stats);
    }
  };
  
  const getVideoUrl = () => {
    if (!currentImage || !currentMask) return null;
    
    // For baselines: sora2, veo3, kling
    // Video filename pattern: {imagestem}-{hash}-mask-{num}_video-001.mp4
    if (selectedModel === 'sora2' || selectedModel === 'veo3' || selectedModel === 'kling') {
      const maskFilename = currentMask.mask_filename;
      const videoFilename = maskFilename.replace('.png', '_video-001.mp4');
      return `${apiBase}/api/video/${selectedModel}/${videoFilename}`;
    }
    
    return null;
  };

  const videoUrl = getVideoUrl();

  return (
    <div className="validation-view">
      <ValidationHeader 
        apiBase={apiBase}
        currentImage={currentImage}
        masks={masks}
        activeMaskIndex={activeMaskIndex}
        onSelectMask={onSelectMask}
        mask={currentMask}
        selectedModel={selectedModel}
        onSelectModel={onSelectModel}
        validationStats={validationStats}
      />
      
      <div className="validation-workspace">
        {videoUrl && (
          <div className="debug-video-path">
            {videoUrl}
          </div>
        )}
        {videoUrl ? (
          <video 
            key={videoUrl}
            controls 
            autoPlay 
            loop 
            className="validation-video"
          >
            <source src={videoUrl} type="video/mp4" />
          </video>
        ) : (
          <div className="no-video-message">
            {currentMask ? 'Select a model to view video' : 'Select a mask first'}
          </div>
        )}
      </div>
      
      <ValidationToolbar 
        currentImage={currentImage}
        currentMask={currentMask}
        annotations={annotations}
        onUpdateAnnotation={handleUpdateAnnotation}
        onGoToPrev={onGoToPrev}
        onGoToNext={onGoToNext}
        onGoToPrevMask={onGoToPrevMask}
        onGoToNextMask={onGoToNextMask}
        hasVideo={!!videoUrl}
      />
      
      <ValidationStatsPanel 
        apiBase={apiBase}
        isOpen={statsPanelOpen}
        onToggle={() => setStatsPanelOpen(!statsPanelOpen)}
        validationStats={validationStats}
      />
    </div>
  );
}

export default ValidationView;