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
  onDeleteCrop,
  masks,
  selectedMaskForGen,
  setSelectedMaskForGen,
  onGenerateSora2,
  onGenerateVeo3,
  jobs,
  currentImage,
  onShowVideos
}) {
  
  const getJobCountsForMask = (maskIndex) => {
    if (!jobs || !currentImage) return { sora2: 0, veo3: 0, kling: 0 };
    
    const maskJobs = jobs.filter(job => 
      job.image_path === currentImage.path && 
      job.mask_index === maskIndex
    );
    
    return {
      sora2: maskJobs.filter(j => j.model === 'sora2').length,
      veo3: maskJobs.filter(j => j.model === 'veo3').length,
      kling: maskJobs.filter(j => j.model === 'kling').length
    };
  };
  
  const handleGenerate = () => {
    if (selectedMaskForGen === null) return;
    
    const counts = getJobCountsForMask(selectedMaskForGen);
    if (counts.sora2 > 0) {
      if (!window.confirm(`This mask already has ${counts.sora2} Sora2 job(s). Generate another one?`)) {
        return;
      }
    }
    
    onGenerateSora2();
  };

  const handleGenerateVeo3 = () => {
    if (selectedMaskForGen === null) return;
    
    const counts = getJobCountsForMask(selectedMaskForGen);
    if (counts.veo3 > 0) {
      if (!window.confirm(`This mask already has ${counts.veo3} Veo3 job(s). Generate another one?`)) {
        return;
      }
    }
    
    onGenerateVeo3();
  };
  
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
      
      <div className="tool-section">
        <h4>Generation</h4>
        
        {masks && masks.length > 0 ? (
          <>
            <div className="mask-select-list">
              {masks.map((mask, idx) => {
                const counts = getJobCountsForMask(idx);
                return (
                  <button 
                    key={idx}
                    className={`mask-select-btn ${selectedMaskForGen === idx ? 'active' : ''}`}
                    onClick={() => setSelectedMaskForGen(idx)}
                  >
                    <span>Mask {idx + 1}</span>
                    <div className="job-counters">
                      <span className={`counter ${counts.sora2 > 0 ? 'has-jobs' : 'no-jobs'}`}>
                        {counts.sora2 > 0 ? counts.sora2 : '●'}
                      </span>
                      <span className={`counter ${counts.veo3 > 0 ? 'has-jobs' : 'no-jobs'}`}>
                        {counts.veo3 > 0 ? counts.veo3 : '●'}
                      </span>
                      <span className={`counter ${counts.kling > 0 ? 'has-jobs' : 'no-jobs'}`}>
                        {counts.kling > 0 ? counts.kling : '●'}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
            
            <button 
              className="generate-btn"
              onClick={handleGenerate}
              disabled={!cropData || selectedMaskForGen === null}
            >
              Generate with Sora2
            </button>
            
            <button 
              className="generate-btn veo3-btn"
              onClick={handleGenerateVeo3}
              disabled={!cropData || selectedMaskForGen === null}
            >
              Generate with Veo3
            </button>
            
            <button 
              className="generate-btn kling-btn"
              onClick={() => alert('Kling coming soon')}
              disabled={!cropData || selectedMaskForGen === null}
            >
              Generate with Kling
            </button>
            
            <button 
              className="show-videos-btn"
              onClick={onShowVideos}
              disabled={selectedMaskForGen === null}
            >
              Show Videos
            </button>
          </>
        ) : (
          <div className="no-masks-message">
            No masks available
          </div>
        )}
      </div>
    </div>
  );
}

export default ToolsSidebar;