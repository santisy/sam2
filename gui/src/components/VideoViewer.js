import React, { useState } from 'react';
import './VideoViewer.css';

function VideoViewer({ 
  isOpen, 
  onClose, 
  maskIndex, 
  imagePrefix,
  apiBase,
  jobs,
  currentImage,
  masks
}) {
  const [sora2Index, setSora2Index] = useState(0);
  const [veo3Index, setVeo3Index] = useState(0);
  const [klingIndex, setKlingIndex] = useState(0);
  
  if (!isOpen) return null;
  
  console.log('VideoViewer debug:', {
    imagePrefix,
    maskIndex,
    currentImage: currentImage?.path
  });
  
  // Get jobs for this mask
  const maskJobs = jobs.filter(job => 
    job.image_path === currentImage?.path && 
    job.mask_index === maskIndex &&
    job.video_exists
  );
  
  const sora2Videos = maskJobs.filter(j => j.model === 'sora2');
  const veo3Videos = maskJobs.filter(j => j.model === 'veo3');
  const klingVideos = maskJobs.filter(j => j.model === 'kling');
  
  // Our video filename format: {prefix}_mask_{num}.mp4
  const ourVideoFilename = imagePrefix ? `${imagePrefix}_mask_${(maskIndex + 1).toString().padStart(3, '0')}.mp4` : null;
  
  console.log('Ours video filename:', ourVideoFilename);
  
  // Get prompt for this mask
  const currentPrompt = (masks && masks[maskIndex]) ? masks[maskIndex].prompt : '';
  
  const renderVideoSlot = (title, videos, currentIndex, setIndex, ourVideo = false) => {
    const hasVideo = ourVideo ? ourVideoFilename : (videos && videos.length > 0);
    const videoSrc = ourVideo 
      ? (ourVideoFilename ? `${apiBase}/api/video/ours/${ourVideoFilename}` : null)
      : (videos && videos.length > 0 ? `${apiBase}/api/video/${videos[currentIndex].model}/${videos[currentIndex].video_filename}` : null);
    
    return (
      <div className="video-slot">
        <div className="video-slot-header">
          <h4>{title}</h4>
          {!ourVideo && videos && videos.length > 1 && (
            <div className="video-nav">
              <button 
                className="video-nav-btn"
                onClick={() => setIndex((currentIndex - 1 + videos.length) % videos.length)}
              >
                ←
              </button>
              <span className="video-counter">{currentIndex + 1}/{videos.length}</span>
              <button 
                className="video-nav-btn"
                onClick={() => setIndex((currentIndex + 1) % videos.length)}
              >
                →
              </button>
            </div>
          )}
        </div>
        <div className="video-container">
          {hasVideo ? (
            <video 
              key={videoSrc}
              src={videoSrc} 
              controls 
              loop
              className="video-player"
            />
          ) : (
            <div className="video-empty">No video</div>
          )}
        </div>
      </div>
    );
  };
  
  return (
    <div className="video-viewer-overlay" onClick={onClose}>
      <div className="video-viewer-panel" onClick={(e) => e.stopPropagation()}>
        <button className="video-viewer-close" onClick={onClose}>✕</button>
        
        {currentPrompt && (
          <div className="video-viewer-prompt">
            <strong>Prompt:</strong> {currentPrompt}
          </div>
        )}
        
        <div className="video-grid">
          {renderVideoSlot('Ours', null, 0, () => {}, true)}
          {renderVideoSlot('Sora2', sora2Videos, sora2Index, setSora2Index)}
          {renderVideoSlot('Veo3', veo3Videos, veo3Index, setVeo3Index)}
          {renderVideoSlot('Kling', klingVideos, klingIndex, setKlingIndex)}
        </div>
      </div>
    </div>
  );
}

export default VideoViewer;