import React from 'react';
import './Sidebar.css';

function Sidebar({ 
  directory, 
  directoryInput, 
  setDirectoryInput, 
  images, 
  currentImage, 
  onChangeDirectory, 
  onSelectImage,
  jobs
}) {
  
  // Count masks with at least one video for each model for a given image
  const getVideoCountsForImage = (imagePath, maskCount) => {
    if (!jobs || maskCount === 0) return { sora2: 0, veo3: 0, kling: 0, sora2Complete: false, veo3Complete: false, klingComplete: false };
    
    // Get all jobs for this image that have videos
    const imageJobs = jobs.filter(job => 
      job.image_path === imagePath && 
      job.video_exists
    );
    
    // Count unique masks that have at least one video for each model
    const masksWithSora2 = new Set(imageJobs.filter(j => j.model === 'sora2').map(j => j.mask_index)).size;
    const masksWithVeo3 = new Set(imageJobs.filter(j => j.model === 'veo3').map(j => j.mask_index)).size;
    const masksWithKling = new Set(imageJobs.filter(j => j.model === 'kling').map(j => j.mask_index)).size;
    
    return {
      sora2: masksWithSora2,
      veo3: masksWithVeo3,
      kling: masksWithKling,
      sora2Complete: masksWithSora2 === maskCount, // true if ALL masks have sora2 videos
      veo3Complete: masksWithVeo3 === maskCount,   // true if ALL masks have veo3 videos
      klingComplete: masksWithKling === maskCount  // true if ALL masks have kling videos
    };
  };
  
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
          {images.map(img => {
            const videoCounts = getVideoCountsForImage(img.path, img.mask_count);
            return (
              <div
                key={img.path}
                className={`image-item ${currentImage?.path === img.path ? 'active' : ''}`}
                onClick={() => onSelectImage(img)}
              >
                <span className="image-name">{img.filename}</span>
                <div className="image-counts">
                  {img.mask_count > 0 && <span className="count masks">{img.mask_count}</span>}
                  {img.mask_count > 0 && (
                    <div className="video-counts">
                      <span className={`count video sora2 ${videoCounts.sora2Complete ? 'complete' : 'incomplete'}`}>
                        {videoCounts.sora2}
                      </span>
                      <span className={`count video veo3 ${videoCounts.veo3Complete ? 'complete' : 'incomplete'}`}>
                        {videoCounts.veo3}
                      </span>
                      <span className={`count video kling ${videoCounts.klingComplete ? 'complete' : 'incomplete'}`}>
                        {videoCounts.kling}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Sidebar;