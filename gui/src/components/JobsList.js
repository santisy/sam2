import React from 'react';
import './JobsList.css';

function JobsList({ jobs, isCollapsed, onToggle, apiBase }) {
  const formatTime = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleTimeString();
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'completed': return '#4ade80';
      case 'downloading': return '#a78bfa';
      case 'failed': return '#ef4444';
      case 'in_progress': return '#fbbf24';
      case 'queued': return '#60a5fa';
      default: return '#888';
    }
  };

  const getStatusLabel = (status) => {
    if (status === 'in_progress') return 'generating';
    return status;
  };

  return (
    <div className={`jobs-panel ${isCollapsed ? 'collapsed' : 'expanded'}`}>
      <div className="jobs-header" onClick={onToggle}>
        <div className="jobs-title">
          <span>Generation Jobs ({jobs.length})</span>
          <span className="jobs-toggle">{isCollapsed ? '▲' : '▼'}</span>
        </div>
      </div>
      
      {!isCollapsed && (
        <div className="jobs-content">
          {jobs.length === 0 ? (
            <div className="no-jobs">No generation jobs yet</div>
          ) : (
            <div className="jobs-list">
              {jobs.map((job) => (
                <div key={job.job_id} className="job-item">
                  <div className="job-header-row">
                    <div className="job-status-group">
                      <span className={`job-model ${job.model}`}>{job.model}</span>
                      <span 
                        className="job-status"
                        style={{ color: getStatusColor(job.status) }}
                      >
                        ● {getStatusLabel(job.status)}
                      </span>
                    </div>
                    <span className="job-time">{formatTime(job.created_at)}</span>
                  </div>
                  
                  <div className="job-info">
                    <div className="job-image-mask">
                      {job.image_path} (Mask {job.mask_index + 1})
                    </div>
                    <div className="job-prompt">{job.prompt}</div>
                  </div>
                  
                  {(job.status === 'in_progress' || job.status === 'downloading') && (
                    <div className="job-progress">
                      <div className="progress-bar">
                        <div 
                          className="progress-fill animating"
                          style={{ 
                            background: job.status === 'downloading' ? '#a78bfa' : '#fbbf24'
                          }}
                        />
                      </div>
                      <span className="progress-text">
                        {job.status === 'downloading' ? 'Downloading...' : 'Generating...'}
                      </span>
                    </div>
                  )}
                  
                  {job.video_exists && (
                    <div className="job-video">
                      ✓ Video ready: {job.video_filename}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default JobsList;