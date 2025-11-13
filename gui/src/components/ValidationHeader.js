import React, { useRef, useEffect, useState } from 'react';
import './ValidationHeader.css';

const MASK_COLORS = [
  [100, 200, 255],
  [255, 100, 200],
  [255, 230, 100],
  [200, 150, 255],
  [100, 255, 200],
];

function ValidationHeader({ apiBase, currentImage, masks, activeMaskIndex, onSelectMask, mask, selectedModel, onSelectModel, validationStats }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const scaleRef = useRef(1);
  const [maskStats, setMaskStats] = useState({});

  useEffect(() => {
    if (currentImage) {
      loadImage();
    }
  }, [currentImage, mask]);

  // Compute mask stats from global validation stats
  useEffect(() => {
    const computeMaskStats = async () => {
      if (!currentImage || !masks || masks.length === 0) {
        setMaskStats({});
        return;
      }

      const models = ['ours', 'rgb', 'wan', 'sora2', 'veo3', 'kling'];
      const stats = {};

      for (let maskIdx = 0; maskIdx < masks.length; maskIdx++) {
        stats[maskIdx] = { byModel: {}, byMetric: {} };

        // Load validation for each model
        for (const model of models) {
          const res = await fetch(
            `${apiBase}/api/validation?image_path=${encodeURIComponent(currentImage.path)}&mask_index=${maskIdx}&model=${model}`
          );
          const data = await res.json();

          stats[maskIdx].byModel[model] = {
            localization: data.localization !== null,
            articulation_type: data.articulation_type !== null,
            sequence_plausible: data.sequence_plausible !== null,
            perspective: false // Not implemented yet
          };
        }

        // Calculate per-metric completion counts
        const metrics = ['localization', 'articulation_type', 'sequence_plausible', 'perspective'];
        for (const metric of metrics) {
          let count = 0;
          for (const model of models) {
            if (stats[maskIdx].byModel[model][metric]) {
              count++;
            }
          }
          stats[maskIdx].byMetric[metric] = count;
        }
      }

      setMaskStats(stats);
    };

    computeMaskStats();
  }, [currentImage, masks, apiBase, validationStats]); // Added validationStats as dependency

  const loadImage = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Get parent height
      const parentHeight = canvas.parentElement.clientHeight;
      const s = parentHeight / img.height;
      scaleRef.current = s;
      
      canvas.width = img.width * s;
      canvas.height = img.height * s;
      
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      imageRef.current = img;
      
      if (mask) {
        drawMask();
      }
    };
    
    img.src = `${apiBase}/api/image/${currentImage.path}`;
  };

  const drawMask = () => {
    if (!imageRef.current || !canvasRef.current || !mask || !mask.mask_data) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext('2d');
    
    const imageData = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
    const data = imageData.data;
    
    const maskData = mask.mask_data;
    
    for (let y = 0; y < maskCanvas.height; y++) {
      for (let x = 0; x < maskCanvas.width; x++) {
        const origY = Math.floor(y / scaleRef.current);
        const origX = Math.floor(x / scaleRef.current);
        
        if (origY < maskData.length && origX < maskData[0].length && maskData[origY][origX]) {
          const idx = (y * maskCanvas.width + x) * 4;
          data[idx] = 0;
          data[idx + 1] = 255;
          data[idx + 2] = 0;
          data[idx + 3] = 255;
        }
      }
    }
    
    maskCtx.putImageData(imageData, 0, 0);
    ctx.globalAlpha = 0.6;
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
  };

  return (
    <div className="validation-header">
      <div className="left-panel">
        <div className="mask-selector-section">
          <div className="section-title">Select Mask:</div>
          <div className="mask-list">
            {masks.map((m, idx) => {
              const color = MASK_COLORS[idx % MASK_COLORS.length];
              const stats = maskStats[idx];
              
              return (
                <div
                  key={idx}
                  className={`mask-item ${activeMaskIndex === idx ? 'selected' : ''}`}
                  onClick={() => onSelectMask(idx)}
                >
                  <div className="mask-header">
                    <div className="mask-identity">
                      <div 
                        className="mask-color-swatch" 
                        style={{backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`}}
                      />
                      <div className="mask-name">Mask {idx + 1}</div>
                    </div>
                    {stats && (
                      <div className="mask-stats">
                        <div className="stats-row">
                          <span className="stat-item" title="Localization">{stats.byMetric.localization || 0}</span>
                          <span className="stat-item" title="Articulation Type">{stats.byMetric.articulation_type || 0}</span>
                          <span className="stat-item" title="Sequence">{stats.byMetric.sequence_plausible || 0}</span>
                          <span className="stat-item" title="Perspective">{stats.byMetric.perspective || 0}</span>
                        </div>
                        <div className="stats-row">
                          <span className="stat-item" title="Ours">{Object.values(stats.byModel.ours || {}).filter(Boolean).length}</span>
                          <span className="stat-item" title="RGB">{Object.values(stats.byModel.rgb || {}).filter(Boolean).length}</span>
                          <span className="stat-item" title="WAN">{Object.values(stats.byModel.wan || {}).filter(Boolean).length}</span>
                          <span className="stat-item" title="Sora2">{Object.values(stats.byModel.sora2 || {}).filter(Boolean).length}</span>
                          <span className="stat-item" title="Veo3">{Object.values(stats.byModel.veo3 || {}).filter(Boolean).length}</span>
                          <span className="stat-item" title="Kling">{Object.values(stats.byModel.kling || {}).filter(Boolean).length}</span>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="mask-prompt">{m.prompt || '(no prompt)'}</div>
                </div>
              );
            })}
          </div>
        </div>
        
        <div className="model-selector-section">
          <div className="section-title">Select Model:</div>
          <div className="model-selector">
            <button 
              className={selectedModel === 'ours' ? 'active' : ''}
              onClick={() => onSelectModel('ours')}
            >
              Ours
            </button>
            <button 
              className={selectedModel === 'rgb' ? 'active' : ''}
              onClick={() => onSelectModel('rgb')}
            >
              RGB
            </button>
            <button 
              className={selectedModel === 'wan' ? 'active' : ''}
              onClick={() => onSelectModel('wan')}
            >
              WAN
            </button>
            <button 
              className={selectedModel === 'sora2' ? 'active' : ''}
              onClick={() => onSelectModel('sora2')}
            >
              Sora2
            </button>
            <button 
              className={selectedModel === 'veo3' ? 'active' : ''}
              onClick={() => onSelectModel('veo3')}
            >
              Veo3
            </button>
            <button 
              className={selectedModel === 'kling' ? 'active' : ''}
              onClick={() => onSelectModel('kling')}
            >
              Kling
            </button>
          </div>
        </div>
        
        <div className="prompt-section">
          <div className="section-title">Prompt:</div>
          <div className="prompt-text">{mask ? mask.prompt : 'Select a mask'}</div>
        </div>
      </div>
      
      <div className="right-panel">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

export default ValidationHeader;