import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';
const MIN_DIST = 12;
const INFERENCE_DELAY = 80;
const MASK_COLORS = [
  [100, 200, 255],  // Cyan - contrasts with warm interiors
  [255, 100, 200],  // Magenta - rare in natural scenes
  [255, 230, 100],  // Yellow - visible but not harsh
  [200, 150, 255],  // Lavender - distinct from wood tones
  [100, 255, 200],  // Mint - contrasts with reds/oranges
];
const MASK_OPACITY = 0.6;

function App() {
  const [directory, setDirectory] = useState('test_data_total_1029');
  const [directoryInput, setDirectoryInput] = useState('test_data_total_1029');
  const [images, setImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [masks, setMasks] = useState([]);
  const [activeMask, setActiveMask] = useState(null);
  const [activeMaskIndex, setActiveMaskIndex] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [showOtherMasks, setShowOtherMasks] = useState(true);
  const [isDrawing, setIsDrawing] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const inferenceTimer = useRef(null);
  const lastPoint = useRef(null);
  const scale = useRef(1);
  const pointsRef = useRef([]);

  useEffect(() => {
    setWorkingDirectory();
  }, []);

  useEffect(() => {
    if (currentImage) {
      loadImage();
    }
  }, [currentImage]);

  useEffect(() => {
    if (imageLoaded && imageRef.current) {
      redrawCanvas();
    }
  }, [imageLoaded, activeMask, activeMaskIndex, showOtherMasks, masks]);

  const setWorkingDirectory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/set-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: directory })
      });
      
      if (res.ok) {
        await loadImages();
      } else {
        setImages([]);
        setCurrentImage(null);
      }
    } catch (err) {
      setImages([]);
      setCurrentImage(null);
    }
  };

  const changeDirectory = async () => {
    setDirectory(directoryInput);
    try {
      const res = await fetch(`${API_BASE}/api/set-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: directoryInput })
      });
      
      if (res.ok) {
        await loadImages();
      } else {
        setImages([]);
        setCurrentImage(null);
      }
    } catch (err) {
      setImages([]);
      setCurrentImage(null);
    }
  };

  const loadImages = async () => {
    const res = await fetch(`${API_BASE}/api/images`);
    const data = await res.json();
    setImages(data);
    if (data.length > 0 && !currentImage) {
      setCurrentImage(data[0]);
    }
  };

  const loadImage = async () => {
    setImageLoaded(false);
    const res = await fetch(`${API_BASE}/api/annotation/${currentImage.path}`);
    const data = await res.json();
    setMasks(data.masks || []);
    
    startNewMask();
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      const maxW = 800;
      const maxH = 600;
      const s = Math.min(maxW / img.width, maxH / img.height);
      scale.current = s;
      
      canvas.width = img.width * s;
      canvas.height = img.height * s;
      
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      imageRef.current = img;
      setImageLoaded(true);
    };
    
    img.src = `${API_BASE}/api/image/${currentImage.path}`;
  };

  const redrawCanvas = () => {
    if (!imageRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 1. Clear and draw base image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    
    // 2. Draw other masks if toggle is on
    if (showOtherMasks) {
      masks.forEach((maskInfo, idx) => {
        if (idx === activeMaskIndex) return; // Skip active mask
        if (maskInfo.mask_data) {
          const color = MASK_COLORS[idx % MASK_COLORS.length];
          drawMaskOverlay(maskInfo.mask_data, color, MASK_OPACITY);
        }
      });
    }
    
    // 3. Always draw active mask on top in green
    if (activeMask) {
      drawMaskOverlay(activeMask, [0, 255, 0], MASK_OPACITY);
    }
  };

  const drawMaskOverlay = (mask, rgb, opacity) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext('2d');
    
    const imageData = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
    const data = imageData.data;
    
    for (let y = 0; y < maskCanvas.height; y++) {
      for (let x = 0; x < maskCanvas.width; x++) {
        const origY = Math.floor(y / scale.current);
        const origX = Math.floor(x / scale.current);
        
        if (origY < mask.length && origX < mask[0].length && mask[origY][origX]) {
          const idx = (y * maskCanvas.width + x) * 4;
          data[idx] = rgb[0];
          data[idx + 1] = rgb[1];
          data[idx + 2] = rgb[2];
          data[idx + 3] = 255;
        }
      }
    }
    
    maskCtx.putImageData(imageData, 0, 0);
    ctx.globalAlpha = opacity;
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
  };

  const startNewMask = () => {
    setActiveMaskIndex(null);
    setActiveMask(null);
    setPrompt('');
    pointsRef.current = [];
    lastPoint.current = null;
  };

  const editMask = (index) => {
    setActiveMaskIndex(index);
    setActiveMask(masks[index].mask_data);
    setPrompt(masks[index].prompt);
    pointsRef.current = [];
    lastPoint.current = null;
  };

  const deleteMask = async (index) => {
    await fetch(`${API_BASE}/api/mask`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: currentImage.path,
        mask_index: index
      })
    });
    
    await loadImages();
    await loadImage();
  };

  const saveMask = async () => {
    if (!activeMask) return;
    
    if (activeMaskIndex === null) {
      await fetch(`${API_BASE}/api/mask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: currentImage.path,
          mask: activeMask,
          prompt: prompt
        })
      });
    } else {
      await fetch(`${API_BASE}/api/mask`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: currentImage.path,
          mask_index: activeMaskIndex,
          mask: activeMask,
          prompt: prompt
        })
      });
    }
    
    await loadImages();
    await loadImage();
  };

  const getCanvasCoords = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale.current;
    const y = (e.clientY - rect.top) / scale.current;
    return [x, y];
  };

  const shouldSample = (x, y) => {
    if (!lastPoint.current) return true;
    const [lx, ly] = lastPoint.current;
    const dist = Math.sqrt((x - lx) ** 2 + (y - ly) ** 2);
    return dist >= MIN_DIST;
  };

  const scheduleInference = () => {
    if (inferenceTimer.current) {
      clearTimeout(inferenceTimer.current);
    }
    inferenceTimer.current = setTimeout(() => {
      runInference();
    }, INFERENCE_DELAY);
  };

  const runInference = async () => {
    if (pointsRef.current.length === 0 || !currentImage) return;
    
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: currentImage.path,
        points: pointsRef.current
      })
    });
    
    const data = await res.json();
    setActiveMask(data.mask);
  };

  const handleMouseDown = (e) => {
    if (!currentImage) return;
    
    const [x, y] = getCanvasCoords(e);
    setIsDrawing(true);
    pointsRef.current = [[x, y]];
    lastPoint.current = [x, y];
    scheduleInference();
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    
    const [x, y] = getCanvasCoords(e);
    if (shouldSample(x, y)) {
      pointsRef.current.push([x, y]);
      lastPoint.current = [x, y];
      scheduleInference();
    }
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    
    if (inferenceTimer.current) {
      clearTimeout(inferenceTimer.current);
      inferenceTimer.current = null;
    }
    
    if (pointsRef.current.length > 0) {
      runInference();
    }
  };

  const goToPrev = () => {
    const idx = images.findIndex(img => img.path === currentImage.path);
    if (idx > 0) {
      setCurrentImage(images[idx - 1]);
    }
  };

  const goToNext = () => {
    const idx = images.findIndex(img => img.path === currentImage.path);
    if (idx < images.length - 1) {
      setCurrentImage(images[idx + 1]);
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <h2>Images</h2>
        <div className="directory-changer">
          <input
            type="text"
            value={directoryInput}
            onChange={(e) => setDirectoryInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && changeDirectory()}
          />
          <button onClick={changeDirectory}>Load</button>
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
                onClick={() => setCurrentImage(img)}
              >
                {img.filename}
                {img.mask_count > 0 && <span className="count">({img.mask_count})</span>}
              </div>
            ))}
          </div>
        )}
      </div>
      
      <div className="mask-sidebar">
        <h3>Masks</h3>
        {currentImage && (
          <>
            <button onClick={startNewMask} className="new-mask-btn">+ New Mask</button>
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
                      <button onClick={() => editMask(idx)}>Edit</button>
                      <button onClick={() => deleteMask(idx)} className="delete-btn">Del</button>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
      
      <div className="main">
        {currentImage ? (
          <>
            {masks.length > 0 && (
              <div className="canvas-header">
                <label>
                  <input
                    type="checkbox"
                    checked={showOtherMasks}
                    onChange={(e) => setShowOtherMasks(e.target.checked)}
                  />
                  Show other masks
                </label>
              </div>
            )}
            <div className="canvas-container">
              <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            </div>
            
            <div className="controls">
              <button onClick={goToPrev}>← Prev</button>
              <input
                type="text"
                placeholder="Enter prompt..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <button onClick={saveMask} disabled={!activeMask}>Save</button>
              <button onClick={goToNext}>Next →</button>
            </div>
          </>
        ) : (
          <div className="no-image-message">
            {images.length === 0 ? 'No images in directory' : 'Select an image'}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;