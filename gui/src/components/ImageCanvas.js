import React, { useRef, useEffect, useState } from 'react';
import './ImageCanvas.css';

const MASK_COLORS = [
  [100, 200, 255],
  [255, 100, 200],
  [255, 230, 100],
  [200, 150, 255],
  [100, 255, 200],
];
const MASK_OPACITY = 0.6;

function ImageCanvas({ 
  apiBase,
  currentImage, 
  masks, 
  activeMask, 
  activeMaskIndex,
  showOtherMasks,
  imageLoaded,
  setImageLoaded,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  maskMode,
  cropMode,
  cropOrientation,
  cropSize,
  cropData,
  onCropChange
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const scaleRef = useRef(1);
  const [isDraggingCrop, setIsDraggingCrop] = useState(false);
  const [cropPosition, setCropPosition] = useState({ x: 0, y: 0 });
  const dragStartRef = useRef(null);

  // Load image when currentImage changes
  useEffect(() => {
    if (currentImage) {
      loadImage();
    }
  }, [currentImage]);

  // Redraw canvas when relevant state changes
  useEffect(() => {
    if (imageLoaded && imageRef.current) {
      redrawCanvas();
    }
  }, [imageLoaded, activeMask, activeMaskIndex, showOtherMasks, masks, cropMode, cropPosition, cropSize, cropOrientation, cropData]);

  // Recalculate crop position when size or orientation changes
  useEffect(() => {
    if (imageLoaded && canvasRef.current) {
      const canvas = canvasRef.current;
      const cropDims = calculateCropDimensions(canvas.width, canvas.height);
      
      // Center the crop when orientation/size changes
      const newX = (canvas.width - cropDims.width) / 2;
      const newY = (canvas.height - cropDims.height) / 2;
      
      setCropPosition({ x: newX, y: newY });
    }
  }, [cropSize, cropOrientation, imageLoaded]);

  // Load crop position from saved data
  useEffect(() => {
    if (imageLoaded && cropData && canvasRef.current) {
      setCropPosition({ 
        x: cropData.x * scaleRef.current, 
        y: cropData.y * scaleRef.current 
      });
    }
  }, [cropData, imageLoaded]);

  const loadImage = async () => {
    setImageLoaded(false);
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      const maxW = 800;
      const maxH = 600;
      const s = Math.min(maxW / img.width, maxH / img.height);
      scaleRef.current = s;
      
      canvas.width = img.width * s;
      canvas.height = img.height * s;
      
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      imageRef.current = img;
      
      // Initialize crop position - will be updated by effect if cropData exists
      const cropDims = calculateCropDimensions(canvas.width, canvas.height);
      setCropPosition({ 
        x: (canvas.width - cropDims.width) / 2, 
        y: (canvas.height - cropDims.height) / 2 
      });
      
      setImageLoaded(true);
    };
    
    img.src = `${apiBase}/api/image/${currentImage.path}`;
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
        if (idx === activeMaskIndex) return;
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
    
    // 4. Draw crop overlay
    if (cropMode || cropData) {
      drawCropOverlay();
    }
  };

  const calculateCropDimensions = (canvasWidth, canvasHeight) => {
    const sizeMultiplier = cropSize / 100;
    let width, height;
    
    if (cropOrientation === 'landscape') {
      // 16:9 landscape
      const maxWidth = canvasWidth * sizeMultiplier;
      const maxHeight = canvasHeight * sizeMultiplier;
      
      // Try fitting by width
      width = maxWidth;
      height = width * (9/16);
      
      // If too tall, fit by height instead
      if (height > maxHeight) {
        height = maxHeight;
        width = height * (16/9);
      }
    } else {
      // 9:16 portrait
      const maxWidth = canvasWidth * sizeMultiplier;
      const maxHeight = canvasHeight * sizeMultiplier;
      
      // Try fitting by height
      height = maxHeight;
      width = height * (9/16);
      
      // If too wide, fit by width instead
      if (width > maxWidth) {
        width = maxWidth;
        height = width * (16/9);
      }
    }
    
    return { width, height };
  };

  const drawCropOverlay = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const cropDims = calculateCropDimensions(canvas.width, canvas.height);
    
    // Draw 4 rectangles to darken everything outside crop area
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    
    // Top
    ctx.fillRect(0, 0, canvas.width, cropPosition.y);
    
    // Bottom
    ctx.fillRect(0, cropPosition.y + cropDims.height, canvas.width, canvas.height - (cropPosition.y + cropDims.height));
    
    // Left
    ctx.fillRect(0, cropPosition.y, cropPosition.x, cropDims.height);
    
    // Right
    ctx.fillRect(cropPosition.x + cropDims.width, cropPosition.y, canvas.width - (cropPosition.x + cropDims.width), cropDims.height);
    
    // Draw crop border
    ctx.strokeStyle = cropMode ? '#16a34a' : '#2563eb';
    ctx.lineWidth = 2;
    ctx.strokeRect(cropPosition.x, cropPosition.y, cropDims.width, cropDims.height);
    
    // Draw corner handles if in crop mode
    if (cropMode) {
      const handleSize = 8;
      ctx.fillStyle = '#16a34a';
      ctx.fillRect(cropPosition.x - handleSize/2, cropPosition.y - handleSize/2, handleSize, handleSize);
      ctx.fillRect(cropPosition.x + cropDims.width - handleSize/2, cropPosition.y - handleSize/2, handleSize, handleSize);
      ctx.fillRect(cropPosition.x - handleSize/2, cropPosition.y + cropDims.height - handleSize/2, handleSize, handleSize);
      ctx.fillRect(cropPosition.x + cropDims.width - handleSize/2, cropPosition.y + cropDims.height - handleSize/2, handleSize, handleSize);
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
        const origY = Math.floor(y / scaleRef.current);
        const origX = Math.floor(x / scaleRef.current);
        
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

  const handleMouseDown = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (cropMode) {
      const cropDims = calculateCropDimensions(canvasRef.current.width, canvasRef.current.height);
      // Check if click is inside crop area
      if (x >= cropPosition.x && x <= cropPosition.x + cropDims.width &&
          y >= cropPosition.y && y <= cropPosition.y + cropDims.height) {
        setIsDraggingCrop(true);
        dragStartRef.current = { x: x - cropPosition.x, y: y - cropPosition.y };
      }
    } else if (maskMode) {
      const realX = x / scaleRef.current;
      const realY = y / scaleRef.current;
      onMouseDown(realX, realY);
    }
  };

  const handleMouseMove = (e) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (isDraggingCrop && cropMode) {
      const cropDims = calculateCropDimensions(canvasRef.current.width, canvasRef.current.height);
      let newX = x - dragStartRef.current.x;
      let newY = y - dragStartRef.current.y;
      
      // Constrain to canvas bounds
      newX = Math.max(0, Math.min(newX, canvasRef.current.width - cropDims.width));
      newY = Math.max(0, Math.min(newY, canvasRef.current.height - cropDims.height));
      
      setCropPosition({ x: newX, y: newY });
    } else if (maskMode) {
      const realX = x / scaleRef.current;
      const realY = y / scaleRef.current;
      onMouseMove(realX, realY);
    }
  };

  const handleMouseUp = () => {
    if (isDraggingCrop && cropMode) {
      setIsDraggingCrop(false);
      // Save crop position
      const cropDims = calculateCropDimensions(canvasRef.current.width, canvasRef.current.height);
      onCropChange({
        x: cropPosition.x / scaleRef.current,
        y: cropPosition.y / scaleRef.current,
        width: cropDims.width / scaleRef.current,
        height: cropDims.height / scaleRef.current,
        orientation: cropOrientation,
        size: cropSize
      });
    } else if (maskMode) {
      onMouseUp();
    }
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: cropMode ? (isDraggingCrop ? 'grabbing' : 'grab') : (maskMode ? 'crosshair' : 'default') }}
      />
    </div>
  );
}

export default ImageCanvas;