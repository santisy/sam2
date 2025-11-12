import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import MaskSidebar from './components/MaskSidebar';
import ImageCanvas from './components/ImageCanvas';
import AnnotationView from './components/AnnotationView';
import ToolsSidebar from './components/ToolsSidebar';
import JobsList from './components/JobsList';
import VideoViewer from './components/VideoViewer';

const PORT = 5876;
const API_BASE = `http://localhost:${PORT}`;
const MIN_DIST = 12;
const INFERENCE_DELAY = 80;
const JOB_POLL_INTERVAL = 5000; // Poll jobs every 5 seconds

function App() {
  // Directory & Images
  const [directory, setDirectory] = useState('test_data_total_1029');
  const [directoryInput, setDirectoryInput] = useState('test_data_total_1029');
  const [images, setImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  
  // Masks & Annotation
  const [masks, setMasks] = useState([]);
  const [activeMask, setActiveMask] = useState(null);
  const [activeMaskIndex, setActiveMaskIndex] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [showOtherMasks, setShowOtherMasks] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);
  
  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false);
  const inferenceTimer = useRef(null);
  const lastPoint = useRef(null);
  const pointsRef = useRef([]);
  
  // Prompt builder state
  const [pbIndicator, setPbIndicator] = useState('');
  const [pbPart, setPbPart] = useState('');
  const [pbDescription, setPbDescription] = useState('');
  const [pbObject, setPbObject] = useState('');
  const [pbAction, setPbAction] = useState('');
  const [pbDirection, setPbDirection] = useState('');
  const [pbContext, setPbContext] = useState('');
  const [lastClicked, setLastClicked] = useState('');
  
  // Crop state
  const [cropMode, setCropMode] = useState(false);
  const [cropOrientation, setCropOrientation] = useState('landscape');
  const [cropSize, setCropSize] = useState(100);
  const [cropData, setCropData] = useState(null);
  const [currentCrop, setCurrentCrop] = useState(null); // Current unsaved crop
  
  // Mask editing mode
  const [maskMode, setMaskMode] = useState(false);
  
  // Generation state
  const [selectedMaskForGen, setSelectedMaskForGen] = useState(null);
  
  // Jobs panel state
  const [jobs, setJobs] = useState([]);
  const [jobsPanelCollapsed, setJobsPanelCollapsed] = useState(true);
  const jobPollTimer = useRef(null);
  
  // Video viewer state
  const [showVideoViewer, setShowVideoViewer] = useState(false);

  useEffect(() => {
    setWorkingDirectory();
    loadJobs(); // Initial load
    startJobPolling(); // Start polling
    
    return () => {
      if (jobPollTimer.current) {
        clearInterval(jobPollTimer.current);
      }
    };
  }, []);

  const loadJobs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/generations`);
      const data = await res.json();
      setJobs(data);
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  };

  const startJobPolling = () => {
    if (jobPollTimer.current) {
      clearInterval(jobPollTimer.current);
    }
    jobPollTimer.current = setInterval(loadJobs, JOB_POLL_INTERVAL);
  };

  useEffect(() => {
    if (currentImage) {
      loadAnnotations();
      // Clear drawing state when switching images
      setActiveMask(null);
      setActiveMaskIndex(null);
      setMaskMode(false);
      setSelectedMaskForGen(null);
      pointsRef.current = [];
      lastPoint.current = null;
    }
  }, [currentImage]);

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

  const loadAnnotations = async () => {
    const res = await fetch(`${API_BASE}/api/annotation/${currentImage.path}`);
    const data = await res.json();
    setMasks(data.masks || []);
    startNewMask();
    
    // Load crop data from same json
    if (data.crop) {
      setCropData(data.crop);
      setCropOrientation(data.crop.orientation || 'landscape');
      setCropSize(data.crop.size || 100);
    } else {
      setCropData(null);
    }
  };

  const startNewMask = () => {
    setActiveMaskIndex(null);
    setActiveMask(null);
    setPrompt('');
    pointsRef.current = [];
    lastPoint.current = null;
    setMaskMode(true);
    setCropMode(false);
  };

  const editMask = (index) => {
    setActiveMaskIndex(index);
    setActiveMask(masks[index].mask_data);
    setPrompt(masks[index].prompt);
    pointsRef.current = [];
    lastPoint.current = null;
    setMaskMode(true);
    setCropMode(false);
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
    await loadAnnotations();
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
    
    // Visual feedback
    const savedIndex = activeMaskIndex;
    await loadImages();
    await loadAnnotations();
    
    // Briefly highlight the saved mask
    if (savedIndex !== null) {
      setTimeout(() => {
        const maskItem = document.querySelectorAll('.mask-item')[savedIndex];
        if (maskItem) {
          maskItem.classList.add('saved-flash');
          setTimeout(() => maskItem.classList.remove('saved-flash'), 1000);
        }
      }, 100);
    }
  };

  const saveCrop = async () => {
    if (!currentImage || !currentCrop) return;
    
    await fetch(`${API_BASE}/api/crop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: currentImage.path,
        crop: currentCrop
      })
    });
    
    // Reload from backend to ensure consistency
    const res = await fetch(`${API_BASE}/api/annotation/${currentImage.path}`);
    const data = await res.json();
    if (data.crop) {
      setCropData(data.crop);
      setCropOrientation(data.crop.orientation || 'landscape');
      setCropSize(data.crop.size || 100);
    }
    
    setCurrentCrop(null);
  };

  const deleteCrop = async () => {
    if (!currentImage) return;
    
    await fetch(`${API_BASE}/api/crop`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_path: currentImage.path
      })
    });
    
    setCropData(null);
    setCurrentCrop(null);
  };

  const generateSora2 = async () => {
    if (!currentImage || selectedMaskForGen === null) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/generate/sora2`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: currentImage.path,
          mask_index: selectedMaskForGen
        })
      });
      
      if (res.ok) {
        const data = await res.json();
        console.log('Generation started:', data.job_id);
        loadJobs(); // Refresh job list
      }
    } catch (err) {
      console.error('Generation failed:', err);
    }
  };

  const generateVeo3 = async () => {
    if (!currentImage || selectedMaskForGen === null) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/generate/veo3`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: currentImage.path,
          mask_index: selectedMaskForGen
        })
      });
      
      if (res.ok) {
        const data = await res.json();
        console.log('Generation started:', data.job_id);
        loadJobs(); // Refresh job list
      }
    } catch (err) {
      console.error('Generation failed:', err);
    }
  };
  
  const handleShowVideos = () => {
    if (selectedMaskForGen === null) return;
    setShowVideoViewer(true);
  };
  
  const getImagePrefix = () => {
    // Get image stem for "ours" video filename from mask filename
    // Mask filename pattern: {imagestem}-{hash}-mask-{num}.png
    // E.g., "28b854a5-4b6116-mask-002.png" -> we want "28b854a5"
    // Ours video: {imagestem}_mask_{num}.mp4
    if (masks.length > 0 && masks[0].mask_filename) {
      console.log('Extracting from mask_filename:', masks[0].mask_filename);
      // Extract everything before first hyphen
      const match = masks[0].mask_filename.match(/^([^-]+)-/);
      const prefix = match ? match[1] : null;
      console.log('Extracted prefix:', prefix);
      return prefix;
    }
    console.log('No masks or mask_filename');
    return null;
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

  const handleMouseDown = (x, y) => {
    if (!currentImage || !maskMode) return;
    
    setIsDrawing(true);
    pointsRef.current = [[x, y]];
    lastPoint.current = [x, y];
    scheduleInference();
  };

  const handleMouseMove = (x, y) => {
    if (!isDrawing || !maskMode) return;
    
    if (shouldSample(x, y)) {
      pointsRef.current.push([x, y]);
      lastPoint.current = [x, y];
      scheduleInference();
    }
  };

  const handleMouseUp = () => {
    if (!isDrawing || !maskMode) return;
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

  const generatePrompt = () => {
    const parts = [];
    parts.push('The');
    if (pbIndicator) {
      parts.push(pbIndicator);
    }
    parts.push(pbPart);
    parts.push('of');
    if (pbDescription) {
      parts.push(pbDescription);
    }
    parts.push(pbObject);
    parts.push('is');
    parts.push(pbAction);
    parts.push(pbDirection);
    parts.push('by itself');
    if (pbContext) {
      parts.push(pbContext);
    }
    const generated = parts.filter(p => p).join(' ') + '. Camera is fixed.';
    setPrompt(generated);
  };

  return (
    <div className="app">
      <Sidebar 
        directory={directory}
        directoryInput={directoryInput}
        setDirectoryInput={setDirectoryInput}
        images={images}
        currentImage={currentImage}
        onChangeDirectory={changeDirectory}
        onSelectImage={setCurrentImage}
      />
      
      <MaskSidebar 
        currentImage={currentImage}
        masks={masks}
        activeMask={activeMask}
        activeMaskIndex={activeMaskIndex}
        prompt={prompt}
        onStartNewMask={startNewMask}
        onEditMask={editMask}
        onDeleteMask={deleteMask}
      />
      
      <div className="main">
        <ImageCanvas 
          apiBase={API_BASE}
          currentImage={currentImage}
          masks={masks}
          activeMask={activeMask}
          activeMaskIndex={activeMaskIndex}
          showOtherMasks={showOtherMasks}
          imageLoaded={imageLoaded}
          setImageLoaded={setImageLoaded}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          maskMode={maskMode}
          cropMode={cropMode}
          cropOrientation={cropOrientation}
          cropSize={cropSize}
          cropData={cropData}
          onCropChange={setCurrentCrop}
        />
        
        <AnnotationView 
          currentImage={currentImage}
          images={images}
          masks={masks}
          showOtherMasks={showOtherMasks}
          setShowOtherMasks={setShowOtherMasks}
          prompt={prompt}
          setPrompt={setPrompt}
          activeMask={activeMask}
          pbIndicator={pbIndicator}
          setPbIndicator={setPbIndicator}
          pbPart={pbPart}
          setPbPart={setPbPart}
          pbDescription={pbDescription}
          setPbDescription={setPbDescription}
          pbObject={pbObject}
          setPbObject={setPbObject}
          pbAction={pbAction}
          setPbAction={setPbAction}
          pbDirection={pbDirection}
          setPbDirection={setPbDirection}
          pbContext={pbContext}
          setPbContext={setPbContext}
          lastClicked={lastClicked}
          setLastClicked={setLastClicked}
          onGeneratePrompt={generatePrompt}
          onSaveMask={saveMask}
          onGoToPrev={goToPrev}
          onGoToNext={goToNext}
        />
      </div>
      
      <ToolsSidebar 
        cropMode={cropMode}
        setCropMode={(mode) => {
          setCropMode(mode);
          if (mode) setMaskMode(false);
        }}
        cropOrientation={cropOrientation}
        setCropOrientation={setCropOrientation}
        cropSize={cropSize}
        setCropSize={setCropSize}
        cropData={cropData}
        currentCrop={currentCrop}
        onSaveCrop={saveCrop}
        onDeleteCrop={deleteCrop}
        masks={masks}
        selectedMaskForGen={selectedMaskForGen}
        setSelectedMaskForGen={setSelectedMaskForGen}
        onGenerateSora2={generateSora2}
        onGenerateVeo3={generateVeo3}
        jobs={jobs}
        currentImage={currentImage}
        onShowVideos={handleShowVideos}
      />
      
      <JobsList 
        jobs={jobs}
        isCollapsed={jobsPanelCollapsed}
        onToggle={() => setJobsPanelCollapsed(!jobsPanelCollapsed)}
        apiBase={API_BASE}
      />
      
      <VideoViewer 
        isOpen={showVideoViewer}
        onClose={() => setShowVideoViewer(false)}
        maskIndex={selectedMaskForGen}
        imagePrefix={getImagePrefix()}
        apiBase={API_BASE}
        jobs={jobs}
        currentImage={currentImage}
        masks={masks}
      />
    </div>
  );
}

export default App;