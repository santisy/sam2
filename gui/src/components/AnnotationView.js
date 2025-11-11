import React from 'react';

function AnnotationView({
  currentImage,
  images,
  masks,
  showOtherMasks,
  setShowOtherMasks,
  prompt,
  setPrompt,
  activeMask,
  pbIndicator,
  setPbIndicator,
  pbPart,
  setPbPart,
  pbDescription,
  setPbDescription,
  pbObject,
  setPbObject,
  pbAction,
  setPbAction,
  pbDirection,
  setPbDirection,
  pbContext,
  setPbContext,
  lastClicked,
  setLastClicked,
  onGeneratePrompt,
  onSaveMask,
  onGoToPrev,
  onGoToNext
}) {
  if (!currentImage) {
    return (
      <div className="no-image-message">
        {images.length === 0 ? 'No images in directory' : 'Select an image'}
      </div>
    );
  }

  return (
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
      
      <div className="prompt-builder">
        <div className="pb-segment">
          <div className="pb-label">indicator (optional)</div>
          <div className="pb-buttons">
            <button className={lastClicked === 'indicator-top' ? 'active' : ''} onClick={() => { setPbIndicator('top'); setLastClicked('indicator-top'); }}>top</button>
            <button className={lastClicked === 'indicator-middle' ? 'active' : ''} onClick={() => { setPbIndicator('middle'); setLastClicked('indicator-middle'); }}>middle</button>
            <button className={lastClicked === 'indicator-bottom' ? 'active' : ''} onClick={() => { setPbIndicator('bottom'); setLastClicked('indicator-bottom'); }}>bottom</button>
            <button className={lastClicked === 'indicator-left' ? 'active' : ''} onClick={() => { setPbIndicator('left'); setLastClicked('indicator-left'); }}>left</button>
            <button className={lastClicked === 'indicator-right' ? 'active' : ''} onClick={() => { setPbIndicator('right'); setLastClicked('indicator-right'); }}>right</button>
          </div>
          <button className="clear-btn" onClick={() => { setPbIndicator(''); setLastClicked(''); }}>Clear</button>
          <input type="text" value={pbIndicator} onChange={(e) => setPbIndicator(e.target.value)} />
        </div>

        <div className="pb-segment">
          <div className="pb-label">part</div>
          <div className="pb-buttons">
            <button className={lastClicked === 'part-drawer' ? 'active' : ''} onClick={() => { setPbPart('drawer'); setLastClicked('part-drawer'); }}>drawer</button>
            <button className={lastClicked === 'part-door' ? 'active' : ''} onClick={() => { setPbPart('door'); setLastClicked('part-door'); }}>door</button>
          </div>
          <input type="text" value={pbPart} onChange={(e) => setPbPart(e.target.value)} />
        </div>

        <div className="pb-segment">
          <div className="pb-label">description (optional)</div>
          <div className="pb-buttons"></div>
          <button className="clear-btn" onClick={() => setPbDescription('')}>Clear</button>
          <input type="text" value={pbDescription} onChange={(e) => setPbDescription(e.target.value)} placeholder="(optional)" />
        </div>

        <div className="pb-segment">
          <div className="pb-label">object</div>
          <div className="pb-buttons">
            <button className={lastClicked === 'object-cabinet' ? 'active' : ''} onClick={() => { setPbObject('cabinet'); setLastClicked('object-cabinet'); }}>cabinet</button>
            <button className={lastClicked === 'object-washer' ? 'active' : ''} onClick={() => { setPbObject('washer'); setLastClicked('object-washer'); }}>washer</button>
            <button className={lastClicked === 'object-refrigerator' ? 'active' : ''} onClick={() => { setPbObject('refrigerator'); setLastClicked('object-refrigerator'); }}>fridge</button>
            <button className={lastClicked === 'object-oven' ? 'active' : ''} onClick={() => { setPbObject('oven'); setLastClicked('object-oven'); }}>oven</button>
          </div>
          <input type="text" value={pbObject} onChange={(e) => setPbObject(e.target.value)} />
        </div>

        <div className="pb-segment">
          <div className="pb-label">action</div>
          <div className="pb-buttons">
            <button className={lastClicked === 'action-sliding' ? 'active' : ''} onClick={() => { setPbAction('sliding'); setLastClicked('action-sliding'); }}>sliding</button>
            <button className={lastClicked === 'action-swinging' ? 'active' : ''} onClick={() => { setPbAction('swinging'); setLastClicked('action-swinging'); }}>swinging</button>
          </div>
          <input type="text" value={pbAction} onChange={(e) => setPbAction(e.target.value)} />
        </div>

        <div className="pb-segment">
          <div className="pb-label">direction</div>
          <div className="pb-buttons">
            <button className={lastClicked === 'direction-out' ? 'active' : ''} onClick={() => { setPbDirection('out'); setLastClicked('direction-out'); }}>out</button>
            <button className={lastClicked === 'direction-side' ? 'active' : ''} onClick={() => { setPbDirection('to the side'); setLastClicked('direction-side'); }}>side</button>
            <button className={lastClicked === 'direction-left' ? 'active' : ''} onClick={() => { setPbDirection('to the left'); setLastClicked('direction-left'); }}>left</button>
            <button className={lastClicked === 'direction-right' ? 'active' : ''} onClick={() => { setPbDirection('to the right'); setLastClicked('direction-right'); }}>right</button>
          </div>
          <input type="text" value={pbDirection} onChange={(e) => setPbDirection(e.target.value)} />
        </div>

        <div className="pb-segment">
          <div className="pb-label">context (optional)</div>
          <div className="pb-buttons"></div>
          <button className="clear-btn" onClick={() => setPbContext('')}>Clear</button>
          <input type="text" value={pbContext} onChange={(e) => setPbContext(e.target.value)} placeholder="(optional)" />
        </div>

        <button onClick={onGeneratePrompt} className="pb-generate">→ Generate</button>
      </div>
      
      <div className="controls">
        <button onClick={onGoToPrev}>← Prev</button>
        <input
          type="text"
          placeholder="Enter prompt..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button onClick={onSaveMask} disabled={!activeMask}>Save</button>
        <button onClick={onGoToNext}>Next →</button>
      </div>
    </>
  );
}

export default AnnotationView;