import React from 'react';
import './ValidationStatsPanel.css';

function ValidationStatsPanel({ apiBase, isOpen, onToggle, validationStats }) {
  const stats = validationStats;

  const calculateGlobalProgress = () => {
    if (!stats) return { 
      annotated: 0, 
      total: 0, 
      percentage: 0, 
      byModel: {}, 
      byMetric: {
        localization: { annotated: 0, total: 0, percentage: 0 },
        articulation_type: { annotated: 0, total: 0, percentage: 0 },
        sequence_plausible: { annotated: 0, total: 0, percentage: 0 }
      } 
    };
    
    let totalAnnotated = 0;
    let totalVideos = 0;
    const byModel = {};
    const byMetric = {
      localization: { annotated: 0, total: 0 },
      articulation_type: { annotated: 0, total: 0 },
      sequence_plausible: { annotated: 0, total: 0 }
    };
    
    Object.entries(stats).forEach(([model, modelStats]) => {
      totalAnnotated += modelStats.annotated;
      totalVideos += modelStats.total;
      
      byModel[model] = {
        annotated: modelStats.annotated,
        total: modelStats.total,
        percentage: modelStats.total > 0 ? ((modelStats.annotated / modelStats.total) * 100).toFixed(1) : 0
      };
      
      // Aggregate per metric
      Object.keys(byMetric).forEach(metric => {
        const trueCount = modelStats.stats[metric][true] || 0;
        const falseCount = modelStats.stats[metric][false] || 0;
        const annotated = trueCount + falseCount;
        
        byMetric[metric].annotated += annotated;
        byMetric[metric].total += modelStats.total;
      });
    });
    
    // Calculate percentages for metrics
    Object.keys(byMetric).forEach(metric => {
      byMetric[metric].percentage = byMetric[metric].total > 0 
        ? ((byMetric[metric].annotated / byMetric[metric].total) * 100).toFixed(1) 
        : 0;
    });
    
    const percentage = totalVideos > 0 ? ((totalAnnotated / totalVideos) * 100).toFixed(1) : 0;
    
    return { annotated: totalAnnotated, total: totalVideos, percentage, byModel, byMetric };
  };

  const formatMetricStats = (modelStats, metric) => {
    const trueCount = modelStats.stats[metric][true] || 0;
    const falseCount = modelStats.stats[metric][false] || 0;
    const annotated = trueCount + falseCount;
    const total = modelStats.total;
    const percentage = annotated > 0 ? ((trueCount / annotated) * 100).toFixed(1) : 0;
    
    return { trueCount, annotated, total, percentage };
  };

  if (!isOpen) {
    return (
      <div className="stats-toggle-btn" onClick={onToggle}>
        📊 Show Stats
      </div>
    );
  }

  const globalProgress = calculateGlobalProgress();
  const models = ['ours', 'rgb', 'wan', 'sora2', 'veo3', 'kling'];
  const metrics = ['localization', 'articulation_type', 'sequence_plausible'];

  return (
    <>
      <div className="stats-panel">
        <div className="stats-header">
          <h3>Validation Statistics</h3>
          <button onClick={onToggle} className="close-btn">✕</button>
        </div>

        <div className="global-progress">
          <div className="progress-row overall">
            <div className="progress-label">Overall Progress:</div>
            <div className="progress-value">
              {globalProgress.annotated} / {globalProgress.total} ({globalProgress.percentage}%)
            </div>
          </div>
          
          <div className="progress-row by-model">
            <div className="progress-label">By Model:</div>
            <div className="progress-breakdown">
              {models.map(model => {
                const modelProgress = globalProgress.byModel?.[model];
                if (!modelProgress) return null;
                return (
                  <span key={model} className="progress-item">
                    <span className="item-label">{model}:</span>
                    <span className="item-value">
                      {modelProgress.annotated}/{modelProgress.total} ({modelProgress.percentage}%)
                    </span>
                  </span>
                );
              })}
            </div>
          </div>
          
          <div className="progress-row by-metric">
            <div className="progress-label">By Metric:</div>
            <div className="progress-breakdown">
              <span className="progress-item">
                <span className="item-label">Localization:</span>
                <span className="item-value">
                  {globalProgress.byMetric.localization.annotated}/{globalProgress.byMetric.localization.total} ({globalProgress.byMetric.localization.percentage}%)
                </span>
              </span>
              <span className="progress-item">
                <span className="item-label">Type:</span>
                <span className="item-value">
                  {globalProgress.byMetric.articulation_type.annotated}/{globalProgress.byMetric.articulation_type.total} ({globalProgress.byMetric.articulation_type.percentage}%)
                </span>
              </span>
              <span className="progress-item">
                <span className="item-label">Sequence:</span>
                <span className="item-value">
                  {globalProgress.byMetric.sequence_plausible.annotated}/{globalProgress.byMetric.sequence_plausible.total} ({globalProgress.byMetric.sequence_plausible.percentage}%)
                </span>
              </span>
            </div>
          </div>
        </div>

        <div className="stats-table-container">
          <table className="stats-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Localization</th>
                <th>Articulation Type</th>
                <th>Sequence Plausible</th>
              </tr>
            </thead>
            <tbody>
              {models.map(model => {
                if (!stats || !stats[model]) return null;
                const modelStats = stats[model];
                
                return (
                  <tr key={model}>
                    <td className="model-name">{model}</td>
                    {metrics.map(metric => {
                      const metricStats = formatMetricStats(modelStats, metric);
                      return (
                        <td key={metric}>
                          <div className="metric-cell">
                            <span className="metric-counts">
                              {metricStats.trueCount}/{metricStats.annotated}/{metricStats.total}
                            </span>
                            <span className="metric-percentage">
                              {metricStats.percentage}%
                            </span>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <div className="stats-overlay" onClick={onToggle}></div>
    </>
  );
}

export default ValidationStatsPanel;