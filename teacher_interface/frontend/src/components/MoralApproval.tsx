import React, { useState } from 'react';
import { StorybookResult } from '../types';

interface MoralApprovalProps {
  results: StorybookResult[];
  onApprove: () => void;
  onRegenerate: (selectedIndices: number[]) => void;
  isLoading: boolean;
}

const MoralApproval: React.FC<MoralApprovalProps> = ({
  results,
  onApprove,
  onRegenerate,
  isLoading
}) => {
  const [selectedMorals, setSelectedMorals] = useState<Set<number>>(new Set());

  const toggleSelection = (index: number) => {
    setSelectedMorals(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  const handleRegenerate = () => {
    const selectedIndices = Array.from(selectedMorals);
    if (selectedIndices.length > 0) {
      onRegenerate(selectedIndices);
      setSelectedMorals(new Set()); // Clear selection after regeneration
    }
  };

  const selectAll = () => {
    setSelectedMorals(new Set(results.map((_, index) => index)));
  };

  const clearSelection = () => {
    setSelectedMorals(new Set());
  };
  return (
    <div className="moral-approval">
      <div className="approval-header">
        <h2>Review Generated Moral Lessons</h2>
        <p>Please review the moral lessons generated for each storybook. You can approve them to proceed with question generation, or regenerate if needed.</p>
      </div>

      <div className="selection-controls">
        <div className="selection-info">
          <span>{selectedMorals.size} of {results.length} selected</span>
        </div>
        <div className="selection-actions">
          <button 
            className="btn btn-sm btn-outline" 
            onClick={selectAll}
            disabled={selectedMorals.size === results.length}
          >
            Select All
          </button>
          <button 
            className="btn btn-sm btn-outline" 
            onClick={clearSelection}
            disabled={selectedMorals.size === 0}
          >
            Clear Selection
          </button>
        </div>
      </div>

      <div className="moral-results">
        {results.map((result, index) => (
          <div 
            key={result.storybook.id} 
            className={`moral-item ${selectedMorals.has(index) ? 'selected' : ''}`}
          >
            <div className="moral-item-header">
              <div className="selection-checkbox">
                <input
                  type="checkbox"
                  checked={selectedMorals.has(index)}
                  onChange={() => toggleSelection(index)}
                  id={`moral-${index}`}
                />
                <label htmlFor={`moral-${index}`}></label>
              </div>
              <div className="storybook-header">
                <h3>{result.storybook.title}</h3>
                {result.moral && (
                  <div className="moral-status">
                    <span className={`status-badge ${result.moral.status || 'pending'}`}>
                      {result.moral.status || 'pending'}
                    </span>
                    {(result.moral.regenerations || 0) > 0 && (
                      <span className="regeneration-count">
                        Regenerated {result.moral.regenerations} time(s)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            {result.moral && (
              <div className="moral-content">
                <h4>Generated Moral Lesson:</h4>
                {result.moral.candidates && result.moral.candidates.length > 0 ? (
                  <div className="moral-candidates">
                    {result.moral.candidates.map((candidate, candidateIndex) => (
                      <div key={candidate.candidate_id} className="moral-candidate">
                        <div className="candidate-header">
                          <div className="candidate-radio">
                            <input
                              type="radio"
                              name={`moral-${index}`}
                              id={`moral-${index}-candidate-${candidateIndex}`}
                              defaultChecked={candidateIndex === 0}
                            />
                            <label htmlFor={`moral-${index}-candidate-${candidateIndex}`}>
                              <span className="candidate-label">
                                Candidate {candidateIndex + 1}
                              </span>
                              <span className="quality-score">
                                Quality: {(candidate.quality_score * 100).toFixed(0)}%
                              </span>
                              <span className="generation-method">
                                {candidate.generation_method}
                              </span>
                            </label>
                          </div>
                        </div>
                        <div className="candidate-moral">
                          {candidate.moral}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="moral-text">
                    {result.moral.generated || result.moral.text || 'No moral lesson generated'}
                  </div>
                )}
              </div>
            )}

            {result.moral?.timestamp && (
              <div className="moral-meta">
                <small>Generated: {new Date(result.moral.timestamp).toLocaleString()}</small>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="approval-actions">
        <button 
          className="btn btn-secondary" 
          onClick={handleRegenerate}
          disabled={isLoading || selectedMorals.size === 0}
        >
          {isLoading ? 'Regenerating...' : `Regenerate Selected (${selectedMorals.size})`}
        </button>
        
        <button 
          className="btn btn-primary" 
          onClick={onApprove}
          disabled={isLoading}
        >
          {isLoading ? 'Generating Questions...' : 'Approve & Generate Questions'}
        </button>
      </div>
    </div>
  );
};

export default MoralApproval;
