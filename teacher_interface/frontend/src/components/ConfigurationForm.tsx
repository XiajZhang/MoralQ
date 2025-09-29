import React, { useState } from 'react';
import { Configuration, Storybook } from '../types';

interface ConfigurationFormProps {
  selectedStorybooks: Storybook[];
  onGenerate: (config: Configuration) => void;
  onBack: () => void;
}

const ConfigurationForm: React.FC<ConfigurationFormProps> = ({
  selectedStorybooks,
  onGenerate,
  onBack,
}) => {
  const [objective, setObjective] = useState('moral');
  const [customObjective, setCustomObjective] = useState('');
  const [questionFrequency, setQuestionFrequency] = useState('per-segment');
  const [cognitiveLevel, setCognitiveLevel] = useState('remember');
  const [toneStyle, setToneStyle] = useState('open-ended');
  const [includeSnippets, setIncludeSnippets] = useState(false);
  const [language, setLanguage] = useState('english');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const config: Configuration = {
      objective: customObjective.trim() || objective,
      questionFrequency,
      cognitiveLevel,
      toneStyle,
      includeSnippets,
      language,
      selectedStorybooks,
    };
    
    onGenerate(config);
  };

  return (
    <div className="config-panel">
      <div className="config-grid">
        <div className="config-item">
          <label htmlFor="objective">Objective</label>
          <select 
            id="objective" 
            className="form-control"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
          >
            <option value="moral">Moral</option>
            <option value="comprehension">Comprehension</option>
            <option value="empathy">Empathy</option>
            <option value="critical-thinking">Critical Thinking</option>
            <option value="kindness">Kindness</option>
            <option value="curiosity">Curiosity</option>
            <option value="responsibility">Responsibility</option>
            <option value="friendship">Friendship</option>
          </select>
        </div>
        
        <div className="config-item">
          <label htmlFor="custom-objective">Custom Objective (Optional)</label>
          <input 
            type="text" 
            id="custom-objective" 
            className="form-control"
            placeholder="Enter a custom objective (overrides dropdown)"
            value={customObjective}
            onChange={(e) => setCustomObjective(e.target.value)}
          />
          <small className="form-text">Leave empty to use the selected objective above</small>
        </div>
        
        <div className="config-item">
          <label htmlFor="questionFrequency">Question Frequency</label>
          <select 
            id="questionFrequency" 
            className="form-control"
            value={questionFrequency}
            onChange={(e) => setQuestionFrequency(e.target.value)}
          >
            <option value="per-segment">Per Segment</option>
          </select>
        </div>
        
        <div className="config-item">
          <label htmlFor="cognitiveLevel">Cognitive Level</label>
          <select 
            id="cognitiveLevel" 
            className="form-control"
            value={cognitiveLevel}
            onChange={(e) => setCognitiveLevel(e.target.value)}
          >
            <option value="remember">Remember</option>
            <option value="understand">Understand</option>
            <option value="apply">Apply</option>
            <option value="analyze">Analyze</option>
            <option value="evaluate">Evaluate</option>
            <option value="create">Create</option>
          </select>
        </div>
        
        <div className="config-item">
          <label htmlFor="toneStyle">Tone/Style</label>
          <select 
            id="toneStyle" 
            className="form-control"
            value={toneStyle}
            onChange={(e) => setToneStyle(e.target.value)}
          >
            <option value="open-ended">Open-ended</option>
            <option value="mcq">Multiple Choice</option>
            <option value="reflective">Reflective</option>
          </select>
        </div>
        
        <div className="config-item">
          <label htmlFor="includeSnippets">Include Story Snippets?</label>
          <div className="toggle-container">
            <input 
              type="checkbox" 
              id="includeSnippets" 
              className="toggle-input"
              checked={includeSnippets}
              onChange={(e) => setIncludeSnippets(e.target.checked)}
            />
            <label htmlFor="includeSnippets" className="toggle-label">
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>
        
        <div className="config-item">
          <label htmlFor="language">Language</label>
          <select 
            id="language" 
            className="form-control"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="english">English</option>
            <option value="arabic">Arabic</option>
          </select>
        </div>
      </div>
      
      <div className="config-actions">
        <button className="btn btn-secondary" onClick={onBack}>
          Back to Selection
        </button>
        <button className="btn btn-primary" onClick={handleSubmit}>
          Generate Questions
        </button>
      </div>
    </div>
  );
};

export default ConfigurationForm;
