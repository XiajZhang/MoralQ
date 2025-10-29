import React, { useState } from 'react';
import { StorybookResult, Question } from '../types';
import TabNavigation from './TabNavigation';
import TabPanel from './TabPanel';
import QuestionItem from './QuestionItem';

interface QuestionFeedback {
  feedback: 'positive' | 'negative';
  reasoning: string;
}

interface ResultsDisplayProps {
  results: StorybookResult[];
  onRegenerate: (generalFeedback: string, questionFeedbacks: Map<number, QuestionFeedback>) => void;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  results,
  onRegenerate,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [generalFeedback, setGeneralFeedback] = useState('');
  const [questionFeedbacks, setQuestionFeedbacks] = useState<Map<number, QuestionFeedback>>(new Map());

  const handleQuestionFeedback = (globalIndex: number, feedback: 'positive' | 'negative' | null, reasoning?: string) => {
    setQuestionFeedbacks(prev => {
      const newMap = new Map(prev);
      if (feedback === null) {
        newMap.delete(globalIndex);
      } else {
        newMap.set(globalIndex, {
          feedback,
          reasoning: reasoning || ''
        });
      }
      return newMap;
    });
  };

  const handleRegenerate = () => {
    if (generalFeedback.trim().length === 0) {
      alert('Please provide general feedback about the entire question set before regenerating.');
      return;
    }
    
    onRegenerate(generalFeedback.trim(), questionFeedbacks);
    setGeneralFeedback('');
    setQuestionFeedbacks(new Map());
  };


  if (results.length === 0) {
    return (
      <div className="error">
        No results generated. Please try again.
      </div>
    );
  }

  // If multiple storybooks, use tabbed interface
  if (results.length > 1) {
    return (
      <>
        <TabNavigation
          storybooks={results.map(result => result.storybook)}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
        
        <div className="tabs-content">
          {results.map((result, index) => {
            const startQuestionIndex = results
              .slice(0, index)
              .reduce((acc, r) => acc + (r.questions?.length || 0), 0);
            
            return (
              <TabPanel
                key={result.storybook.id}
                result={result}
                startQuestionIndex={startQuestionIndex}
                onQuestionFeedback={handleQuestionFeedback}
                isActive={activeTab === index}
              />
            );
          })}
        </div>
        
        <div className="feedback-section">
          <h4>General Feedback <span className="required">*</span></h4>
          <p>Provide overall feedback about the question set. This helps the AI learn and improve future question generation.</p>
          
          <div className="general-feedback-section">
            <label htmlFor="generalFeedback">
              <strong>Your Feedback on the Question Set</strong>
              <small>Share your thoughts on the overall quality, appropriateness, and areas for improvement</small>
            </label>
            <textarea
              id="generalFeedback"
              value={generalFeedback}
              onChange={(e) => setGeneralFeedback(e.target.value)}
              placeholder="e.g., The questions are good overall but could be more age-appropriate for 4-6 year olds. Some questions are too complex, while others are perfect for the target age group..."
              rows={4}
              className="general-feedback-input"
            />
          </div>
          
          <div className="feedback-actions">
            <button 
              className="btn btn-primary btn-large" 
              onClick={handleRegenerate}
              style={{
                background: generalFeedback.trim() ? '#27ae60' : '#3498db'
              }}
            >
              Regenerate Questions
            </button>
            <p className="feedback-note">
              Provide your feedback above, then click regenerate to improve future questions based on your input!
            </p>
          </div>
        </div>
      </>
    );
  }

  // Single storybook display
  const result = results[0];
  const { storybook, moral, objective, learning_objectives, questions } = result;
  const moralText = moral.generated;

  return (
    <div className="results-container">
      <div className="storybook-result">
        <h3>{storybook.title}</h3>
        
        <div className="objective-info">
          <h4>Learning Objective:</h4>
          <p>{objective}</p>
        </div>
        
        <div className="moral-lesson">
          <h4>Moral Lesson:</h4>
          <p>{moralText}</p>
        </div>
        
        {learning_objectives && learning_objectives.length > 0 && (
          <div className="learning-objectives">
            <h4>Learning Objectives:</h4>
            <ul>
              {learning_objectives.map((objective, index) => (
                <li key={index}>{objective}</li>
              ))}
            </ul>
          </div>
        )}
        
        {questions && questions.length > 0 && (
          <div className="questions">
            <h4>Generated Questions:</h4>
            {questions.map((question, qIndex) => (
              <QuestionItem
                key={qIndex}
                question={question}
                questionIndex={qIndex}
                globalIndex={qIndex}
                onFeedbackChange={handleQuestionFeedback}
              />
            ))}
          </div>
        )}
      </div>
      
      <div className="feedback-section">
        <h4>General Feedback <span className="required">*</span></h4>
        <p>Provide overall feedback about the question set. This helps the AI learn and improve future question generation.</p>
        
        <div className="general-feedback-section">
          <label htmlFor="generalFeedback">
            <strong>Your Feedback on the Question Set</strong>
            <small>Share your thoughts on the overall quality, appropriateness, and areas for improvement</small>
          </label>
          <textarea
            id="generalFeedback"
            value={generalFeedback}
            onChange={(e) => setGeneralFeedback(e.target.value)}
            placeholder="e.g., The questions are good overall but could be more age-appropriate for 4-6 year olds. Some questions are too complex, while others are perfect for the target age group..."
            rows={4}
            className="general-feedback-input"
          />
        </div>
        
        <div className="feedback-actions">
          <button 
            className="btn btn-primary btn-large" 
            onClick={handleRegenerate}
            style={{
              background: generalFeedback.trim() ? '#27ae60' : '#3498db'
            }}
          >
            Regenerate Questions
          </button>
          <p className="feedback-note">
            Provide your feedback above, then click regenerate to improve future questions based on your input!
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
