import React, { useState } from 'react';
import { Question } from '../types';

interface QuestionItemProps {
  question: Question;
  questionIndex: number;
  globalIndex: number;
  onFeedbackChange: (globalIndex: number, feedback: 'positive' | 'negative' | null, reasoning?: string) => void;
}

const QuestionItem: React.FC<QuestionItemProps> = ({
  question,
  questionIndex,
  globalIndex,
  onFeedbackChange,
}) => {
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null);
  const [reasoning, setReasoning] = useState('');

  const handleFeedbackChange = (newFeedback: 'positive' | 'negative' | null) => {
    setFeedback(newFeedback);
    if (newFeedback === null) {
      setReasoning('');
    }
    onFeedbackChange(globalIndex, newFeedback, reasoning);
  };

  const handleReasoningChange = (newReasoning: string) => {
    setReasoning(newReasoning);
    onFeedbackChange(globalIndex, feedback, newReasoning);
  };

  return (
    <div className="question-item" data-question-id={globalIndex}>
      <div className="question-header">
        <div className="question-meta">
          <h5>Question {questionIndex + 1}: {question.type || 'Moral Question'}</h5>
          <div className="question-details">
            <span className="difficulty">{question.difficulty}</span>
            <span className="page">Page {question.page_number}</span>
          </div>
        </div>
        <div className="question-feedback-controls">
          <label className="feedback-option">
            <input
              type="radio"
              name={`feedback-${globalIndex}`}
              checked={feedback === 'positive'}
              onChange={() => handleFeedbackChange('positive')}
            />
            <span className="feedback-icon positive">[+]</span>
            <span className="feedback-label">Good</span>
          </label>
          <label className="feedback-option">
            <input
              type="radio"
              name={`feedback-${globalIndex}`}
              checked={feedback === 'negative'}
              onChange={() => handleFeedbackChange('negative')}
            />
            <span className="feedback-icon negative">[-]</span>
            <span className="feedback-label">Bad</span>
          </label>
        </div>
      </div>
      
      <div className="question-content">
        <p className="question-text">{question.question}</p>
        {question.explanation && (
          <p className="question-explanation"><strong>Explanation:</strong> {question.explanation}</p>
        )}
      </div>

      {feedback && (
        <div className="question-reasoning">
          <label htmlFor={`reasoning-${globalIndex}`}>
            <strong>
              {feedback === 'positive' ? 'Why do you like this question?' : 'Why don\'t you like this question?'}
            </strong>
          </label>
          <textarea
            id={`reasoning-${globalIndex}`}
            value={reasoning}
            onChange={(e) => handleReasoningChange(e.target.value)}
            placeholder={
              feedback === 'positive' 
                ? 'e.g., This question is age-appropriate and encourages critical thinking...'
                : 'e.g., This question is too complex for 4-6 year olds...'
            }
            rows={2}
            className="reasoning-input"
          />
        </div>
      )}
    </div>
  );
};

export default QuestionItem;
