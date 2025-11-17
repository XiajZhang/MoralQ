import React from 'react';
import { StorybookResult, Question } from '../types';
import QuestionItem from './QuestionItem';

interface TabPanelProps {
  result: StorybookResult;
  startQuestionIndex: number;
  onQuestionFeedback: (globalIndex: number, feedback: 'positive' | 'negative' | null, reasoning?: string) => void;
  isActive: boolean;
}

const TabPanel: React.FC<TabPanelProps> = ({
  result,
  startQuestionIndex,
  onQuestionFeedback,
  isActive,
}) => {
  const { storybook, moral, objective, learning_objectives, questions } = result;
  const moralText = moral?.generated || moral?.text || 'No moral lesson generated';

  return (
    <div className={`tab-panel ${isActive ? 'active' : ''}`}>
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
          {questions.map((question, qIndex) => {
            const globalIndex = startQuestionIndex + qIndex;
            return (
              <QuestionItem
                key={globalIndex}
                question={question}
                questionIndex={qIndex}
                globalIndex={globalIndex}
                onFeedbackChange={onQuestionFeedback}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

export default TabPanel;
