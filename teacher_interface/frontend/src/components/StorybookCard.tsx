import React from 'react';
import { Storybook } from '../types';
import { storybookApi } from '../services/api';

interface StorybookCardProps {
  storybook: Storybook;
  isSelected: boolean;
  selectionNumber?: number;
  onToggle: (storybookId: string) => void;
}

const StorybookCard: React.FC<StorybookCardProps> = ({
  storybook,
  isSelected,
  selectionNumber,
  onToggle,
}) => {
  const handleClick = () => {
    onToggle(storybook.id);
  };

  return (
    <div 
      className={`storybook-card ${isSelected ? 'selected' : ''}`}
      onClick={handleClick}
    >
      <div className="selection-indicator">
        {selectionNumber}
      </div>
      
      <div className="storybook-cover">
        {storybook.coverImage ? (
          <img 
            src={storybookApi.getStorybookImage(storybook.id)} 
            alt={storybook.title}
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const nextElement = target.nextElementSibling as HTMLElement;
              if (nextElement) nextElement.style.display = 'flex';
            }}
          />
        ) : null}
        <div style={{ display: storybook.coverImage ? 'none' : 'flex' }}>
          {storybook.title}
        </div>
      </div>
      
      <div className="storybook-info">
        <h3 className="storybook-title">{storybook.title}</h3>
        <div className="storybook-meta">Pages: {storybook.pageCount || 'Unknown'}</div>
        <div className="storybook-meta">Age: {storybook.ageRange || '4-6'}</div>
        
        {storybook.tags && storybook.tags.length > 0 && (
          <div className="storybook-tags">
            {storybook.tags.map((tag, index) => (
              <span key={index} className="tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default StorybookCard;
