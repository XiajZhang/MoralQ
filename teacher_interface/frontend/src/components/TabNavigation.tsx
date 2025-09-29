import React from 'react';
import { Storybook } from '../types';

interface TabNavigationProps {
  storybooks: Storybook[];
  activeTab: number;
  onTabChange: (tabIndex: number) => void;
}

const TabNavigation: React.FC<TabNavigationProps> = ({
  storybooks,
  activeTab,
  onTabChange,
}) => {
  return (
    <div className="tabs-container">
      <div className="tabs-nav">
        {storybooks.map((storybook, index) => (
          <button
            key={storybook.id}
            className={`tab-button ${activeTab === index ? 'active' : ''}`}
            onClick={() => onTabChange(index)}
          >
            {storybook.title}
          </button>
        ))}
      </div>
    </div>
  );
};

export default TabNavigation;
