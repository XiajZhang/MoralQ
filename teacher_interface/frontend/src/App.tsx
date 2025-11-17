import React, { useState, useEffect } from 'react';
import { Storybook, StorybookResult, Configuration, FeedbackData } from './types/index';
import { storybookApi, questionApi, firebaseApi } from './services/api';
import StorybookCard from './components/StorybookCard';
import ConfigurationForm from './components/ConfigurationForm';
import MoralApproval from './components/MoralApproval';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';

type AppStep = 'selection' | 'configuration' | 'results';

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<AppStep>('selection');
  const [storybooks, setStorybooks] = useState<Storybook[]>([]);
  const [filteredStorybooks, setFilteredStorybooks] = useState<Storybook[]>([]);
  const [selectedStorybooks, setSelectedStorybooks] = useState<Storybook[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<StorybookResult[]>([]);
  const [questionFeedbacks, setQuestionFeedbacks] = useState<Map<number, {
    feedback: 'positive' | 'negative';
    reasoning: string;
  }>>(new Map());

  // Load storybooks on component mount
  useEffect(() => {
    loadStorybooks();
  }, []);

  // Filter storybooks when search term changes
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredStorybooks(storybooks);
    } else {
      const term = searchTerm.toLowerCase();
      const filtered = storybooks.filter(book => 
        book.title.toLowerCase().includes(term) ||
        (book.tags && book.tags.some(tag => tag.toLowerCase().includes(term)))
      );
      setFilteredStorybooks(filtered);
    }
  }, [searchTerm, storybooks]);

  const loadStorybooks = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await storybookApi.getStorybooks();
      setStorybooks(data.storybooks || []);
      setFilteredStorybooks(data.storybooks || []);
    } catch (err) {
      setError('Error loading storybooks. Please try again.');
      console.error('Error loading storybooks:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleStorybookSelection = (bookId: string) => {
    const book = filteredStorybooks.find(b => b.id === bookId);
    if (!book) return;
    
    const existingIndex = selectedStorybooks.findIndex(selected => selected.id === bookId);
    
    if (existingIndex >= 0) {
      // Remove from selection
      setSelectedStorybooks(prev => prev.filter(book => book.id !== bookId));
    } else {
      // Add to selection
      setSelectedStorybooks(prev => [...prev, book]);
    }
  };

  const proceedToConfiguration = () => {
    if (selectedStorybooks.length === 0) {
      alert('Please select at least one storybook before proceeding.');
      return;
    }
    setCurrentStep('configuration');
  };

  const goBackToSelection = () => {
    setCurrentStep('selection');
  };

  const generateQuestions = async (config: Configuration) => {
    try {
      setLoading(true);
      setError(null);
      setCurrentStep('results');
      
      // Generate moral lessons and questions directly
      const data = await questionApi.generateMoral(config);
      
      if (data.success) {
        setResults(data.results);
        setQuestionFeedbacks(new Map());
      } else {
        setError(data.error || 'Failed to generate questions');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Error generating questions:', err);
    } finally {
      setLoading(false);
    }
  };

  const approveMoral = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await questionApi.approveMoral(results);
      
      if (data.success) {
        setResults(data.results);
        setCurrentStep('results');
      } else {
        setError(data.error || 'Failed to approve moral and generate questions');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Error approving moral:', err);
    } finally {
      setLoading(false);
    }
  };

  const regenerateMoral = async (selectedIndices: number[]) => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await questionApi.regenerateMoral(results, selectedIndices, 'negative');
      
      if (data.success) {
        setResults(data.results);
      } else {
        setError(data.error || 'Failed to regenerate moral lessons');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Error regenerating moral:', err);
    } finally {
      setLoading(false);
    }
  };


  const regenerateQuestions = async (generalFeedback: string, questionFeedbacks: Map<number, { feedback: 'positive' | 'negative'; reasoning: string; }>) => {
    if (results.length === 0) {
      alert('No results to regenerate from.');
      return;
    }
    
    if (!generalFeedback.trim()) {
      alert('Please provide general feedback about the question set before regenerating.');
      return;
    }
    
    try {
      setLoading(true);
      
      // Convert Map to object for JSON serialization
      const questionFeedbacksObj: { [key: number]: { feedback: 'positive' | 'negative'; reasoning: string; } } = {};
      questionFeedbacks.forEach((value, key) => {
        questionFeedbacksObj[key] = value;
      });
      
      // Only send storybooks that have feedback
      const storybooksWithFeedback = new Set<number>();
      
      // If there's general feedback, include all storybooks
      if (generalFeedback.trim()) {
        for (let i = 0; i < results.length; i++) {
          storybooksWithFeedback.add(i);
        }
      }
      
      // Also include storybooks with individual question feedback
      questionFeedbacks.forEach((_, globalIndex) => {
        // Find which storybook this question belongs to
        let currentIndex = 0;
        for (let i = 0; i < results.length; i++) {
          const questionCount = results[i].questions?.length || 0;
          if (globalIndex < currentIndex + questionCount) {
            storybooksWithFeedback.add(i);
            break;
          }
          currentIndex += questionCount;
        }
      });
      
      const config = {
        selectedStorybooks: Array.from(storybooksWithFeedback).map(index => results[index].storybook),
        objective: results[0].objective || 'moral',
        generalFeedback: generalFeedback,
        questionFeedbacks: questionFeedbacksObj,
        originalQuestions: Array.from(storybooksWithFeedback).map(index => results[index].questions || [])
      };
      
      const data = await questionApi.regenerateQuestions(config);
      
      if (data.success) {
        setResults(data.results);
        setQuestionFeedbacks(new Map());
        alert('Questions regenerated using your feedback! The AI has learned from your input.');
      } else {
        setError(data.error || 'Failed to regenerate questions');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Error regenerating questions:', err);
    } finally {
      setLoading(false);
    }
  };

  const exportQuestions = () => {
    alert('Questions exported successfully! (This will integrate with your backend)');
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <h1>MoralQ Teacher Interface</h1>
        <p className="subtitle">Teacher Interface for Storybook Question Generation</p>
      </header>

      {/* Fixed Action Bar */}
      {selectedStorybooks.length > 0 && currentStep === 'selection' && (
        <div className="fixed-action-bar">
          <div className="action-bar-content">
            <div className="selected-info">
              <span className="selected-count">{selectedStorybooks.length}</span> stories selected
            </div>
            <button className="btn btn-primary btn-large" onClick={proceedToConfiguration}>
              Configure Questions
            </button>
          </div>
        </div>
      )}

      {/* Step 1: Story Selection */}
      {currentStep === 'selection' && (
        <section className="step-section">
          <div className="step-header">
            <h2>Step 1: Select Storybooks</h2>
            <p>Choose the storybooks you want to create questions for</p>
          </div>
          
          <div className="search-container">
            <input 
              type="text" 
              placeholder="Search storybooks..." 
              className="search-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <button className="search-btn" onClick={() => setSearchTerm(searchTerm)}>
              Search
            </button>
          </div>
          
          <div className="storybook-grid">
            {loading ? (
              <div className="loading">Loading storybooks...</div>
            ) : error ? (
              <div className="error">{error}</div>
            ) : filteredStorybooks.length === 0 ? (
              <div className="no-results">No storybooks found.</div>
            ) : (
              filteredStorybooks.map(book => {
                const isSelected = selectedStorybooks.some(selected => selected.id === book.id);
                const selectionNumber = isSelected ? selectedStorybooks.findIndex(selected => selected.id === book.id) + 1 : undefined;
                
                return (
                  <StorybookCard
                    key={book.id}
                    storybook={book}
                    isSelected={isSelected}
                    selectionNumber={selectionNumber}
                    onToggle={toggleStorybookSelection}
                  />
                );
              })
            )}
          </div>
        </section>
      )}

      {/* Step 2: Configuration */}
      {currentStep === 'configuration' && (
        <section className="step-section">
          <div className="step-header">
            <h2>Step 2: Configure Questions</h2>
            <p>Set your objectives and preferences for question generation</p>
          </div>
          
          <ConfigurationForm
            selectedStorybooks={selectedStorybooks}
            onGenerate={generateQuestions}
            onBack={goBackToSelection}
          />
        </section>
      )}


      {/* Step 2: Results */}
      {currentStep === 'results' && (
        <section className="step-section">
          <div className="step-header">
            <h2>Step 2: Generated Questions</h2>
            <p>Review and customize your generated questions</p>
          </div>
          
          {loading ? (
            <div className="loading">Generating Questions...</div>
          ) : error ? (
            <div className="error">
              <h4>Error</h4>
              <p>{error}</p>
              <button className="btn btn-secondary" onClick={() => setCurrentStep('configuration')}>
                Try Again
              </button>
            </div>
          ) : (
            <ResultsDisplay
              results={results}
              onRegenerate={regenerateQuestions}
            />
          )}
          
          <div className="results-actions">
            <button className="btn btn-secondary" onClick={() => setCurrentStep('configuration')}>
              Back to Configuration
            </button>
            <button className="btn btn-success" onClick={exportQuestions}>
              Export Questions
            </button>
          </div>
        </section>
      )}
    </div>
  );
};

export default App;
