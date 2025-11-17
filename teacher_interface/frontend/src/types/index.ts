// Types and interfaces for the MoralQ Teacher Interface

export interface Storybook {
  id: string;
  title: string;
  pageCount?: number;
  ageRange?: string;
  tags?: string[];
  coverImage?: string;
}

export interface Question {
  question: string;
  type: string;
  difficulty: string;
  page_number: number;
  explanation: string;
}

export interface Segment {
  name: string;
  SUMMARY: string;
  REASONING: string;
  START: number;
  END: number;
}

export interface MoralCandidate {
  moral: string;
  segments: Segment[];
  candidate_id: number | string;
  quality_score: number;
  generation_method: string;
  is_gepa_improved?: boolean;
}

export interface MoralData {
  generated?: string; // Optional - may be missing in error cases
  text?: string; // Optional fallback for generated
  candidates?: MoralCandidate[];
  optimization_applied?: boolean;
  status?: 'pending' | 'approved' | 'rejected';
  feedback?: 'positive' | 'negative' | null;
  regenerations?: number;
  timestamp?: string;
}

export interface QuestionsData {
  generated: Question[];
  status: 'not_generated' | 'generated' | 'approved' | 'rejected' | 'failed';
  feedback_summary: string | null;
  regenerations: number;
  timestamp: string | null;
}

export interface StorybookResult {
  storybook: Storybook;
  objective?: string;
  moral?: MoralData;
  segments?: Segment[];
  questions?: Question[];
  learning_objectives?: string[];
  error?: string; // Error message if generation failed
}

export interface Configuration {
  objective: string;
  questionFrequency: string;
  cognitiveLevel: string;
  toneStyle: string;
  includeSnippets: boolean;
  language: string;
  selectedStorybooks: Storybook[];
}

export interface FeedbackData {
  highlightedQuestions: Set<number>;
}

export interface FirebaseStatus {
  connected: boolean;
  server_url?: string;
  authenticated?: boolean;
  error?: string;
}

export interface FirebaseData {
  students?: any;
  storybooks?: any;
  data?: any;
  node?: string;
  error?: string;
}
