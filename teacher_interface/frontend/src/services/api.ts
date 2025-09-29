import axios from 'axios';
import { Storybook, Configuration, StorybookResult, FirebaseStatus, FirebaseData, Question } from '../types';

const API_BASE_URL = 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const storybookApi = {
  getStorybooks: async (): Promise<{ storybooks: Storybook[] }> => {
    const response = await api.get('/api/storybooks');
    return response.data as { storybooks: Storybook[] };
  },

  getStorybookImage: (storybookId: string): string => {
    return `${API_BASE_URL}/api/image/${storybookId}`;
  },
};

export const questionApi = {
  generateMoral: async (config: Configuration): Promise<{ success: boolean; results: StorybookResult[]; stage: string; error?: string }> => {
    const response = await api.post('/api/generate-moral', config);
    return response.data as { success: boolean; results: StorybookResult[]; stage: string; error?: string };
  },

  approveMoral: async (results: StorybookResult[]): Promise<{ success: boolean; results: StorybookResult[]; stage: string; error?: string }> => {
    const response = await api.post('/api/approve-moral', { results });
    return response.data as { success: boolean; results: StorybookResult[]; stage: string; error?: string };
  },

  regenerateMoral: async (results: StorybookResult[], selectedIndices: number[], feedback: string): Promise<{ success: boolean; results: StorybookResult[]; stage: string; error?: string }> => {
    const response = await api.post('/api/regenerate-moral', { results, selectedIndices, feedback });
    return response.data as { success: boolean; results: StorybookResult[]; stage: string; error?: string };
  },

  generateQuestions: async (config: Configuration): Promise<{ success: boolean; results: StorybookResult[]; error?: string }> => {
    const response = await api.post('/api/generate-questions', config);
    return response.data as { success: boolean; results: StorybookResult[]; error?: string };
  },

  regenerateQuestions: async (config: {
    selectedStorybooks: Storybook[];
    objective: string;
    generalFeedback: string;
    questionFeedbacks?: { [key: number]: { feedback: 'positive' | 'negative'; reasoning: string; } };
    originalQuestions?: Question[][];
  }): Promise<{ success: boolean; results: StorybookResult[]; error?: string }> => {
    const response = await api.post('/api/regenerate-questions', config);
    return response.data as { success: boolean; results: StorybookResult[]; error?: string };
  },
};

export const firebaseApi = {
  checkStatus: async (): Promise<FirebaseStatus> => {
    const response = await api.get('/api/firebase/status');
    return response.data as FirebaseStatus;
  },

  getStudents: async (): Promise<FirebaseData> => {
    const response = await api.get('/api/firebase/students');
    return response.data as FirebaseData;
  },

  getStorybooks: async (): Promise<FirebaseData> => {
    const response = await api.get('/api/firebase/storybooks');
    return response.data as FirebaseData;
  },

  getTopLevelNode: async (): Promise<FirebaseData> => {
    const response = await api.get('/api/firebase/top-level-node');
    return response.data as FirebaseData;
  },
};

export default api;
