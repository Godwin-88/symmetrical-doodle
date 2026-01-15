/**
 * API Client for Intelligence Layer and Execution Core
 */

const INTELLIGENCE_API_URL = import.meta.env.VITE_INTELLIGENCE_API_URL || 'http://localhost:8000';
const EXECUTION_API_URL = import.meta.env.VITE_EXECUTION_API_URL || 'http://localhost:8001';

export interface ApiError {
  message: string;
  status: number;
  detail?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async get<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      return response.json();
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }

  async delete<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw {
          message: error.detail || 'Request failed',
          status: response.status,
          detail: error.detail,
        } as ApiError;
      }

      // DELETE might return empty response
      const text = await response.text();
      return text ? JSON.parse(text) : ({} as T);
    } catch (error) {
      if ((error as ApiError).status) {
        throw error;
      }
      throw {
        message: 'Network error',
        status: 0,
        detail: (error as Error).message,
      } as ApiError;
    }
  }
}

// API clients
export const intelligenceApi = new ApiClient(INTELLIGENCE_API_URL);
export const executionApi = new ApiClient(EXECUTION_API_URL);

// Health check
export async function checkIntelligenceHealth(): Promise<{ status: string; service: string }> {
  return intelligenceApi.get('/health');
}

export async function checkExecutionHealth(): Promise<string> {
  return executionApi.get('/health');
}
