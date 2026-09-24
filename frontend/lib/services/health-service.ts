import { HttpClient, defaultHttpClient } from '../api-client';
import { HealthResponse } from '../types';

export class HealthService {
  private client: HttpClient;

  constructor(client: HttpClient = defaultHttpClient) {
    this.client = client;
  }

  /**
   * Checks server health status and loaded model counts
   */
  public async checkHealth(): Promise<HealthResponse> {
    try {
      return await this.client.get<HealthResponse>('/health', { timeoutMs: 3000 });
    } catch (err: any) {
      return {
        status: 'offline',
        models_loaded: 0,
        error: err.message || 'Backend server is offline or unreachable',
      };
    }
  }
}

export const healthService = new HealthService();
