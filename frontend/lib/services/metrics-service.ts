import { HttpClient, defaultHttpClient } from '../api-client';
import { MetricsResponse } from '../types';

export class MetricsService {
  private client: HttpClient;

  constructor(client: HttpClient = defaultHttpClient) {
    this.client = client;
  }

  /**
   * Fetches pre-computed offline evaluation metrics for all 5 models
   */
  public async getMetrics(): Promise<MetricsResponse> {
    return this.client.get<MetricsResponse>('/metrics');
  }
}

export const metricsService = new MetricsService();
