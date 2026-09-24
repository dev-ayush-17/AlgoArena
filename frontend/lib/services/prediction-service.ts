import { HttpClient, defaultHttpClient } from '../api-client';
import { FormSchemaResponse, PredictionResponse } from '../types';

export class PredictionService {
  private client: HttpClient;

  constructor(client: HttpClient = defaultHttpClient) {
    this.client = client;
  }

  /**
   * Fetches dynamic feature configuration schema for form rendering
   */
  public async getFormSchema(): Promise<FormSchemaResponse> {
    return this.client.get<FormSchemaResponse>('/form-schema');
  }

  /**
   * Submits session metrics to run inference across all 5 models
   */
  public async predict(features: Record<string, any>): Promise<PredictionResponse> {
    return this.client.post<PredictionResponse>('/predict', { features });
  }
}

export const predictionService = new PredictionService();
