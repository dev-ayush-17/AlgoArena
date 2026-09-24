import { predictionService } from './services/prediction-service';
import { metricsService } from './services/metrics-service';
import { healthService } from './services/health-service';
import { ApiError } from './api-client';
import {
  FormSchemaResponse,
  PredictionResponse,
  MetricsResponse,
  HealthResponse,
} from './types';

// Backward compatible alias
export const APIError = ApiError;

/**
 * API Gateway Facade Functions
 */
export async function fetchFormSchema(): Promise<FormSchemaResponse> {
  return predictionService.getFormSchema();
}

export async function submitPrediction(
  features: Record<string, any>
): Promise<PredictionResponse> {
  return predictionService.predict(features);
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return metricsService.getMetrics();
}

export async function fetchHealth(): Promise<HealthResponse> {
  return healthService.checkHealth();
}

export { predictionService, metricsService, healthService };
