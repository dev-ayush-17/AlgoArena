export interface NumericalFeature {
  name: string;
  dtype: string;
  min: number;
  max: number;
  default: number;
  description?: string;
}

export interface CategoricalFeature {
  name: string;
  dtype: string;
  categories: (string | number)[];
  default: string | number;
  description?: string;
}

export interface TargetInfo {
  name: string;
  classes: number[];
  class_labels: string[];
}

export interface FormSchemaResponse {
  numerical_features: NumericalFeature[];
  categorical_features: CategoricalFeature[];
  target: TargetInfo;
}

export interface SingleModelPrediction {
  model_name: string;
  model_key: string;
  prediction: number;
  prediction_label: string;
  confidence: number;
  inference_time_ms: number;
}

export interface ConsensusSummary {
  majority_prediction: number;
  majority_label: string;
  agreement_ratio: number;
  models_agreeing: number;
  models_total: number;
}

export interface PredictionResponse {
  predictions: SingleModelPrediction[];
  consensus: ConsensusSummary;
  input_echo: Record<string, any>;
}

export interface ModelMetricDetail {
  display_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  avg_inference_time_ms: number;
  model_file_size_kb: number;
  training_time_seconds: number;
}

export interface DatasetSizeInfo {
  total_rows: number;
  train_rows: number;
  test_rows: number;
  features: number;
  features_after_encoding: number;
}

export interface MetricsResponse {
  generated_at: string;
  dataset: string;
  dataset_size: DatasetSizeInfo;
  models: Record<string, ModelMetricDetail>;
  best_per_metric?: {
    highest_accuracy: string;
    highest_f1: string;
    fastest_inference: string;
    smallest_file: string;
  };
}

export interface HealthResponse {
  status: string;
  models_loaded: number;
  models?: string[];
  transformers_loaded?: boolean;
  uptime_seconds?: number;
  error?: string;
}
