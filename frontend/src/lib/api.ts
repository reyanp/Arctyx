/**
 * DataFoundry Backend API Client
 * 
 * This module provides typed wrappers for all DataFoundry Flask API endpoints.
 * Base URL is configured via VITE_BACKEND_URL environment variable.
 */

// ============================================================================
// Configuration
// ============================================================================

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface ApiResponse<T = any> {
  success?: boolean;
  message?: string;
  error?: string;
  traceback?: string;
  [key: string]: any;
}

export interface HealthResponse {
  status: string;
  version: string;
  features: {
    data_labeling: boolean;
    model_training: boolean;
    data_generation: boolean;
    anomaly_detection: boolean;
    agent_pipelines: boolean;
    orchestrator?: boolean;
  };
}

export interface UploadFileResponse {
  success: boolean;
  file_path: string;  // Absolute path (preferred for API use)
  relative_path: string;  // Relative path (for display)
  filename: string;
  original_filename: string;
  size_bytes: number;
  message: string;
}

export interface DatasetInfoResponse {
  num_rows: number;
  num_columns: number;
  columns: string[];
  dtypes: Record<string, string>;
  sample_data: Record<string, any>[];
  missing_values: Record<string, number>;
}

export interface LabelingFunction {
  name: string;
  code: string;
}

export interface CreateLabelsRequest {
  data_path: string;
  output_path: string;
  labeling_functions: LabelingFunction[];
}

export interface CreateLabelsResponse {
  success: boolean;
  message: string;
  output_path: string;
}

export interface TrainingParams {
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  [key: string]: any;
}

export interface ModelParams {
  latent_dim?: number;
  encoder_hidden_layers?: number[];
  decoder_hidden_layers?: number[];
  [key: string]: any;
}

export interface CreateConfigRequest {
  data_path: string;
  output_dir: string;
  model_type?: 'tabular_cvae' | 'mixed_data_cvae' | 'tabular_vae_gmm' | 'tabular_ctgan';
  model_params?: ModelParams;
  training_params?: TrainingParams;
}

export interface CreateConfigResponse {
  success: boolean;
  message: string;
  config_path: string;
  config: any;
}

export interface TrainModelRequest {
  config_path: string;
}

export interface TrainModelResponse {
  success: boolean;
  message: string;
  model_path: string;
  preprocessor_path: string;
  config_path: string;
}

export interface GenerateDataRequest {
  model_path: string;
  config_path: string;
  label: number;
  num_to_generate: number;
  output_path?: string;
  output_format?: 'parquet' | 'csv' | 'pt';
}

export interface GenerateDataResponse {
  success: boolean;
  message: string;
  output_path: string;
  num_generated: number;
  label: number;
}

export interface DetectAnomaliesRequest {
  config_path: string;
  model_path: string;
  preprocessor_path: string;
  data_to_scan_path: string;
  output_path?: string;
}

export interface DetectAnomaliesResponse {
  success: boolean;
  message: string;
  output_path: string;
  num_samples_scanned: number;
  anomaly_score_range: [number, number];
  mean_anomaly_score: number;
}

export interface FileListResponse {
  files: string[];
  directories: string[];
}

export interface ConvertToCsvRequest {
  parquet_path: string;
  output_path?: string;
}

export interface ConvertToCsvResponse {
  success: boolean;
  csv_path: string;
  num_rows: number;
  num_columns: number;
  columns: string[];
  message: string;
}

export interface ConvertToParquetRequest {
  csv_path: string;
  output_path?: string;
}

export interface ConvertToParquetResponse {
  success: boolean;
  parquet_path: string;
  num_rows: number;
  num_columns: number;
  columns: string[];
  message: string;
}

export interface AgentGenerateRequest {
  input_message: string;
  dataset_path?: string;
}

export interface AgentGenerateResponse {
  output: string;
  file_paths: {
    labeled_output_path: string | null;
    config_path: string | null;
    model_path: string | null;
    preprocessor_path: string | null;
    synthetic_output_path: string | null;
    anomaly_report_path: string | null;
  };
  steps_completed: string[];
  error?: string;
  message?: string;
}

// ============================================================================
// API Helper Functions
// ============================================================================

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T = any>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BACKEND_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    // Get response text first for better error debugging
    const responseText = await response.text();
    
    // Try to parse as JSON
    let data;
    try {
      data = JSON.parse(responseText);
    } catch (parseError) {
      console.error('Failed to parse JSON response:', responseText);
      console.error('Parse error:', parseError);
      throw new Error(`Invalid JSON response from ${endpoint}: ${parseError}`);
    }

    // Check for API errors
    if (data.error) {
      throw new Error(data.error);
    }

    return data as T;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`API request failed: ${String(error)}`);
  }
}

// ============================================================================
// API Client Functions
// ============================================================================

/**
 * Check if the backend API is healthy and available
 */
export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

/**
 * Upload a file to the backend
 */
export async function uploadFile(file: File): Promise<UploadFileResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${BACKEND_URL}/api/upload`, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - browser will set it with boundary for multipart/form-data
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(errorData.error || `Upload failed: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Get information about a dataset
 */
export async function getDatasetInfo(dataPath: string): Promise<DatasetInfoResponse> {
  return apiFetch<DatasetInfoResponse>('/api/dataset/info', {
    method: 'POST',
    body: JSON.stringify({ data_path: dataPath }),
  });
}

/**
 * Create labels for a dataset using weak supervision
 */
export async function createLabels(
  request: CreateLabelsRequest
): Promise<CreateLabelsResponse> {
  return apiFetch<CreateLabelsResponse>('/api/labeling/create-labels', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run the labeling pipeline (AI agent)
 */
export async function runLabelingPipeline(params: {
  user_goal: string;
  raw_data_path: string;
  hand_labeled_examples_path?: string;
  target_auc_score?: number;
  max_attempts?: number;
}): Promise<any> {
  return apiFetch('/api/labeling/run-pipeline', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Create a training configuration
 */
export async function createTrainingConfig(
  request: CreateConfigRequest
): Promise<CreateConfigResponse> {
  return apiFetch<CreateConfigResponse>('/api/training/create-config', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Train a model using a configuration file
 */
export async function trainModel(
  request: TrainModelRequest
): Promise<TrainModelResponse> {
  return apiFetch<TrainModelResponse>('/api/training/train-model', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run the training pipeline (AI agent)
 */
export async function runTrainingPipeline(params: {
  labeled_data_path: string;
  holdout_test_path?: string;
  target_utility_pct?: number;
  max_attempts?: number;
}): Promise<any> {
  return apiFetch('/api/training/run-pipeline', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Generate synthetic data from a trained model
 */
export async function generateData(
  request: GenerateDataRequest
): Promise<GenerateDataResponse> {
  return apiFetch<GenerateDataResponse>('/api/generation/generate-data', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run the generation pipeline (AI agent)
 */
export async function runGenerationPipeline(params: {
  model_path: string;
  config_path: string;
  label: number;
  num_to_generate: number;
}): Promise<any> {
  return apiFetch('/api/generation/run-pipeline', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Detect anomalies in a dataset
 */
export async function detectAnomalies(
  request: DetectAnomaliesRequest
): Promise<DetectAnomaliesResponse> {
  return apiFetch<DetectAnomaliesResponse>('/api/cleaning/detect-anomalies', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run the anomaly detection pipeline (AI agent)
 */
export async function runAnomalyPipeline(params: {
  config_path: string;
  model_path: string;
  preprocessor_path: string;
  data_to_scan_path: string;
}): Promise<any> {
  return apiFetch('/api/cleaning/run-pipeline', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * List files in the backend directory
 */
export async function listFiles(): Promise<FileListResponse> {
  return apiFetch<FileListResponse>('/api/files/list');
}

/**
 * Get the download URL for a file
 */
export function getFileDownloadUrl(filePath: string): string {
  return `${BACKEND_URL}/api/files/download?path=${encodeURIComponent(filePath)}`;
}

/**
 * Download a file from the backend
 */
export async function downloadFile(filePath: string): Promise<Blob> {
  const url = getFileDownloadUrl(filePath);
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to download file: ${response.statusText}`);
  }
  
  return response.blob();
}

/**
 * Convert a Parquet file to CSV format
 */
export async function convertToCsv(
  request: ConvertToCsvRequest
): Promise<ConvertToCsvResponse> {
  return apiFetch<ConvertToCsvResponse>('/api/files/convert-to-csv', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Convert a CSV file to Parquet format
 */
export async function convertToParquet(
  request: ConvertToParquetRequest
): Promise<ConvertToParquetResponse> {
  return apiFetch<ConvertToParquetResponse>('/api/files/convert-to-parquet', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run the agent orchestrator with natural language input
 */
export async function runAgentGenerate(
  request: AgentGenerateRequest
): Promise<AgentGenerateResponse> {
  return apiFetch<AgentGenerateResponse>('/api/agent/generate', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ============================================================================
// Export the base URL for direct use if needed
// ============================================================================

export { BACKEND_URL };

