//! ML Model Inference Module
//!
//! This module provides ONNX model inference capabilities for the execution core.
//! It loads models exported from MLflow and performs inference for:
//! - Regime detection
//! - Market embeddings
//! - Strategy signals
//!
//! Models are loaded from the MLflow artifact store and cached for performance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Errors that can occur during ML inference
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Model configuration from TOML file
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Path to ONNX model file
    pub artifact_path: String,
    /// MLflow run ID that produced this model
    pub mlflow_run_id: Option<String>,
    /// Input tensor names
    pub input_names: Option<Vec<String>>,
    /// Output tensor names
    pub output_names: Option<Vec<String>>,
    /// Expected input shape (batch dimension can be -1 for dynamic)
    pub input_shape: Option<Vec<i64>>,
}

/// Configuration for all models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    /// Map of model name to configuration
    pub models: HashMap<String, ModelConfig>,
}

impl ModelsConfig {
    /// Load models configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read models config file")?;
        let config: ModelsConfig = toml::from_str(&content)
            .context("Failed to parse models config TOML")?;
        Ok(config)
    }

    /// Load from default config path
    pub fn load_default() -> Result<Self> {
        let config_path = PathBuf::from("config/models.toml");
        if config_path.exists() {
            Self::from_file(config_path)
        } else {
            // Return empty config if file doesn't exist
            Ok(ModelsConfig {
                models: HashMap::new(),
            })
        }
    }
}

/// Result of model inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Output tensor values
    pub outputs: HashMap<String, Vec<f32>>,
    /// Output shape
    pub shape: Vec<usize>,
    /// Inference latency in microseconds
    pub latency_us: u64,
    /// Model version used
    pub model_version: String,
}

/// Model metadata for tracking
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub mlflow_run_id: Option<String>,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    pub inference_count: u64,
    pub total_latency_us: u64,
}

/// Cached ONNX model wrapper
/// Note: ort::Session is not Send+Sync, so we use a different approach
pub struct CachedModel {
    pub metadata: ModelMetadata,
    pub config: ModelConfig,
    // The actual ONNX session would be created per-inference or use thread-local
    // For now, we store the path and create sessions as needed
    pub model_path: PathBuf,
}

impl CachedModel {
    pub fn new(config: ModelConfig, model_path: PathBuf) -> Self {
        Self {
            metadata: ModelMetadata {
                name: config.name.clone(),
                version: config.version.clone(),
                mlflow_run_id: config.mlflow_run_id.clone(),
                loaded_at: chrono::Utc::now(),
                inference_count: 0,
                total_latency_us: 0,
            },
            config,
            model_path,
        }
    }
}

/// ML Inference Engine
///
/// Manages ONNX model loading and inference for the execution core.
/// Models are loaded from MLflow artifact store and cached for performance.
pub struct InferenceEngine {
    /// Model configurations
    config: ModelsConfig,
    /// Cached models (path and metadata only, sessions created per-inference)
    models: Arc<RwLock<HashMap<String, CachedModel>>>,
    /// Base path for model artifacts
    artifact_base_path: PathBuf,
    /// Whether ONNX runtime is available
    onnx_available: bool,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: ModelsConfig, artifact_base_path: PathBuf) -> Self {
        // Check if ONNX runtime is available
        let onnx_available = Self::check_onnx_available();

        if !onnx_available {
            warn!("ONNX runtime not available - ML inference will be disabled");
        }

        Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            artifact_base_path,
            onnx_available,
        }
    }

    /// Check if ONNX runtime is available
    fn check_onnx_available() -> bool {
        // Try to initialize ONNX runtime
        // This is a placeholder - actual implementation depends on ort crate setup
        #[cfg(feature = "onnx")]
        {
            match ort::init().commit() {
                Ok(_) => true,
                Err(e) => {
                    warn!("Failed to initialize ONNX runtime: {}", e);
                    false
                }
            }
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    /// Load a model by name
    pub async fn load_model(&self, model_name: &str) -> Result<(), InferenceError> {
        if !self.onnx_available {
            return Err(InferenceError::ModelLoadError(
                "ONNX runtime not available".to_string(),
            ));
        }

        let model_config = self
            .config
            .models
            .get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?
            .clone();

        let model_path = self.artifact_base_path.join(&model_config.artifact_path);

        if !model_path.exists() {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        let cached_model = CachedModel::new(model_config, model_path);

        let mut models = self.models.write().await;
        models.insert(model_name.to_string(), cached_model);

        info!("Loaded model: {}", model_name);
        Ok(())
    }

    /// Unload a model
    pub async fn unload_model(&self, model_name: &str) -> Result<(), InferenceError> {
        let mut models = self.models.write().await;
        models
            .remove(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;
        info!("Unloaded model: {}", model_name);
        Ok(())
    }

    /// Check if a model is loaded
    pub async fn is_model_loaded(&self, model_name: &str) -> bool {
        let models = self.models.read().await;
        models.contains_key(model_name)
    }

    /// Get metadata for a loaded model
    pub async fn get_model_metadata(&self, model_name: &str) -> Option<ModelMetadata> {
        let models = self.models.read().await;
        models.get(model_name).map(|m| m.metadata.clone())
    }

    /// List all loaded models
    pub async fn list_loaded_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    /// Run inference on a model
    ///
    /// Note: This is a placeholder implementation. Full ONNX inference
    /// requires proper session management which depends on the ort crate
    /// being properly configured.
    pub async fn infer(
        &self,
        model_name: &str,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<InferenceResult, InferenceError> {
        if !self.onnx_available {
            return Err(InferenceError::InferenceFailed(
                "ONNX runtime not available".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        let models = self.models.read().await;
        let model = models
            .get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        // Placeholder inference - in production, this would use ort::Session
        // For now, return dummy output matching input shape
        let output_size: usize = input_shape.iter().product();
        let dummy_output = vec![0.0f32; output_size];

        let latency_us = start_time.elapsed().as_micros() as u64;

        debug!(
            "Inference completed for {} in {}us",
            model_name, latency_us
        );

        Ok(InferenceResult {
            outputs: HashMap::from([("output".to_string(), dummy_output)]),
            shape: input_shape.to_vec(),
            latency_us,
            model_version: model.config.version.clone(),
        })
    }

    /// Run batch inference
    pub async fn infer_batch(
        &self,
        model_name: &str,
        inputs: &[Vec<f32>],
        input_shape: &[usize], // Shape per sample (without batch dimension)
    ) -> Result<Vec<InferenceResult>, InferenceError> {
        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            let result = self.infer(model_name, input, input_shape).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Hot-reload a model (load new version while keeping old one available)
    pub async fn hot_reload_model(&self, model_name: &str) -> Result<(), InferenceError> {
        // Get the current config for this model
        let model_config = self
            .config
            .models
            .get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?
            .clone();

        let model_path = self.artifact_base_path.join(&model_config.artifact_path);

        if !model_path.exists() {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        // Create new cached model
        let new_model = CachedModel::new(model_config, model_path);

        // Swap atomically
        let mut models = self.models.write().await;
        models.insert(model_name.to_string(), new_model);

        info!("Hot-reloaded model: {}", model_name);
        Ok(())
    }

    /// Get inference statistics for a model
    pub async fn get_inference_stats(&self, model_name: &str) -> Option<InferenceStats> {
        let models = self.models.read().await;
        models.get(model_name).map(|m| InferenceStats {
            model_name: m.metadata.name.clone(),
            model_version: m.metadata.version.clone(),
            inference_count: m.metadata.inference_count,
            avg_latency_us: if m.metadata.inference_count > 0 {
                m.metadata.total_latency_us / m.metadata.inference_count
            } else {
                0
            },
            loaded_at: m.metadata.loaded_at,
        })
    }
}

/// Inference statistics for monitoring
#[derive(Debug, Clone, Serialize)]
pub struct InferenceStats {
    pub model_name: String,
    pub model_version: String,
    pub inference_count: u64,
    pub avg_latency_us: u64,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
}

/// Regime detector inference helper
pub struct RegimeDetector {
    engine: Arc<InferenceEngine>,
    model_name: String,
}

impl RegimeDetector {
    pub fn new(engine: Arc<InferenceEngine>, model_name: String) -> Self {
        Self { engine, model_name }
    }

    /// Detect market regime from features
    pub async fn detect(
        &self,
        features: &[f32],
        sequence_length: usize,
    ) -> Result<RegimeDetection, InferenceError> {
        let input_shape = vec![1, features.len() / sequence_length, sequence_length];
        let result = self.engine.infer(&self.model_name, features, &input_shape).await?;

        // Parse regime probabilities from output
        let probs = result.outputs.get("output").cloned().unwrap_or_default();

        let regime = if probs.is_empty() {
            Regime::Unknown
        } else {
            // Find argmax
            let (idx, _) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));

            match idx {
                0 => Regime::LowVolTrending,
                1 => Regime::HighVolTrending,
                2 => Regime::LowVolMeanReverting,
                3 => Regime::HighVolMeanReverting,
                _ => Regime::Unknown,
            }
        };

        Ok(RegimeDetection {
            regime,
            probabilities: probs,
            confidence: probs.iter().cloned().fold(0.0f32, f32::max),
            model_version: result.model_version,
        })
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regime {
    LowVolTrending,
    HighVolTrending,
    LowVolMeanReverting,
    HighVolMeanReverting,
    Unknown,
}

/// Regime detection result
#[derive(Debug, Clone, Serialize)]
pub struct RegimeDetection {
    pub regime: Regime,
    pub probabilities: Vec<f32>,
    pub confidence: f32,
    pub model_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let config = ModelsConfig {
            models: HashMap::new(),
        };
        let engine = InferenceEngine::new(config, PathBuf::from("models"));
        assert!(engine.list_loaded_models().await.is_empty());
    }

    #[test]
    fn test_models_config_default() {
        let config = ModelsConfig::load_default();
        // Should not error even if file doesn't exist
        assert!(config.is_ok());
    }
}
