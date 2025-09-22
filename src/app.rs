// src/app.rs
use klarnet_audio_ingest::AudioIngest;
use klarnet_vad::{VadProcessor, VadConfig};
use klarnet_buffering::AudioBuffer;
use klarnet_whisper_stt::WhisperEngine;
use klarnet_nlu::NluEngine;
use klarnet_actions::ActionExecutor;
use klarnet_tts::TtsEngine;
use klarnet_api::ApiServer;
use klarnet_observability::MetricsCollector;

pub struct KlarnetApp {
    config: AppConfig,
    pipeline: AudioPipeline,
    api_server: Option<ApiServer>,
    metrics: Arc<MetricsCollector>,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub whisper: klarnet_whisper_stt::WhisperConfig,
    pub nlu: klarnet_nlu::NluConfig,
    pub api_enabled: bool,
    pub api_port: u16,
    pub pre_roll_ms: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            vad: VadConfig::default(),
            whisper: klarnet_whisper_stt::WhisperConfig::default(),
            nlu: klarnet_nlu::NluConfig::default(),
            api_enabled: true,
            api_port: 3000,
            pre_roll_ms: 1000,
        }
    }
}

impl KlarnetApp {
    pub async fn new(config: AppConfig) -> KlarnetResult<Self> {
        info!("Initializing KLARNET components...");

        let metrics = Arc::new(MetricsCollector::new());

        let pipeline = AudioPipeline::new(
            config.clone(),
            metrics.clone(),
        ).await?;

        let api_server = if config.api_enabled {
            Some(ApiServer::new(config.api_port, metrics.clone()).await?)
        } else {
            None
        };

        Ok(Self {
            config,
            pipeline,
            api_server,
            metrics,
        })
    }

    pub async fn run(&mut self) -> KlarnetResult<()> {
        info!("Starting KLARNET assistant...");

        // Start API server if enabled
        if let Some(server) = &self.api_server {
            tokio::spawn(async move {
                if let Err(e) = server.serve().await {
                    error!("API server error: {}", e);
                }
            });
        }

        // Start the audio pipeline
        self.pipeline.start().await?;

        // Wait for shutdown signal
        self.wait_for_shutdown().await?;

        // Cleanup
        self.pipeline.stop().await?;

        Ok(())
    }

    async fn wait_for_shutdown(&self) -> KlarnetResult<()> {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received shutdown signal");
                Ok(())
            }
            Err(e) => {
                error!("Failed to listen for shutdown signal: {}", e);
                Err(KlarnetError::Unknown(e.to_string()))
            }
        }
    }
}