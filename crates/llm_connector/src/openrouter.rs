// crates/llm_connector/src/openrouter.rs

pub struct OpenRouterProvider {
    config: LlmConfig,
    client: Client,
    api_key: String,
}

impl OpenRouterProvider {
    pub async fn new(config: LlmConfig) -> KlarnetResult<Self> {
        let api_key = std::env::var(&config.api_key_env)
            .map_err(|_| KlarnetError::Nlu(format!("API key not found: {}", config.api_key_env)))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_s))
            .build()
            .map_err(|e| KlarnetError::Network(e.to_string()))?;

        Ok(Self {
            config,
            client,
            api_key,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn complete(&self, request: CompletionRequest) -> KlarnetResult<CompletionResponse> {
        let url = self.config.base_url.as_ref()
            .map(|u| format!("{}/chat/completions", u))
            .unwrap_or_else(|| "https://openrouter.ai/api/v1/chat/completions".to_string());

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://github.com/klarnet")
            .json(&json!({
                "model": self.config.model,
                "messages": request.messages,
                "max_tokens": request.max_tokens.unwrap_or(self.config.max_tokens),
                "temperature": request.temperature.unwrap_or(self.config.temperature),
                "top_p": request.top_p.unwrap_or(self.config.top_p),
            }))
            .send()
            .await
            .map_err(|e| KlarnetError::Network(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(KlarnetError::Nlu(format!("API error {}: {}", status, text)));
        }

        let json: Value = response.json().await
            .map_err(|e| KlarnetError::Nlu(format!("Failed to parse response: {}", e)))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = Usage {
            prompt_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
            total_tokens: json["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
        };

        Ok(CompletionResponse {
            content,
            function_call: None,
            usage,
        })
    }

    fn name(&self) -> &str {
        "OpenRouter"
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audio_buffer() {
        let buffer = klarnet_buffering::AudioBuffer::new(16000);

        let frame = klarnet_core::AudioFrame {
            data: Arc::from(vec![0.0f32; 1024].into_boxed_slice()),
            timestamp: chrono::Utc::now(),
            duration: Duration::from_millis(64),
            sample_rate: 16000,
        };

        buffer.add_frame(frame.clone());

        let pre_roll = buffer.start_segment();
        assert!(!pre_roll.is_empty());

        buffer.add_frame(frame);

        let chunk = buffer.end_segment();
        assert!(chunk.is_some());
    }

    #[tokio::test]
    async fn test_vad_detection() {
        let config = klarnet_vad::VadConfig::default();
        let (mut processor, mut rx) = klarnet_vad::VadProcessor::new(config).unwrap();

        // Simulate speech frame
        let frame = klarnet_core::AudioFrame {
            data: Arc::from(vec![0.1f32; 480].into_boxed_slice()),
            timestamp: chrono::Utc::now(),
            duration: Duration::from_millis(30),
            sample_rate: 16000,
        };

        processor.process_frame(frame).await.unwrap();

        // Check for VAD events
        if let Ok(event) = rx.try_recv() {
            match event {
                klarnet_core::VadEvent::SpeechStart { .. } => {
                    println!("Speech started");
                }
                klarnet_core::VadEvent::SpeechEnd { .. } => {
                    println!("Speech ended");
                }
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let nlu = klarnet_nlu::PatternMatcher::new(None).await.unwrap();

        let result = nlu.process("включи свет в гостиной").await.unwrap();

        assert!(result.intent.is_some());
        assert_eq!(result.intent.unwrap().name, "lights_on");
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = klarnet_buffering::ring_buffer::RingBuffer::<f32>::new(10);

        for i in 0..15 {
            buffer.push(i as f32);
        }

        assert_eq!(buffer.len(), 10);

        let last_5 = buffer.get_last_n(5);
        assert_eq!(last_5, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }
}

// Integration test example
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_pipeline() {
        // This would be a full end-to-end test
        // In production, use test fixtures and mock services

        let config = klarnet_core::AudioConfig::default();
        let _audio = klarnet_audio_ingest::AudioIngest::new(
            config,
            Duration::from_millis(1000),
        );

        // Test would continue with full pipeline setup
    }
}