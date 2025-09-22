// crates/tts/src/piper.rs

pub struct PiperTts {
    config: TtsConfig,
}

impl PiperTts {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        Ok(Self { config })
    }
}

#[async_trait]
impl TtsBackend for PiperTts {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>> {
        // Call Piper binary
        let output = Command::new("piper")
            .arg("--model")
            .arg(&self.config.model)
            .arg("--output_raw")
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .output()
            .await
            .map_err(|e| KlarnetError::Action(format!("Piper failed: {}", e)))?;

        Ok(output.stdout)
    }

    fn name(&self) -> &str {
        "Piper"
    }
}