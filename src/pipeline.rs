// src/pipeline.rs
pub struct AudioPipeline {
    audio_ingest: AudioIngest,
    vad: VadProcessor,
    buffer: AudioBuffer,
    whisper: WhisperEngine,
    nlu: NluEngine,
    actions: ActionExecutor,
    tts: Option<TtsEngine>,
    metrics: Arc<MetricsCollector>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl AudioPipeline {
    pub async fn new(
        config: AppConfig,
        metrics: Arc<MetricsCollector>,
    ) -> KlarnetResult<Self> {
        let audio_ingest = AudioIngest::new(
            config.audio.clone(),
            Duration::from_millis(config.pre_roll_ms),
        )?;

        let (vad, vad_rx) = VadProcessor::new(config.vad.clone())?;
        let buffer = AudioBuffer::new(config.audio.sample_rate);
        let whisper = WhisperEngine::new(config.whisper.clone()).await?;
        let nlu = NluEngine::new(config.nlu.clone()).await?;
        let actions = ActionExecutor::new().await?;
        let tts = if config.tts_enabled {
            Some(TtsEngine::new(config.tts.clone()).await?)
        } else {
            None
        };

        Ok(Self {
            audio_ingest,
            vad,
            buffer,
            whisper,
            nlu,
            actions,
            tts,
            metrics,
            shutdown_tx: None,
        })
    }

    pub async fn start(&mut self) -> KlarnetResult<()> {
        info!("Starting audio pipeline");

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Start audio capture
        let mut audio_rx = self.audio_ingest.start().await?;

        // Start VAD processing in background task
        let vad = self.vad.clone();
        let vad_handle = tokio::spawn(async move {
            while let Some(frame) = audio_rx.recv().await {
                if let Err(e) = vad.process_frame(frame).await {
                    warn!("VAD processing error: {}", e);
                }
            }
        });

        // Start speech processing pipeline
        let whisper = self.whisper.clone();
        let nlu = self.nlu.clone();
        let actions = self.actions.clone();
        let tts = self.tts.clone();
        let buffer = self.buffer.clone();

        let pipeline_handle = tokio::spawn(async move {
            let mut vad_rx = vad_rx;
            let mut speech_buffer = Vec::new();
            let mut is_recording = false;

            while let Some(event) = vad_rx.recv().await {
                match event {
                    VadEvent::SpeechStart { timestamp, .. } => {
                        info!("Speech detected at {:?}", timestamp);
                        is_recording = true;
                        speech_buffer.clear();
                    }

                    VadEvent::SpeechFrame { pcm, .. } => {
                        if is_recording {
                            speech_buffer.extend_from_slice(&pcm);
                        }
                    }

                    VadEvent::SpeechEnd { duration, .. } => {
                        if is_recording && !speech_buffer.is_empty() {
                            info!("Speech ended, duration: {:?}", duration);
                            is_recording = false;

                            // Create audio chunk
                            let chunk = AudioChunk::from_pcm(
                                &speech_buffer,
                                16000,
                            );

                            // Transcribe
                            match whisper.transcribe(chunk).await {
                                Ok(transcript) => {
                                    info!("Transcript: {}", transcript.full_text);

                                    // Process with NLU
                                    match nlu.process(&transcript).await {
                                        Ok(result) => {
                                            info!("Intent: {:?}", result.intent);

                                            // Execute action
                                            if let Some(action) = result.to_action() {
                                                if let Err(e) = actions.execute(action).await {
                                                    error!("Action execution failed: {}", e);
                                                }
                                            }

                                            // Generate TTS response if available
                                            if let Some(tts) = &tts {
                                                if let Some(response) = result.response {
                                                    if let Err(e) = tts.speak(&response).await {
                                                        error!("TTS failed: {}", e);
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => error!("NLU processing failed: {}", e),
                                    }
                                }
                                Err(e) => error!("Transcription failed: {}", e),
                            }

                            speech_buffer.clear();
                        }
                    }

                    _ => {}
                }
            }
        });

        // Wait for shutdown or task completion
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("Pipeline shutdown requested");
            }
            _ = vad_handle => {
                warn!("VAD task completed unexpectedly");
            }
            _ = pipeline_handle => {
                warn!("Pipeline task completed unexpectedly");
            }
        }

        Ok(())
    }

    pub async fn stop(&mut self) -> KlarnetResult<()> {
        info!("Stopping audio pipeline");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        self.audio_ingest.stop().await?;

        Ok(())
    }
}