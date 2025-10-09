use std::time::Duration;
mod processor;
mod resampler;
mod source;

use klarnet_core::{AudioConfig, AudioFrame, KlarnetError, KlarnetResult};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

#[cfg(feature = "hardware")]
use crate::source::MicrophoneSource;
use crate::source::{AudioSource, StubSource};
use processor::AudioProcessor;
use resampler::Resampler;

pub struct AudioIngest {
    config: AudioConfig,
    pre_roll_duration: Duration,
    pre_roll_capacity: usize,
    pre_roll_buffer: Arc<Mutex<VecDeque<f32>>>,
    source: Box<dyn AudioSource>,
    worker: Option<JoinHandle<()>>,
    stop_signal: Option<oneshot::Sender<()>>,
    tx: Option<mpsc::UnboundedSender<AudioFrame>>,
    processor: Arc<AudioProcessor>,
    resampler: Arc<Resampler>,
}

impl AudioIngest {
    pub fn new(config: AudioConfig, pre_roll_duration: Duration) -> KlarnetResult<Self> {
        let source = Self::select_source()?;
        let pre_roll_capacity = Self::calculate_pre_roll_capacity(&config, pre_roll_duration);
        let processor = Arc::new(AudioProcessor::new(config.clone()));
        let resampler = Arc::new(Resampler::new(config.sample_rate));
        Ok(Self {
            config,
            pre_roll_duration,
            pre_roll_capacity,
            pre_roll_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(pre_roll_capacity))),
            source,
            worker: None,
            stop_signal: None,
            tx: None,
            processor,
            resampler,
        })
    }

    pub async fn start(&mut self) -> KlarnetResult<mpsc::UnboundedReceiver<AudioFrame>> {
        if self.worker.is_some() {
            return Err(KlarnetError::Audio(
                "Audio ingest already started".to_string(),
            ));
        }

        let source_name = self.source.name().to_string();
        let (client_tx, client_rx) = mpsc::unbounded_channel();
        self.tx = Some(client_tx.clone());
        let worker_tx = client_tx;
        let (internal_tx, internal_rx) = mpsc::unbounded_channel();
        let (stop_tx, stop_rx) = oneshot::channel();
        self.stop_signal = Some(stop_tx);
        let pre_roll_buffer = Arc::clone(&self.pre_roll_buffer);
        let pre_roll_capacity = self.pre_roll_capacity;
        let processor = Arc::clone(&self.processor);
        let resampler = Arc::clone(&self.resampler);

        if let Err(err) = self.source.start(internal_tx, self.config.clone()).await {
            self.tx.take();
            self.stop_signal.take();
            return Err(err);
        }

        let worker = tokio::spawn(async move {
            Self::run_worker(
                internal_rx,
                worker_tx,
                processor,
                resampler,
                pre_roll_buffer,
                pre_roll_capacity,
                stop_rx,
                source_name,
            ).await;
        });

        self.worker = Some(worker);

        info!(
            sample_rate = self.config.sample_rate,
            pre_roll_ms = self.pre_roll_duration.as_millis(),
            source = %self.source.name(),
            "Audio ingest started"
        );

        Ok(client_rx)
    }

    pub async fn stop(&mut self) -> KlarnetResult<()> {
        if self.worker.is_none() {
            return Ok(());
        }

        if let Some(stop_tx) = self.stop_signal.take() {
            let _ = stop_tx.send(());
        }

        let source_name = self.source.name().to_string();
        let stop_result = self.source.stop().await;

        if let Some(handle) = self.worker.take() {
            if let Err(err) = handle.await {
                if err.is_cancelled() {
                    warn!("Audio ingest worker task was cancelled");
                } else {
                    error!("Audio ingest worker task failed: {err}");
                }
            }
        }

        self.tx.take();

        info!(source = %source_name, "Audio ingest stopped");
        stop_result
    }

    pub fn get_pre_roll(&self) -> Vec<f32> {
        match self.pre_roll_buffer.lock() {
            Ok(buffer) => buffer.iter().copied().collect(),
            Err(_) => Vec::new(),
        }
    }

    fn calculate_pre_roll_capacity(config: &AudioConfig, duration: Duration) -> usize {
        let samples = (duration.as_secs_f64() * config.sample_rate as f64 * config.channels as f64)
            .round() as usize;
        let minimum = (config.buffer_size * config.channels as usize).max(1);
        samples.max(minimum)
    }

    fn select_source() -> KlarnetResult<Box<dyn AudioSource>> {
        #[cfg(feature = "hardware")]
        {
            match MicrophoneSource::new() {
                Ok(source) => {
                    info!("Hardware audio source initialised");
                    return Ok(Box::new(source));
                }
                Err(err) => {
                    warn!("Falling back to stub audio source: {err}");
                }
            }
        }

        info!("Using stub audio source");
        Ok(Box::new(StubSource::new()))
    }

    async fn run_worker(
        mut source_rx: mpsc::UnboundedReceiver<AudioFrame>,
        tx: mpsc::UnboundedSender<AudioFrame>,
        processor: Arc<AudioProcessor>,
        resampler: Arc<Resampler>,
        pre_roll: Arc<Mutex<VecDeque<f32>>>,
        capacity: usize,
        mut stop_rx: oneshot::Receiver<()>,
        source_name: String,
    ) {
        loop {
            tokio::select! {
                _ = &mut stop_rx => {
                    info!(source = %source_name, "Audio ingest worker stopping");
                    break;
                }
                maybe_frame = source_rx.recv() => {
                    match maybe_frame {
                        Some(frame) => {
                            let processed = Self::process_frame(&processor, &resampler, frame);

                            if let Some((frame, pcm)) = processed {
                                if let Ok(mut buffer) = pre_roll.lock() {
                                    for sample in pcm.iter().copied() {
                                        if buffer.len() >= capacity {
                                            buffer.pop_front();
                                        }
                                        buffer.push_back(sample);
                                    }
                                }
                            if let Err(err) = tx.send(frame) {
                                    warn!("Audio frame receiver dropped: {err}");
                                    break;
                                }
                            } else {
                                warn!("Dropping malformed audio frame from {source_name}");
                            }
                        }
                        None => {
                            warn!(source = %source_name, "Audio source stream ended unexpectedly");
                            break;
                        }
                    }
                }
            }
        }
    }
    fn process_frame(
        processor: &AudioProcessor,
        resampler: &Resampler,
        frame: AudioFrame,
    ) -> Option<(AudioFrame, Arc<[f32]>)> {
        let pcm: Vec<f32> = frame.data.iter().copied().collect();
        let resampled = resampler.resample(&pcm, frame.sample_rate);
        let processed = match processor.process(&resampled) {
            Ok(data) => data,
            Err(err) => {
                warn!("Audio processor error: {err}");
                return None;
            }
        };
        let normalized = processor.normalize(&processed);
        let arc_data: Arc<[f32]> = Arc::from(normalized.into_boxed_slice());

        let duration =
            Duration::from_secs_f32(arc_data.len() as f32 / resampler.target_rate() as f32);

        let frame = AudioFrame {
            data: Arc::clone(&arc_data),
            timestamp: frame.timestamp,
            duration,
            sample_rate: resampler.target_rate(),
        };

        Some((frame, arc_data))
    }
}
