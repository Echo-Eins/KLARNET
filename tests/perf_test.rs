// Performance test (tests/perf_test.rs)

#[cfg(test)]
mod perf_tests {
    use std::time::Instant;
    use klarnet_core::*;

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored
    async fn test_stt_performance() {
        let config = klarnet_whisper_stt::WhisperConfig::default();
        let engine = klarnet_whisper_stt::WhisperEngine::new(config).await
            .expect("Failed to create engine");

        let test_sizes = vec![
            (16000, "1 second"),
            (160000, "10 seconds"),
            (480000, "30 seconds"),
        ];

        for (samples, label) in test_sizes {
            let audio = vec![0.0f32; samples];
            let chunk = AudioChunk::from_pcm(&audio, 16000);

            let start = Instant::now();
            let _ = engine.transcribe(chunk).await;
            let elapsed = start.elapsed();

            let rtf = elapsed.as_secs_f64() / (samples as f64 / 16000.0);
            println!("{}: {:.2}ms (RTF: {:.3})", label, elapsed.as_millis(), rtf);

            // Assert performance requirements
            assert!(rtf < 0.1, "RTF too high: {}", rtf);
        }
    }

    #[test]
    fn test_memory_usage() {
        use klarnet_buffering::AudioBuffer;

        let buffer = AudioBuffer::new(16000);

        // Add 1 minute of audio
        for _ in 0..60 {
            let frame = AudioFrame {
                data: Arc::from(vec![0.0f32; 16000].into_boxed_slice()),
                timestamp: chrono::Utc::now(),
                duration: Duration::from_secs(1),
                sample_rate: 16000,
            };
            buffer.add_frame(frame);
        }

        let metrics = buffer.get_metrics();

        // Check memory usage is reasonable (< 100MB for 1 minute)
        assert!(metrics.peak_buffer_size_bytes < 100_000_000);
    }
}