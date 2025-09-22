// crates/api/src/websocket.rs

pub async fn handle_stt_stream(socket: WebSocket, handlers: Arc<ApiHandlers>) {
    let session_id = {
        let mut sessions = handlers.active_sessions.write();
        sessions.create_session()
    };

    info!("WebSocket STT session started: {}", session_id);

    let (mut sender, mut receiver) = socket.split();

    // Send welcome message
    let welcome = json!({
        "type": "welcome",
        "session_id": session_id,
        "sample_rate": 16000,
        "format": "pcm_f32",
    });

    if sender.send(Message::Text(welcome.to_string())).await.is_err() {
        return;
    }

    // Process incoming audio
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Binary(data)) => {
                if let Err(e) = process_audio_chunk(&handlers, &session_id, data, &mut sender).await {
                    error!("Error processing audio chunk: {}", e);
                    break;
                }
            }
            Ok(Message::Text(text)) => {
                if let Err(e) = handle_control_message(&text, &mut sender).await {
                    error!("Error handling control message: {}", e);
                    break;
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket closed by client");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    info!("WebSocket STT session ended: {}", session_id);
}

async fn process_audio_chunk(
    handlers: &Arc<ApiHandlers>,
    session_id: &uuid::Uuid,
    data: Vec<u8>,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> KlarnetResult<()> {
    // Add audio to session buffer
    let should_process = {
        let mut sessions = handlers.active_sessions.write();
        if let Some(session) = sessions.get_session_mut(session_id) {
            // Convert bytes to f32 samples
            let samples: Vec<f32> = data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            session.audio_buffer.extend(&samples);

            // Process if we have enough data (e.g., 2 seconds)
            session.audio_buffer.len() >= 32000
        } else {
            return Err(klarnet_core::KlarnetError::Unknown("Session not found".to_string()));
        }
    };

    if should_process {
        let audio_data = {
            let mut sessions = handlers.active_sessions.write();
            if let Some(session) = sessions.get_session_mut(session_id) {
                let data = session.audio_buffer.clone();
                session.audio_buffer.clear();
                data
            } else {
                return Ok(());
            }
        };

        // Transcribe audio
        if let Some(whisper) = &handlers.whisper {
            let chunk = AudioChunk::from_pcm(&audio_data, 16000);

            match whisper.transcribe(chunk).await {
                Ok(transcript) => {
                    let response = json!({
                        "type": "transcript",
                        "text": transcript.full_text,
                        "segments": transcript.segments,
                        "is_final": false,
                    });

                    let _ = sender.send(Message::Text(response.to_string())).await;
                }
                Err(e) => {
                    let error_response = json!({
                        "type": "error",
                        "message": e.to_string(),
                    });

                    let _ = sender.send(Message::Text(error_response.to_string())).await;
                }
            }
        }
    }

    Ok(())
}

async fn handle_control_message(
    text: &str,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> KlarnetResult<()> {
    let msg: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| klarnet_core::KlarnetError::Serialization(e))?;

    match msg["type"].as_str() {
        Some("ping") => {
            let pong = json!({
                "type": "pong",
                "timestamp": chrono::Utc::now(),
            });
            sender.send(Message::Text(pong.to_string())).await
                .map_err(|e| klarnet_core::KlarnetError::Network(e.to_string()))?;
        }
        Some("end_stream") => {
            let response = json!({
                "type": "stream_ended",
                "timestamp": chrono::Utc::now(),
            });
            sender.send(Message::Text(response.to_string())).await
                .map_err(|e| klarnet_core::KlarnetError::Network(e.to_string()))?;
        }
        _ => {
            debug!("Unknown control message type: {:?}", msg["type"]);
        }
    }

    Ok(())
}