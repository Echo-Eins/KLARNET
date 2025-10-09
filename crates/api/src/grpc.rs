use std::net::SocketAddr;
use std::sync::Arc;

use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::info;

use crate::handlers::{build_chunk_from_pcm, decode_pcm_s16le, ApiHandlers};

pub mod klarnet_proto {
    tonic::include_proto!("klarnet");
}

use klarnet_proto::stt_service_server::{SttService, SttServiceServer};
use klarnet_proto::{StreamRequest, StreamResponse, TranscribeRequest, TranscribeResponse};

pub struct GrpcService {
    handlers: Arc<ApiHandlers>,
    sample_rate: u32,
}

impl GrpcService {
    pub fn new(handlers: Arc<ApiHandlers>, sample_rate: u32) -> Self {
        Self {
            handlers,
            sample_rate,
        }
    }

    pub async fn serve(
        self,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(%addr, "Starting gRPC STT service");
        Server::builder()
            .add_service(SttServiceServer::new(self))
            .serve(addr)
            .await?;
        Ok(())
    }
}

#[tonic::async_trait]
impl SttService for GrpcService {
    async fn transcribe(
        &self,
        request: Request<TranscribeRequest>,
    ) -> Result<Response<TranscribeResponse>, Status> {
        let req = request.into_inner();
        let pcm = decode_pcm_s16le(&req.audio_data)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let chunk = build_chunk_from_pcm(&pcm, self.sample_rate);

        let whisper = self
            .handlers
            .whisper_engine()
            .ok_or_else(|| Status::unimplemented("Streaming STT is not configured"))?;

        let mut engine = whisper.lock().await;
        let transcript = engine
            .transcribe(chunk)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;

        Ok(Response::new(TranscribeResponse {
            text: transcript.full_text,
            language: transcript.language,
            confidence: 0.95,
        }))
    }

    type StreamTranscribeStream = ReceiverStream<Result<StreamResponse, Status>>;

    async fn stream_transcribe(
        &self,
        request: Request<tonic::Streaming<StreamRequest>>,
    ) -> Result<Response<Self::StreamTranscribeStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let whisper = self
            .handlers
            .whisper_engine()
            .ok_or_else(|| Status::unimplemented("Streaming STT is not configured"))?;
        let sample_rate = self.sample_rate;

        tokio::spawn(async move {
            let mut buffer = Vec::new();
            while let Some(message) = stream.next().await {
                match message {
                    Ok(req) => {
                        buffer.extend_from_slice(&req.audio_chunk);
                        if buffer.len() >= 2 * crate::handlers::STREAM_CHUNK_SAMPLES {
                            match decode_pcm_s16le(&buffer) {
                                Ok(pcm) => {
                                    buffer.clear();
                                    let chunk = build_chunk_from_pcm(&pcm, sample_rate);
                                    let mut engine = whisper.lock().await;
                                    match engine.transcribe(chunk).await {
                                        Ok(transcript) => {
                                            let response = StreamResponse {
                                                text: transcript.full_text,
                                                is_final: false,
                                            };
                                            let _ = tx.send(Ok(response)).await;
                                        }
                                        Err(err) => {
                                            let _ = tx
                                                .send(Err(Status::internal(err.to_string())))
                                                .await;
                                        }
                                    }
                                }
                                Err(err) => {
                                    let _ = tx
                                        .send(Err(Status::invalid_argument(err.to_string())))
                                        .await;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx.send(Err(Status::from_error(Box::new(err)))).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
