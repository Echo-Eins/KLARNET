// crates/api/src/grpc.rs

use tonic::{transport::Server, Request, Response, Status};

pub mod klarnet_proto {
    tonic::include_proto!("klarnet");
}

use klarnet_proto::{
    stt_service_server::{SttService, SttServiceServer},
    TranscribeRequest, TranscribeResponse,
    StreamRequest, StreamResponse,
};

pub struct GrpcService {
    handlers: Arc<ApiHandlers>,
}

impl GrpcService {
    pub fn new(handlers: Arc<ApiHandlers>) -> Self {
        Self { handlers }
    }

    pub async fn serve(self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("0.0.0.0:{}", port).parse()?;

        info!("gRPC server listening on {}", addr);

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

        match self.handlers.transcribe_file(Bytes::from(req.audio_data)).await {
            Ok(transcript) => {
                let response = TranscribeResponse {
                    text: transcript.full_text,
                    language: transcript.language,
                    confidence: 0.95,
                };

                Ok(Response::new(response))
            }
            Err(e) => {
                Err(Status::internal(format!("Transcription failed: {}", e)))
            }
        }
    }

    type StreamTranscribeStream = tokio_stream::wrappers::ReceiverStream<Result<StreamResponse, Status>>;

    async fn stream_transcribe(
        &self,
        request: Request<tonic::Streaming<StreamRequest>>,
    ) -> Result<Response<Self::StreamTranscribeStream>, Status> {
        let mut stream = request.into_inner();
        let handlers = self.handlers.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(async move {
            let mut audio_buffer = Vec::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        audio_buffer.extend_from_slice(&req.audio_chunk);

                        // Process when we have enough data
                        if audio_buffer.len() >= 32000 {
                            let chunk_data = audio_buffer.clone();
                            audio_buffer.clear();

                            // Convert and transcribe
                            let pcm: Vec<f32> = chunk_data.chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                                .collect();

                            if let Some(whisper) = &handlers.whisper {
                                let chunk = AudioChunk::from_pcm(&pcm, 16000);

                                match whisper.transcribe(chunk).await {
                                    Ok(transcript) => {
                                        let response = StreamResponse {
                                            text: transcript.full_text,
                                            is_final: false,
                                        };

                                        let _ = tx.send(Ok(response)).await;
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            tokio_stream::wrappers::ReceiverStream::new(rx)
        ))
    }
}