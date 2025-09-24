// build.rs

fn main() {
    // Compile protobuf files
    #[cfg(feature = "grpc")]
    {
        let proto_path = std::path::Path::new("proto/klarnet.proto");
        if proto_path.exists() {
            tonic_build::configure()
                .build_server(true)
                .build_client(false)
                .compile(&["proto/klarnet.proto"], &["proto"])
                .expect("Failed to compile protobuf files");
        } else {
            println!("cargo:warning=proto/klarnet.proto not found; skipping gRPC code generation");
        }
    }

    // Set build-time environment variables
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", chrono::Utc::now());

    // Check for CUDA availability
    if std::process::Command::new("nvidia-smi").output().is_ok() {
        println!("cargo:rustc-cfg=has_cuda");
    }
}
