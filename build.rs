// build.rs

fn main() {
    // Compile protobuf files
    #[cfg(feature = "grpc")]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(false)
            .compile(&["proto/klarnet.proto"], &["proto"])
            .expect("Failed to compile protobuf files");
    }

    // Set build-time environment variables
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", chrono::Utc::now());

    // Check for CUDA availability
    if std::process::Command::new("nvidia-smi")
        .output()
        .is_ok()
    {
        println!("cargo:rustc-cfg=has_cuda");
    }

    // Link audio libraries
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=asound");
        println!("cargo:rustc-link-lib=portaudio");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=ole32");
    }
}