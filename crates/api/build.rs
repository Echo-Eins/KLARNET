fn main() {
    let proto = std::path::Path::new("../../proto/klarnet.proto");
    if !proto.exists() {
        println!("cargo:warning=proto/klarnet.proto missing; gRPC stubs will not be generated");
        return;
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(&[proto.to_str().unwrap()], &["../../proto"])
        .expect("failed to build gRPC definitions");
}
