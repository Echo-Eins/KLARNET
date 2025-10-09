fn main() {
    let proto = std::path::Path::new("../../proto/klarnet.proto");
    if !proto.exists() {
        println!("cargo:warning=proto/klarnet.proto missing; gRPC stubs will not be generated");
        return;
    }

    if std::env::var_os("PROTOC").is_none() {
        match protoc_bin_vendored::protoc_bin_path() {
            Ok(path) => std::env::set_var("PROTOC", path),
            Err(err) => panic!("failed to locate protoc binary for gRPC code generation: {err}"),
        }
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto.to_str().unwrap()], &["../../proto"])
        .expect("failed to build gRPC definitions");
}
