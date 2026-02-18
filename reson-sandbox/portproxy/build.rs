use std::path::PathBuf;

fn set_protoc_env() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }
    let include = protoc_bin_vendored::include_path()?;
    unsafe {
        std::env::set_var("PROTOC_INCLUDE", &include);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_protoc_env()?;
    let proto_root = PathBuf::from("../proto");
    let proto_file = proto_root.join("bracket/portproxy/v1/portproxy.proto");

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_well_known_types(true)
        .compile_protos(&[proto_file], &[proto_root])?;

    println!("cargo:rerun-if-changed=../proto/bracket/portproxy/v1/portproxy.proto");
    Ok(())
}
