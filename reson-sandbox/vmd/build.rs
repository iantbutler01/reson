// @dive-file: Build-time protobuf generation for vmd service and portproxy client stubs.
// @dive-rel: Produces generated modules consumed by vmd/src/lib.rs and control-bus exec command handling.
// @dive-rel: Uses vendored protoc so CI and local builds resolve well-known types consistently.
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
    let vmd_proto = proto_root.join("bracket/vmd/v1/vmd.proto");
    let portproxy_proto = proto_root.join("bracket/portproxy/v1/portproxy.proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[vmd_proto, portproxy_proto], &[proto_root])?;

    println!("cargo:rerun-if-changed=../proto/bracket/vmd/v1/vmd.proto");
    println!("cargo:rerun-if-changed=../proto/bracket/portproxy/v1/portproxy.proto");

    Ok(())
}
