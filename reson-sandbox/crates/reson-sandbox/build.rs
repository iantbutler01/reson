// @dive-file: Build-time protobuf generation for facade crate stubs used in distributed/local sandbox clients.
// @dive-rel: Generates vmd and portproxy protobuf modules consumed by crates/reson-sandbox/src/lib.rs and distributed.rs.
// @dive-rel: Must stay aligned with proto/bracket/* service definitions to preserve facade compatibility contracts.

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

    let proto_root = PathBuf::from("../../proto");
    let vmd_proto = proto_root.join("bracket/vmd/v1/vmd.proto");
    let portproxy_proto = proto_root.join("bracket/portproxy/v1/portproxy.proto");

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_well_known_types(true)
        .compile_protos(&[vmd_proto, portproxy_proto], &[proto_root])?;

    println!("cargo:rerun-if-changed=../../proto/bracket/vmd/v1/vmd.proto");
    println!("cargo:rerun-if-changed=../../proto/bracket/portproxy/v1/portproxy.proto");

    Ok(())
}
