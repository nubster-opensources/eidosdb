#![allow(missing_docs, unsafe_code)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Point prost-build at a vendored protoc so the build needs no system
    // protobuf compiler (works identically in CI and for every contributor).
    // An explicit PROTOC in the environment still wins.
    if std::env::var_os("PROTOC").is_none_or(|value| value.is_empty()) {
        let protoc = protoc_bin_vendored::protoc_bin_path()?;
        // SAFETY: a build script runs single-threaded before any other code in
        // this process, so mutating the environment here is sound.
        unsafe { std::env::set_var("PROTOC", protoc) };
    }
    tonic_prost_build::configure().compile_protos(&["proto/eidosdb.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/eidosdb.proto");
    Ok(())
}
