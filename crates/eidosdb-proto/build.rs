#![allow(missing_docs)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure().compile_protos(&["proto/eidosdb.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/eidosdb.proto");
    Ok(())
}
