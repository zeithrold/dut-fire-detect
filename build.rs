use std::{path::Path, process::Command};

fn get_xcode_path() -> String {
    let output = Command::new("xcode-select")
        .arg("-p")
        .output()
        .expect("Failed to get Xcode path")
        .stdout;
    let output = String::from_utf8(output).expect("Failed to get Xcode path");
    let output = Path::new(&output).join("usr").join("lib");
    output.to_str().expect("Failed to get Xcode path").to_string()
}



fn main() {
    #[cfg(feature = "coreml")]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-env=DYLD_FALLBACK_LIBRARY_PATH={}", get_xcode_path());
}
