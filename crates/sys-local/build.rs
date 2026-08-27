use std::env;
use std::path::PathBuf;

fn main() {
    // Path to the local mozjpeg source (sibling of project root)
    // crates/sys-local -> crates -> project_root -> parent -> mozjpeg
    let mozjpeg_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("mozjpeg");

    println!("cargo:rerun-if-changed={}", mozjpeg_src.display());
    println!(
        "cargo:rerun-if-changed={}/mozjpeg_test_exports.c",
        mozjpeg_src.display()
    );
    println!(
        "cargo:rerun-if-changed={}/mozjpeg_test_exports.h",
        mozjpeg_src.display()
    );

    // Build mozjpeg with CMake
    // Note: mozjpeg uses BUILD_SHARED_LIBS, not ENABLE_SHARED/ENABLE_STATIC
    // We only build jpeg-static target (libjpeg.a), not the executables
    let dst = cmake::Config::new(&mozjpeg_src)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WITH_TURBOJPEG", "FALSE")
        .define("WITH_JAVA", "FALSE")
        .define("PNG_SUPPORTED", "FALSE")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build_target("jpeg-static")
        .build();

    // Link the static library
    // When using build_target, output is in dst/build, not dst/lib
    let build_dir = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Check what was actually built
    if build_dir.join("libjpeg.a").exists() {
        println!("cargo:rustc-link-lib=static=jpeg");
    } else {
        let lib_dir = dst.join("lib");
        if lib_dir.join("libjpeg.a").exists() {
            println!("cargo:rustc-link-lib=static=jpeg");
        } else if lib_dir.join("libjpeg-static.a").exists() {
            println!("cargo:rustc-link-lib=static=jpeg-static");
        } else {
            // Fall back to dynamic linking
            println!("cargo:rustc-link-lib=jpeg");
        }
    }

    // Tell cargo where headers are (for generated headers like jconfig.h)
    println!("cargo:include={}/build", dst.display());
    println!("cargo:include={}", mozjpeg_src.display());

    // Test exports that live in this crate rather than in the mozjpeg C tree.
    // `mozjpeg_test_dc_trellis_optimize` is declared in src/lib.rs but the
    // mozjpeg fork's mozjpeg_test_exports.c never defined it, so without this
    // shim the ffi_comparison test and trace_pipeline example fail to link.
    // The shim is header-free (stdint types only), so it does not depend on
    // the CMake-generated jconfig.h.
    let csrc = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("csrc");
    let dc_trellis = csrc.join("mozjpeg_test_dc_trellis.c");
    println!("cargo:rerun-if-changed={}", dc_trellis.display());
    cc::Build::new()
        .file(&dc_trellis)
        .warnings(true)
        .compile("mozjpeg_test_dc_trellis");
}
