//! Check what scans C mozjpeg produces with jpeg_simple_progression

use mozjpeg_sys::*;
use std::ptr;

fn main() {
    println!("=== Test 1: Default behavior (optimize_scans=true from JCP_MAX_COMPRESSION) ===\n");
    check_scans(true);

    println!("\n\n=== Test 2: With optimize_scans=false BEFORE jpeg_simple_progression ===\n");
    check_scans(false);
}

fn check_scans(optimize_scans: bool) {
    unsafe {
        let mut cinfo: jpeg_compress_struct = std::mem::zeroed();
        let mut jerr: jpeg_error_mgr = std::mem::zeroed();

        cinfo.common.err = jpeg_std_error(&mut jerr);
        jpeg_CreateCompress(
            &mut cinfo,
            JPEG_LIB_VERSION as i32,
            std::mem::size_of::<jpeg_compress_struct>(),
        );

        let mut outbuffer: *mut u8 = ptr::null_mut();
        let mut outsize: libc::c_ulong = 0;
        jpeg_mem_dest(&mut cinfo, &mut outbuffer, &mut outsize);

        cinfo.image_width = 64;
        cinfo.image_height = 64;
        cinfo.input_components = 3;
        cinfo.in_color_space = J_COLOR_SPACE::JCS_RGB;

        jpeg_set_defaults(&mut cinfo);

        println!("After jpeg_set_defaults:");
        println!("  num_components: {}", cinfo.num_components);
        println!("  num_scans: {}", cinfo.num_scans);

        // Set optimize_scans BEFORE jpeg_simple_progression
        jpeg_c_set_bool_param(
            &mut cinfo,
            JBOOLEAN_OPTIMIZE_SCANS,
            if optimize_scans { 1 } else { 0 },
        );

        println!(
            "\nAfter setting optimize_scans={}:",
            if optimize_scans { "true" } else { "false" }
        );
        println!("  num_scans: {}", cinfo.num_scans);

        jpeg_simple_progression(&mut cinfo);

        println!("\nAfter jpeg_simple_progression:");
        println!("  num_scans: {}", cinfo.num_scans);

        if !cinfo.scan_info.is_null() {
            for i in 0..cinfo.num_scans {
                let scan = &*cinfo.scan_info.offset(i as isize);
                println!(
                    "  Scan {:2}: comps={} Ss={:2} Se={:2} Ah={} Al={}",
                    i, scan.comps_in_scan, scan.Ss, scan.Se, scan.Ah, scan.Al
                );
            }
        }

        jpeg_destroy_compress(&mut cinfo);
    }
}
