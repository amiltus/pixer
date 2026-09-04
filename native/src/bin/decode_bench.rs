// Standalone benchmark comparing two JPEG thumbnail decode strategies:
//   full   - image::open() (full-resolution decode) then .resize() down
//   scaled - TurboJPEG DCT-scaled decode to the smallest covering size,
//            then the same final .resize() for the exact target
//
// Run each mode in its own process (so peak RSS reflects only that mode)
// under `/usr/bin/time -l` to get real, measured wall time and peak memory,
// not estimates:
//
//   cargo run --release --bin decode_bench -- full   <path> <w> <h>
//   cargo run --release --bin decode_bench -- scaled <path> <w> <h>

use image::{DynamicImage, imageops::FilterType};
use std::time::Instant;

fn pick_scaling_factor(
    src_width: usize,
    src_height: usize,
    target_width: u32,
    target_height: u32,
) -> turbojpeg::ScalingFactor {
    let identity = turbojpeg::ScalingFactor::new(1, 1);
    let needed = f64::max(
        target_width as f64 / src_width as f64,
        target_height as f64 / src_height as f64,
    );
    if needed >= 1.0 {
        return identity;
    }
    let mut factors = turbojpeg::Decompressor::supported_scaling_factors();
    factors.sort_by(|a, b| {
        let ra = a.num() as f64 / a.denom() as f64;
        let rb = b.num() as f64 / b.denom() as f64;
        ra.partial_cmp(&rb).unwrap()
    });
    factors
        .into_iter()
        .find(|f| (f.num() as f64 / f.denom() as f64) >= needed - 1e-9)
        .unwrap_or(identity)
}

fn decode_scaled(bytes: &[u8], target_width: u32, target_height: u32) -> DynamicImage {
    let mut decompressor = turbojpeg::Decompressor::new().unwrap();
    let header = decompressor.read_header(bytes).unwrap();
    let factor = pick_scaling_factor(header.width, header.height, target_width, target_height);
    decompressor.set_scaling_factor(factor).unwrap();
    let scaled = header.scaled(factor);
    let pitch = scaled.width * turbojpeg::PixelFormat::RGB.size();
    let mut pixels = vec![0u8; pitch * scaled.height];
    let image = turbojpeg::Image {
        pixels: &mut pixels[..],
        width: scaled.width,
        pitch,
        height: scaled.height,
        format: turbojpeg::PixelFormat::RGB,
    };
    decompressor.decompress(bytes, image).unwrap();
    eprintln!(
        "  decode-scaled to: {}x{} (factor {}/{})",
        scaled.width, scaled.height, factor.num(), factor.denom()
    );
    DynamicImage::ImageRgb8(image::RgbImage::from_raw(scaled.width as u32, scaled.height as u32, pixels).unwrap())
}

/// Streaming PNG decode: reads rows one at a time via `png::Reader::next_row`,
/// box-downsamples vertically on the fly (summing `row_group` source rows per
/// output row) and discards each source row immediately after. Never holds
/// more than one output row's worth of source data plus the (much smaller)
/// intermediate output buffer - unlike `image::open`, which must materialise
/// every row of the full-resolution image before returning.
///
/// Only handles 8-bit RGB, non-interlaced PNGs (this repo's real photo/
/// screenshot uploads are overwhelmingly this case) - anything else should
/// fall back to the normal full-decode path in production code.
fn decode_png_streaming(path: &str, target_w: u32, target_h: u32) -> DynamicImage {
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    assert_eq!(info.color_type, png::ColorType::Rgb, "bench only handles RGB8 PNG");
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "bench only handles 8-bit PNG");
    let width = info.width as usize;
    let height = info.height as usize;
    let channels = 3usize;

    // Match the JS worker's own heuristic: shrink roughly to 2x the final
    // target during the cheap streaming pass, then let a proper filter do
    // the precise final resize on the now-small buffer. Both dimensions use
    // the SAME group factor (derived from one uniform scale), so the
    // intermediate buffer keeps the source's aspect ratio - box-downsampling
    // only one axis would otherwise distort it.
    let scale = f64::min(
        (target_w as f64 * 2.0) / width as f64,
        (target_h as f64 * 2.0) / height as f64,
    )
    .min(1.0);
    let group = (1.0 / scale).round().max(1.0) as usize;
    let out_width = width.div_ceil(group);

    let mut sum_row = vec![0u32; width * channels];
    let mut group_count = 0u32;
    let mut output = Vec::<u8>::with_capacity(out_width * channels * (height / group + 1));
    let mut output_rows = 0usize;

    let flush_row = |sum_row: &mut [u32], count: u32, output: &mut Vec<u8>| {
        // Horizontal box-downsample: average `group` consecutive pixels per
        // output column, then divide by the number of source rows summed.
        for chunk_start in (0..width).step_by(group) {
            let chunk_end = (chunk_start + group).min(width);
            for c in 0..channels {
                let mut acc = 0u32;
                let mut n = 0u32;
                for px in chunk_start..chunk_end {
                    acc += sum_row[px * channels + c];
                    n += 1;
                }
                output.push((acc / (n * count)) as u8);
            }
        }
        sum_row.iter_mut().for_each(|v| *v = 0);
    };

    while let Some(row) = reader.next_row().unwrap() {
        let data = row.data();
        for (i, &b) in data.iter().enumerate() {
            sum_row[i] += b as u32;
        }
        group_count += 1;
        if group_count as usize == group {
            flush_row(&mut sum_row, group_count, &mut output);
            group_count = 0;
            output_rows += 1;
        }
    }
    if group_count > 0 {
        flush_row(&mut sum_row, group_count, &mut output);
        output_rows += 1;
    }

    eprintln!(
        "  streaming-decoded to: {}x{} (group={group})",
        out_width, output_rows
    );
    let intermediate = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(out_width as u32, output_rows as u32, output).unwrap(),
    );
    // Final precise resize uses the ORIGINAL source dimensions to compute
    // fit-within target size, matching `DynamicImage::resize`'s own
    // semantics - then applies it via resize_exact since the intermediate
    // buffer's aspect ratio may differ slightly from the source due to
    // integer rounding of `group`.
    let final_scale = f64::min(target_w as f64 / width as f64, target_h as f64 / height as f64);
    let final_w = ((width as f64 * final_scale).round() as u32).max(1);
    let final_h = ((height as f64 * final_scale).round() as u32).max(1);
    intermediate.resize_exact(final_w, final_h, FilterType::Lanczos3)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = &args[1];
    let path = &args[2];
    let target_w: u32 = args[3].parse().unwrap();
    let target_h: u32 = args[4].parse().unwrap();

    let bytes = std::fs::read(path).unwrap();
    eprintln!("mode={mode} input_bytes={}", bytes.len());

    let t0 = Instant::now();
    let decoded = match mode.as_str() {
        "full" => image::load_from_memory(&bytes).unwrap(),
        "scaled" => decode_scaled(&bytes, target_w, target_h),
        "png-streaming" => decode_png_streaming(path, target_w, target_h),
        other => panic!("unknown mode: {other}"),
    };
    let t_decode = t0.elapsed();
    eprintln!(
        "  decoded: {}x{} in {:.2}ms",
        decoded.width(),
        decoded.height(),
        t_decode.as_secs_f64() * 1000.0
    );

    let t1 = Instant::now();
    let resized = if mode == "png-streaming" {
        decoded // already resized to the exact final size inside decode_png_streaming
    } else {
        decoded.resize(target_w, target_h, FilterType::Lanczos3)
    };
    let t_resize = t1.elapsed();

    let total = t0.elapsed();
    eprintln!(
        "  final resize: {}x{} in {:.2}ms",
        resized.width(),
        resized.height(),
        t_resize.as_secs_f64() * 1000.0
    );
    println!(
        "RESULT mode={mode} decode_ms={:.2} resize_ms={:.2} total_ms={:.2} final={}x{}",
        t_decode.as_secs_f64() * 1000.0,
        t_resize.as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0,
        resized.width(),
        resized.height()
    );
}
