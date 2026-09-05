use image::{
    DynamicImage, ImageError, ImageFormat, ImageReader, codecs::jpeg::JpegEncoder,
    imageops::FilterType,
};
use turbojpeg::{Decompressor, Image as TjImage, PixelFormat, ScalingFactor};
use std::{
    ffi::{CStr, CString},
    io::Cursor,
    os::raw::c_char,
    path::Path,
    slice,
};

/// Error code returned through `out_error` pointers and as the result of
/// operations that don't return a handle.
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum ImageErrorCode {
    /// The operation succeeded.
    Success = 0,
    /// The provided path is empty, malformed, or refers to a non-existent file.
    InvalidPath = 1,
    /// The image format is not recognised or not supported by this build.
    UnsupportedFormat = 2,
    /// The image bytes are corrupt or do not match the expected format.
    DecodingError = 3,
    /// Encoding the image to the requested format failed.
    EncodingError = 4,
    /// An underlying I/O operation (read/write) failed.
    IoError = 5,
    /// Width, height, or crop bounds are zero or exceed the image.
    InvalidDimensions = 6,
    /// A handle or output pointer was null, or the image has been freed.
    InvalidPointer = 7,
    /// A scalar parameter (e.g. JPEG quality, blur sigma) is out of range.
    InvalidParameter = 8,
    /// An unclassified error occurred.
    Unknown = 99,
}

/// Image container format used for both decoding and encoding.
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum ImageFormatEnum {
    /// Portable Network Graphics — lossless, alpha supported.
    Png = 0,
    /// JPEG — lossy, no alpha. Quality is configurable on encode.
    Jpeg = 1,
    /// Graphics Interchange Format — palette-based, supports animation
    /// (single-frame only via this API).
    Gif = 2,
    /// WebP — lossy or lossless, alpha supported.
    WebP = 3,
    /// Windows Bitmap — uncompressed, large files.
    Bmp = 4,
    /// Windows Icon — multi-resolution container.
    Ico = 5,
    /// Tagged Image File Format — typically lossless.
    Tiff = 6,
}

impl ImageFormatEnum {
    fn to_image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::Gif => ImageFormat::Gif,
            Self::WebP => ImageFormat::WebP,
            Self::Bmp => ImageFormat::Bmp,
            Self::Ico => ImageFormat::Ico,
            Self::Tiff => ImageFormat::Tiff,
        }
    }
}

/// Sampling filter used when resizing.
///
/// Quality and cost roughly increase from top to bottom; `Lanczos3` is the
/// default and produces the sharpest results, `Nearest` is the fastest.
#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum FilterTypeEnum {
    /// Nearest-neighbour. Fastest, blocky output. Good for pixel art.
    Nearest = 0,
    /// Linear (a.k.a. bilinear). Cheap, slightly blurry.
    Triangle = 1,
    /// Catmull-Rom cubic. Sharper than `Triangle`, can ring on edges.
    CatmullRom = 2,
    /// Gaussian. Soft output, useful for downscaling without aliasing.
    Gaussian = 3,
    /// Lanczos with `a = 3`. Highest quality, slowest. Default.
    Lanczos3 = 4,
}

impl FilterTypeEnum {
    fn to_filter_type(self) -> FilterType {
        match self {
            Self::Nearest => FilterType::Nearest,
            Self::Triangle => FilterType::Triangle,
            Self::CatmullRom => FilterType::CatmullRom,
            Self::Gaussian => FilterType::Gaussian,
            Self::Lanczos3 => FilterType::Lanczos3,
        }
    }
}

#[repr(C)]
pub struct ImageHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub color_type: u8,
    /// Matches [`ImageFormatEnum`]'s discriminants; `255` means the format
    /// is unknown or not applicable (e.g. metadata read from an already-
    /// decoded [`ImageHandle`], which no longer carries its source format).
    pub format: u8,
}

fn format_to_code(format: ImageFormat) -> u8 {
    match format {
        ImageFormat::Png => ImageFormatEnum::Png as u8,
        ImageFormat::Jpeg => ImageFormatEnum::Jpeg as u8,
        ImageFormat::Gif => ImageFormatEnum::Gif as u8,
        ImageFormat::WebP => ImageFormatEnum::WebP as u8,
        ImageFormat::Bmp => ImageFormatEnum::Bmp as u8,
        ImageFormat::Ico => ImageFormatEnum::Ico as u8,
        ImageFormat::Tiff => ImageFormatEnum::Tiff as u8,
        _ => 255,
    }
}

fn with_image<R>(handle: *const ImageHandle, f: impl FnOnce(&DynamicImage) -> R) -> Option<R> {
    if handle.is_null() {
        None
    } else {
        let img = unsafe { &*(handle as *const DynamicImage) };
        Some(f(img))
    }
}

fn into_handle(img: DynamicImage) -> *mut ImageHandle {
    Box::into_raw(Box::new(img)) as *mut ImageHandle
}

fn set_error(out_error: *mut ImageErrorCode, error: ImageErrorCode) {
    if !out_error.is_null() {
        unsafe {
            *out_error = error;
        }
    }
}

fn cstr_to_str(ptr: *const c_char) -> Result<String, ImageErrorCode> {
    if ptr.is_null() {
        return Err(ImageErrorCode::InvalidPointer);
    }
    unsafe {
        CStr::from_ptr(ptr)
            .to_str()
            .map(|s| s.to_owned())
            .map_err(|_| ImageErrorCode::InvalidPath)
    }
}

fn buffer_output(buffer: Vec<u8>, out_data: *mut *mut u8, out_len: *mut usize) {
    let mut boxed = buffer.into_boxed_slice();
    unsafe {
        *out_len = boxed.len();
        *out_data = boxed.as_mut_ptr();
    }
    std::mem::forget(boxed);
}

fn error_to_code(err: &ImageError) -> ImageErrorCode {
    match err {
        ImageError::Decoding(_) => ImageErrorCode::DecodingError,
        ImageError::Encoding(_) => ImageErrorCode::EncodingError,
        ImageError::IoError(_) => ImageErrorCode::IoError,
        ImageError::Limits(_) => ImageErrorCode::InvalidDimensions,
        ImageError::Unsupported(_) => ImageErrorCode::UnsupportedFormat,
        ImageError::Parameter(_) => ImageErrorCode::InvalidParameter,
    }
}

fn get_metadata(img: &DynamicImage) -> ImageMetadata {
    let color_type = match img.color() {
        image::ColorType::L8 | image::ColorType::L16 => 0,
        image::ColorType::La8 | image::ColorType::La16 => 1,
        image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => 2,
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => 3,
        _ => 3,
    };
    ImageMetadata {
        width: img.width(),
        height: img.height(),
        color_type,
        // `DynamicImage` doesn't retain the format it was decoded from.
        format: 255,
    }
}

fn read_metadata_from_reader<R: std::io::BufRead + std::io::Seek>(
    reader: ImageReader<R>,
) -> Result<ImageMetadata, ImageErrorCode> {
    let reader = reader.with_guessed_format().map_err(|_| ImageErrorCode::IoError)?;
    let format = reader.format().map(format_to_code).unwrap_or(255);
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| error_to_code(&e))?;
    Ok(ImageMetadata {
        width,
        height,
        // Header-only dimensions are available without decoding pixels. The
        // image crate does not expose a generic header-only color type here, so
        // keep this conservative for callers that estimate decoded memory.
        color_type: 3,
        format,
    })
}

fn write_to_jpeg_with_quality(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, ImageError> {
    let mut buffer = Vec::new();
    img.write_with_encoder(JpegEncoder::new_with_quality(&mut buffer, quality))?;
    Ok(buffer)
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free a string allocated by Rust
#[unsafe(no_mangle)]
pub extern "C" fn pixer_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Free image data buffer
#[unsafe(no_mangle)]
pub extern "C" fn pixer_free_buffer(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

/// Free an image handle
#[unsafe(no_mangle)]
pub extern "C" fn pixer_free(handle: *mut ImageHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut DynamicImage);
        }
    }
}

// ============================================================================
// Image Loading
// ============================================================================

/// Load an image from a file path
/// Returns null on error
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load(path: *const c_char) -> *mut ImageHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    match cstr_to_str(path)
        .and_then(|p| image::open(Path::new(&p)).map_err(|_| ImageErrorCode::InvalidPath))
    {
        Ok(img) => into_handle(img),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Load an image from memory buffer
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_from_memory(data: *const u8, len: usize) -> *mut ImageHandle {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }

    let buffer = unsafe { slice::from_raw_parts(data, len) };
    match image::load_from_memory(buffer) {
        Ok(img) => into_handle(img),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Load an image from memory with specific format
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_from_memory_with_format(
    data: *const u8,
    len: usize,
    format: ImageFormatEnum,
) -> *mut ImageHandle {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }

    let buffer = unsafe { slice::from_raw_parts(data, len) };

    match image::load_from_memory_with_format(buffer, format.to_image_format()) {
        Ok(img) => into_handle(img),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Load an image from a file path with error code output
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_with_error(
    path: *const c_char,
    out_error: *mut ImageErrorCode,
) -> *mut ImageHandle {
    if path.is_null() {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return std::ptr::null_mut();
    }

    match cstr_to_str(path).and_then(|p| image::open(Path::new(&p)).map_err(|e| error_to_code(&e)))
    {
        Ok(img) => {
            set_error(out_error, ImageErrorCode::Success);
            into_handle(img)
        }
        Err(code) => {
            set_error(out_error, code);
            std::ptr::null_mut()
        }
    }
}

/// Load an image from memory buffer with error code output
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_from_memory_with_error(
    data: *const u8,
    len: usize,
    out_error: *mut ImageErrorCode,
) -> *mut ImageHandle {
    if data.is_null() || len == 0 {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return std::ptr::null_mut();
    }

    let buffer = unsafe { slice::from_raw_parts(data, len) };

    match image::load_from_memory(buffer) {
        Ok(img) => {
            set_error(out_error, ImageErrorCode::Success);
            into_handle(img)
        }
        Err(e) => {
            set_error(out_error, error_to_code(&e));
            std::ptr::null_mut()
        }
    }
}

/// Load an image from memory with specific format and error code output
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_from_memory_with_format_and_error(
    data: *const u8,
    len: usize,
    format: ImageFormatEnum,
    out_error: *mut ImageErrorCode,
) -> *mut ImageHandle {
    if data.is_null() || len == 0 {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return std::ptr::null_mut();
    }

    let buffer = unsafe { slice::from_raw_parts(data, len) };

    match image::load_from_memory_with_format(buffer, format.to_image_format()) {
        Ok(img) => {
            set_error(out_error, ImageErrorCode::Success);
            into_handle(img)
        }
        Err(e) => {
            set_error(out_error, error_to_code(&e));
            std::ptr::null_mut()
        }
    }
}

/// Picks the smallest TurboJPEG scaling factor whose output is still at
/// least as large as `(target_width, target_height)` in both dimensions
/// (i.e. "cover", not "contain" - callers that only need "contain" simply
/// downscale further afterwards, which is cheap once the buffer is small).
/// Falls back to 1/1 (no scaling) if no smaller factor covers the target,
/// or if the target is zero/degenerate.
fn pick_scaling_factor(src_width: usize, src_height: usize, target_width: u32, target_height: u32) -> ScalingFactor {
    let identity = ScalingFactor::new(1, 1);
    if target_width == 0 || target_height == 0 || src_width == 0 || src_height == 0 {
        return identity;
    }
    let needed = f64::max(
        target_width as f64 / src_width as f64,
        target_height as f64 / src_height as f64,
    );
    if needed >= 1.0 {
        return identity;
    }
    let mut factors = Decompressor::supported_scaling_factors();
    factors.sort_by(|a, b| {
        let ra = a.num() as f64 / a.denom() as f64;
        let rb = b.num() as f64 / b.denom() as f64;
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });
    factors
        .into_iter()
        .find(|f| (f.num() as f64 / f.denom() as f64) >= needed - 1e-9)
        .unwrap_or(identity)
}

/// Attempts a decode-time-downscaled JPEG load via TurboJPEG's DCT scaling
/// (dropping high-frequency coefficients during decode, never materialising
/// the full-resolution pixel buffer). Returns `None` for non-JPEG input or
/// on any TurboJPEG failure, so callers can fall back to the general-purpose
/// `image`-crate decode path transparently.
fn try_decode_scaled_jpeg(bytes: &[u8], target_width: u32, target_height: u32) -> Option<DynamicImage> {
    let mut decompressor = Decompressor::new().ok()?;
    let header = decompressor.read_header(bytes).ok()?;
    let factor = pick_scaling_factor(header.width, header.height, target_width, target_height);
    decompressor.set_scaling_factor(factor).ok()?;

    let scaled = header.scaled(factor);
    let pitch = scaled.width * PixelFormat::RGB.size();
    let mut pixels = vec![0u8; pitch * scaled.height];
    let image = TjImage {
        pixels: &mut pixels[..],
        width: scaled.width,
        pitch,
        height: scaled.height,
        format: PixelFormat::RGB,
    };
    decompressor.decompress(bytes, image).ok()?;

    image::RgbImage::from_raw(scaled.width as u32, scaled.height as u32, pixels).map(DynamicImage::ImageRgb8)
}

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn is_jpeg_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

fn is_png_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[..8] == PNG_SIGNATURE
}

/// Attempts a decode-time-downscaled PNG load by streaming scanlines via
/// `png::Reader::next_row` and box-downsampling both axes on the fly,
/// discarding each source row immediately after - never materialising more
/// than one output row's worth of source data plus the (much smaller)
/// intermediate buffer, unlike a full decode which must hold every row of
/// the full-resolution image before returning.
///
/// Only handles 8-bit, non-interlaced RGB/RGBA PNGs (this covers real-world
/// photo/screenshot uploads overwhelmingly): paletted, 16-bit, grayscale
/// (with or without alpha), and Adam7-interlaced PNGs all return `None` so
/// callers fall back to the general-purpose full decode - interlaced PNGs in
/// particular deliver rows out of sequence across 7 passes and would
/// silently corrupt output if run through this row-by-row logic as-is.
///
/// The returned image is shrunk to roughly *twice* the requested target size
/// (per libvips' own guidance: shrink-on-load to ~2x target, then let a
/// proper filter do the precise final resize) rather than exactly the
/// target - box-downsampling is a crude filter, and going straight to the
/// exact final size with it would visibly soften or alias the result versus
/// finishing with a proper resize afterward, same as the browser-side
/// implementation in `clientTransferWorker.pica.js`. Returns `None` (no
/// benefit over full decode) if the image doesn't need to shrink by at
/// least half in both dimensions.
fn try_decode_scaled_png<R: std::io::BufRead + std::io::Seek>(
    reader: R,
    target_width: u32,
    target_height: u32,
) -> Option<DynamicImage> {
    let decoder = png::Decoder::new(reader);
    let mut png_reader = decoder.read_info().ok()?;
    let info = png_reader.info();
    if info.bit_depth != png::BitDepth::Eight || info.interlaced {
        return None;
    }
    let channels = match info.color_type {
        png::ColorType::Rgb => 3usize,
        png::ColorType::Rgba => 4usize,
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha | png::ColorType::Indexed => {
            return None;
        }
    };
    let width = info.width as usize;
    let height = info.height as usize;
    if width == 0 || height == 0 {
        return None;
    }

    let scale = f64::min(
        (target_width as f64 * 2.0) / width as f64,
        (target_height as f64 * 2.0) / height as f64,
    )
    .min(1.0);
    let group = (1.0 / scale).round().max(1.0) as usize;
    if group <= 1 {
        // Wouldn't shrink enough to be worth a box-filtered intermediate;
        // let the general path do a single proper-quality decode instead.
        return None;
    }
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

    while let Some(row) = png_reader.next_row().ok()? {
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

    match channels {
        3 => image::RgbImage::from_raw(out_width as u32, output_rows as u32, output)
            .map(DynamicImage::ImageRgb8),
        4 => image::RgbaImage::from_raw(out_width as u32, output_rows as u32, output)
            .map(DynamicImage::ImageRgba8),
        _ => unreachable!("channels is only ever set to 3 or 4 above"),
    }
}

/// Dispatches to whichever decode-time-downscaled path matches `bytes`'
/// format - JPEG via TurboJPEG's DCT scaling, PNG via streaming scanline
/// box-downsampling - or `None` for any other format or on failure, so
/// callers can fall back to the general-purpose full decode transparently.
///
/// `pub` only so `decode_bench` can measure the exact real code path used
/// by `pixer_load_scaled_from_memory_with_error`/`_from_file_with_error`.
pub fn try_decode_scaled(bytes: &[u8], target_width: u32, target_height: u32) -> Option<DynamicImage> {
    if is_jpeg_magic(bytes) {
        return try_decode_scaled_jpeg(bytes, target_width, target_height);
    }
    if is_png_magic(bytes) {
        return try_decode_scaled_png(Cursor::new(bytes), target_width, target_height);
    }
    None
}

/// Loads an image from memory, decoding at a reduced resolution suited to
/// `(target_width, target_height)` when the source is a JPEG or PNG.
///
/// This exists for thumbnail generation: a JPEG's DCT structure and a PNG's
/// scanline structure both let the decoder avoid reconstructing full
/// resolution when the caller only needs a much smaller output, cutting
/// decode memory and CPU roughly in proportion to the scale chosen. Any
/// other format, or a decode-time-downscale failure, falls back
/// transparently to the regular full decode.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_scaled_from_memory_with_error(
    data: *const u8,
    len: usize,
    target_width: u32,
    target_height: u32,
    out_error: *mut ImageErrorCode,
) -> *mut ImageHandle {
    if data.is_null() || len == 0 {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return std::ptr::null_mut();
    }
    let buffer = unsafe { slice::from_raw_parts(data, len) };

    if let Some(img) = try_decode_scaled(buffer, target_width, target_height) {
        set_error(out_error, ImageErrorCode::Success);
        return into_handle(img);
    }

    match image::load_from_memory(buffer) {
        Ok(img) => {
            set_error(out_error, ImageErrorCode::Success);
            into_handle(img)
        }
        Err(e) => {
            set_error(out_error, error_to_code(&e));
            std::ptr::null_mut()
        }
    }
}

/// File-path counterpart of [`pixer_load_scaled_from_memory_with_error`].
#[unsafe(no_mangle)]
pub extern "C" fn pixer_load_scaled_from_file_with_error(
    path: *const c_char,
    target_width: u32,
    target_height: u32,
    out_error: *mut ImageErrorCode,
) -> *mut ImageHandle {
    let path_str = match cstr_to_str(path) {
        Ok(p) => p,
        Err(code) => {
            set_error(out_error, code);
            return std::ptr::null_mut();
        }
    };

    if let Ok(bytes) = std::fs::read(&path_str) {
        if let Some(img) = try_decode_scaled(&bytes, target_width, target_height) {
            set_error(out_error, ImageErrorCode::Success);
            return into_handle(img);
        }
    }

    match image::open(Path::new(&path_str)) {
        Ok(img) => {
            set_error(out_error, ImageErrorCode::Success);
            into_handle(img)
        }
        Err(e) => {
            set_error(out_error, error_to_code(&e));
            std::ptr::null_mut()
        }
    }
}

/// Read image metadata from a file path without decoding pixel data
#[unsafe(no_mangle)]
pub extern "C" fn pixer_read_metadata_from_file_with_error(
    path: *const c_char,
    out_metadata: *mut ImageMetadata,
    out_error: *mut ImageErrorCode,
) -> ImageErrorCode {
    if path.is_null() || out_metadata.is_null() {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return ImageErrorCode::InvalidPointer;
    }

    let result = cstr_to_str(path).and_then(|p| {
        ImageReader::open(Path::new(&p))
            .map_err(|_| ImageErrorCode::IoError)
            .and_then(read_metadata_from_reader)
    });

    match result {
        Ok(metadata) => {
            unsafe {
                *out_metadata = metadata;
            }
            set_error(out_error, ImageErrorCode::Success);
            ImageErrorCode::Success
        }
        Err(code) => {
            set_error(out_error, code);
            code
        }
    }
}

/// Read image metadata from memory without decoding pixel data
#[unsafe(no_mangle)]
pub extern "C" fn pixer_read_metadata_from_memory_with_error(
    data: *const u8,
    len: usize,
    out_metadata: *mut ImageMetadata,
    out_error: *mut ImageErrorCode,
) -> ImageErrorCode {
    if data.is_null() || len == 0 || out_metadata.is_null() {
        set_error(out_error, ImageErrorCode::InvalidPointer);
        return ImageErrorCode::InvalidPointer;
    }

    let buffer = unsafe { slice::from_raw_parts(data, len) };
    let cursor = Cursor::new(buffer);
    match read_metadata_from_reader(ImageReader::new(cursor)) {
        Ok(metadata) => {
            unsafe {
                *out_metadata = metadata;
            }
            set_error(out_error, ImageErrorCode::Success);
            ImageErrorCode::Success
        }
        Err(code) => {
            set_error(out_error, code);
            code
        }
    }
}

// ============================================================================
// Image Saving
// ============================================================================

/// Save an image to a file path
#[unsafe(no_mangle)]
pub extern "C" fn pixer_save(handle: *const ImageHandle, path: *const c_char) -> ImageErrorCode {
    if path.is_null() {
        return ImageErrorCode::InvalidPointer;
    }

    with_image(handle, |img| {
        match cstr_to_str(path).and_then(|p| img.save(Path::new(&p)).map_err(|e| error_to_code(&e)))
        {
            Ok(_) => ImageErrorCode::Success,
            Err(code) => code,
        }
    })
    .unwrap_or(ImageErrorCode::InvalidPointer)
}

/// Write an image to a buffer in the specified format
/// Caller must free the buffer using pixer_free_buffer
#[unsafe(no_mangle)]
pub extern "C" fn pixer_write_to(
    handle: *const ImageHandle,
    format: ImageFormatEnum,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> ImageErrorCode {
    if out_data.is_null() || out_len.is_null() {
        return ImageErrorCode::InvalidPointer;
    }

    with_image(handle, |img| {
        match {
            let mut cursor = std::io::Cursor::new(Vec::new());
            img.write_to(&mut cursor, format.to_image_format())
                .map(|_| cursor.into_inner())
        } {
            Ok(buffer) => {
                buffer_output(buffer, out_data, out_len);
                ImageErrorCode::Success
            }
            Err(e) => error_to_code(&e),
        }
    })
    .unwrap_or(ImageErrorCode::InvalidPointer)
}

/// Write an image to a JPEG buffer with the specified quality.
///
/// `quality` must be in `1..=100`; `format` must be `Jpeg`. Use
/// `pixer_write_to` for other formats. Caller must free the buffer using
/// `pixer_free_buffer`.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_write_to_with_quality(
    handle: *const ImageHandle,
    format: ImageFormatEnum,
    quality: u8,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> ImageErrorCode {
    if out_data.is_null() || out_len.is_null() {
        return ImageErrorCode::InvalidPointer;
    }

    if !matches!(format, ImageFormatEnum::Jpeg) || !(1..=100).contains(&quality) {
        return ImageErrorCode::InvalidParameter;
    }

    with_image(handle, |img| {
        match write_to_jpeg_with_quality(img, quality) {
            Ok(buffer) => {
                buffer_output(buffer, out_data, out_len);
                ImageErrorCode::Success
            }
            Err(e) => error_to_code(&e),
        }
    })
    .unwrap_or(ImageErrorCode::InvalidPointer)
}

// ============================================================================
// Image Information
// ============================================================================

/// Get image metadata
#[unsafe(no_mangle)]
pub extern "C" fn pixer_get_metadata(
    handle: *const ImageHandle,
    out_metadata: *mut ImageMetadata,
) -> ImageErrorCode {
    if out_metadata.is_null() {
        return ImageErrorCode::InvalidPointer;
    }

    let Some(metadata) = with_image(handle, get_metadata) else {
        return ImageErrorCode::InvalidPointer;
    };

    unsafe {
        *out_metadata = metadata;
    }
    ImageErrorCode::Success
}

// ============================================================================
// Image Transformations
// ============================================================================

/// Resize the image to fit *within* `width` x `height` while preserving
/// aspect ratio.
///
/// The result is at most `width` x `height`; the smaller dimension is scaled
/// proportionally so the image is never distorted. Use `pixer_resize_exact`
/// to force exact dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_resize(
    handle: *const ImageHandle,
    width: u32,
    height: u32,
    filter: FilterTypeEnum,
) -> *mut ImageHandle {
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    with_image(handle, |img| {
        into_handle(img.resize(width, height, filter.to_filter_type()))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Resize the image to exactly `width` x `height`, ignoring aspect ratio.
///
/// May visibly stretch or squash the image.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_resize_exact(
    handle: *const ImageHandle,
    width: u32,
    height: u32,
    filter: FilterTypeEnum,
) -> *mut ImageHandle {
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    with_image(handle, |img| {
        into_handle(img.resize_exact(width, height, filter.to_filter_type()))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Crop an image (immutable)
#[unsafe(no_mangle)]
pub extern "C" fn pixer_crop_imm(
    handle: *const ImageHandle,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> *mut ImageHandle {
    if width == 0 || height == 0 {
        return std::ptr::null_mut();
    }

    with_image(handle, |img| {
        let Some(max_x) = x.checked_add(width) else {
            return std::ptr::null_mut();
        };
        let Some(max_y) = y.checked_add(height) else {
            return std::ptr::null_mut();
        };

        if max_x > img.width() || max_y > img.height() {
            return std::ptr::null_mut();
        }

        into_handle(img.crop_imm(x, y, width, height))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Rotate an image 90 degrees clockwise
#[unsafe(no_mangle)]
pub extern "C" fn pixer_rotate90(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.rotate90())).unwrap_or(std::ptr::null_mut())
}

/// Rotate an image 180 degrees
#[unsafe(no_mangle)]
pub extern "C" fn pixer_rotate180(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.rotate180())).unwrap_or(std::ptr::null_mut())
}

/// Rotate an image 270 degrees clockwise
#[unsafe(no_mangle)]
pub extern "C" fn pixer_rotate270(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.rotate270())).unwrap_or(std::ptr::null_mut())
}

/// Flip an image horizontally
#[unsafe(no_mangle)]
pub extern "C" fn pixer_fliph(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.fliph())).unwrap_or(std::ptr::null_mut())
}

/// Flip an image vertically
#[unsafe(no_mangle)]
pub extern "C" fn pixer_flipv(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.flipv())).unwrap_or(std::ptr::null_mut())
}

// ============================================================================
// Image Filters & Adjustments
// ============================================================================

/// Apply a Gaussian blur with the given standard deviation in pixels.
///
/// `sigma` must be finite and `>= 0`. `sigma == 0` returns an unchanged copy.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_blur(handle: *const ImageHandle, sigma: f32) -> *mut ImageHandle {
    if !sigma.is_finite() || sigma < 0.0 {
        return std::ptr::null_mut();
    }

    with_image(handle, |img| into_handle(img.blur(sigma))).unwrap_or(std::ptr::null_mut())
}

/// Add `value` to every channel of every pixel.
///
/// Values are clamped per-channel to `[0, 255]`. Negative values darken,
/// positive values brighten. The practical range is roughly `-255..=255`;
/// larger magnitudes simply saturate.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_brighten(handle: *const ImageHandle, value: i32) -> *mut ImageHandle {
    with_image(handle, |img| into_handle(img.brighten(value))).unwrap_or(std::ptr::null_mut())
}

/// Adjust contrast around the midpoint.
///
/// `c == 0.0` leaves the image unchanged. Positive values increase contrast,
/// negative values decrease it. `c` must be finite.
#[unsafe(no_mangle)]
pub extern "C" fn pixer_adjust_contrast(handle: *const ImageHandle, c: f32) -> *mut ImageHandle {
    if !c.is_finite() {
        return std::ptr::null_mut();
    }

    with_image(handle, |img| into_handle(img.adjust_contrast(c))).unwrap_or(std::ptr::null_mut())
}

/// Convert to grayscale
#[unsafe(no_mangle)]
pub extern "C" fn pixer_grayscale(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| {
        into_handle(DynamicImage::ImageLuma8(img.to_luma8()))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Invert colors (returns new image)
#[unsafe(no_mangle)]
pub extern "C" fn pixer_invert(handle: *const ImageHandle) -> *mut ImageHandle {
    with_image(handle, |img| {
        let mut cloned = img.clone();
        cloned.invert();
        into_handle(cloned)
    })
    .unwrap_or(std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes a synthetic PNG with an arbitrary color type/bit depth/
    /// interlacing combination, for exercising `try_decode_scaled_png`'s
    /// format gating without needing checked-in binary fixtures. Pixel
    /// content is an arbitrary varying fill - these tests only care about
    /// dimensions and which format/interlace combinations are accepted.
    fn encode_test_png(
        width: u32,
        height: u32,
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
        interlaced: bool,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut info = png::Info::default();
        info.width = width;
        info.height = height;
        info.color_type = color_type;
        info.bit_depth = bit_depth;
        info.interlaced = interlaced;
        let mut encoder = png::Encoder::with_info(&mut buf, info).unwrap();
        if color_type == png::ColorType::Indexed {
            // A tiny palette is enough for a synthetic fixture; real
            // paletted PNGs always carry one, which is why the streaming
            // path can't just treat the index bytes as pixel samples.
            encoder.set_palette(vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255]);
        }
        let mut writer = encoder.write_header().unwrap();

        let bytes_per_sample = if bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
        let channels = match color_type {
            png::ColorType::Grayscale | png::ColorType::Indexed => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
        };
        let byte_count = width as usize * height as usize * channels * bytes_per_sample;
        let data: Vec<u8> = (0..byte_count).map(|i| (i % 200 + 20) as u8).collect();
        writer.write_image_data(&data).unwrap();
        drop(writer);
        buf
    }

    #[test]
    fn scaled_png_streams_rgb8_non_interlaced() {
        let bytes = encode_test_png(400, 200, png::ColorType::Rgb, png::BitDepth::Eight, false);
        let img = try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25)
            .expect("RGB8 non-interlaced PNG should use the streaming path");
        assert!(img.width() < 400 && img.width() >= 50);
        assert!(img.height() < 200 && img.height() >= 25);
    }

    #[test]
    fn scaled_png_streams_rgba8_non_interlaced() {
        let bytes = encode_test_png(400, 200, png::ColorType::Rgba, png::BitDepth::Eight, false);
        let img = try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25)
            .expect("RGBA8 non-interlaced PNG should use the streaming path");
        assert!(img.width() < 400);
        assert_eq!(img.color(), image::ColorType::Rgba8);
    }

    #[test]
    fn scaled_png_returns_none_for_images_that_barely_shrink() {
        // Requesting a target close to the source size shouldn't produce a
        // degenerate 1-group-per-pixel "box downsample" - the general path
        // handles this better with a real filter.
        let bytes = encode_test_png(100, 100, png::ColorType::Rgb, png::BitDepth::Eight, false);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 90, 90).is_none());
    }

    #[test]
    fn scaled_png_falls_back_for_paletted() {
        let bytes = encode_test_png(400, 200, png::ColorType::Indexed, png::BitDepth::Eight, false);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25).is_none());
    }

    #[test]
    fn scaled_png_falls_back_for_16bit() {
        let bytes = encode_test_png(400, 200, png::ColorType::Rgb, png::BitDepth::Sixteen, false);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25).is_none());
    }

    #[test]
    fn scaled_png_falls_back_for_grayscale() {
        let bytes = encode_test_png(400, 200, png::ColorType::Grayscale, png::BitDepth::Eight, false);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25).is_none());
    }

    #[test]
    fn scaled_png_falls_back_for_grayscale_alpha() {
        let bytes =
            encode_test_png(400, 200, png::ColorType::GrayscaleAlpha, png::BitDepth::Eight, false);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25).is_none());
    }

    #[test]
    fn scaled_png_falls_back_for_adam7_interlaced() {
        let bytes = encode_test_png(400, 200, png::ColorType::Rgb, png::BitDepth::Eight, true);
        assert!(try_decode_scaled_png(Cursor::new(bytes.as_slice()), 50, 25).is_none());
    }

    #[test]
    fn fallback_formats_still_decode_correctly_via_the_general_path() {
        // The streaming fast path bailing must never mean the image is
        // undecodable - it must fall through to a correct full decode.
        //
        // Adam7-interlaced is deliberately not covered here: `write_image_data`
        // expects pixel data pre-arranged into Adam7 pass order when
        // `info.interlaced` is set, which `encode_test_png`'s plain raster
        // fill doesn't do - `scaled_png_falls_back_for_adam7_interlaced`
        // covers the actually load-bearing behavior (the fast path declines
        // it) using the same encoder without asserting a correct full
        // decode of that malformed fixture.
        for (color_type, bit_depth) in [
            (png::ColorType::Indexed, png::BitDepth::Eight),
            (png::ColorType::Rgb, png::BitDepth::Sixteen),
            (png::ColorType::Grayscale, png::BitDepth::Eight),
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight),
        ] {
            let bytes = encode_test_png(64, 32, color_type, bit_depth, false);
            assert!(
                try_decode_scaled(&bytes, 16, 16).is_none(),
                "expected streaming fast path to bail for {color_type:?}/{bit_depth:?}"
            );
            let full = image::load_from_memory(&bytes).unwrap_or_else(|e| {
                panic!("fallback full decode must still succeed for {color_type:?}/{bit_depth:?}: {e}")
            });
            assert_eq!((full.width(), full.height()), (64, 32));
        }
    }

    #[test]
    fn dispatch_ignores_non_png_non_jpeg_bytes() {
        assert!(try_decode_scaled(b"not an image", 50, 50).is_none());
    }
}
