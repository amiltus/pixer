// `pub` only so `decode_bench` (a bin target in this same package) can call
// the real decode-time-downscale dispatch directly for benchmarking -
// nothing outside this package depends on this crate as a Rust library.
pub mod ffi;
