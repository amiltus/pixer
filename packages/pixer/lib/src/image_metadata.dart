import 'dart:ffi' as ffi;
import 'bindings/bindings.dart';

/// Pixel layout of an image: which channels are present.
enum ColorType {
  /// Single luminance channel (8 or 16 bit).
  luminance(0),

  /// Luminance + alpha.
  luminanceAlpha(1),

  /// Red, green, blue.
  rgb(2),

  /// Red, green, blue, alpha.
  rgba(3);

  const ColorType(this.value);

  final int value;

  /// Looks up the [ColorType] matching the native u8 code.
  ///
  /// Throws [ArgumentError] on unknown values — native and Dart enums are
  /// expected to stay in sync.
  static ColorType fromValue(int value) => switch (value) {
    0 => luminance,
    1 => luminanceAlpha,
    2 => rgb,
    3 => rgba,
    _ => throw ArgumentError('Unknown value for ColorType: $value'),
  };
}

/// The sentinel `format` byte native code sends when the source format is
/// unknown or not applicable (e.g. metadata read from an already-decoded
/// handle, which no longer carries its source format).
const _formatUnknown = 255;

/// Width, height, color layout, and (where known) source format of an image.
final class PixerMetadata {
  const PixerMetadata({
    required this.width,
    required this.height,
    required this.colorType,
    this.format,
  });

  /// Image width in pixels.
  final int width;

  /// Image height in pixels.
  final int height;

  /// Pixel layout (channels present).
  final ColorType colorType;

  /// The image's source container format, when known.
  ///
  /// Populated by [Pixer.readMetadataFromFile]/[Pixer.readMetadataFromMemory]
  /// (header-only reads, before any pixel data is decoded). `null` for
  /// metadata read from an already-decoded [Pixer] handle — the format isn't
  /// tracked once an image has been loaded — or for a format this binding
  /// doesn't have a case for.
  final ImageFormatEnum? format;

  /// Creates metadata from the native struct.
  factory PixerMetadata.fromNative(ffi.Pointer<ImageMetadata> ptr) {
    final metadata = ptr.ref;
    return PixerMetadata(
      width: metadata.width,
      height: metadata.height,
      colorType: ColorType.fromValue(metadata.color_type),
      format: metadata.format == _formatUnknown
          ? null
          : ImageFormatEnum.fromValue(metadata.format),
    );
  }

  @override
  String toString() =>
      'PixerMetadata(width: $width, height: $height, colorType: ${colorType.name}, format: ${format?.name})';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is PixerMetadata &&
          width == other.width &&
          height == other.height &&
          colorType == other.colorType &&
          format == other.format;

  @override
  int get hashCode => Object.hash(width, height, colorType, format);
}
