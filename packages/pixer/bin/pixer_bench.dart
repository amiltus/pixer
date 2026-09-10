import 'dart:io';

import 'package:pixer/pixer.dart';

void main() async {
  final img = Pixer.fromFile('assets/example_img.jpg');

  // Upscale the image to 3840x2160
  final resizedImage = img.resize(3840, 2160, preserveAspectRatio: false);

  // Encode the image to PNG
  final pngBytes = resizedImage.encode(PixerPngEncoder());
  File('example_img.png').writeAsBytesSync(pngBytes);

  // Dispose the images to release memory
  img.dispose();
  resizedImage.dispose();
}
