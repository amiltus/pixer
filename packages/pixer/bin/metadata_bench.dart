import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as dart_image;
import 'package:pixer/pixer.dart';

const _mib = 1024 * 1024;

Future<void> main(List<String> args) async {
  final options = _Options.parse(args);
  final file = File(options.input);
  if (!await file.exists()) {
    stderr.writeln('Input does not exist: ${options.input}');
    exitCode = 64;
    return;
  }

  final bytes = await file.readAsBytes();
  print(
    'schema=pixer.metadataBench.v1 input=${file.absolute.path} '
    'bytes=${bytes.length} iterations=${options.iterations}',
  );

  await _runEngine(
    'pixer_header_memory',
    options.iterations,
    () => _pixerHeaderMemory(bytes),
  );
  await _runEngine(
    'pixer_header_file',
    options.iterations,
    () => _pixerHeaderFile(file.path),
  );
  await _runEngine(
    'pixer_full_decode_memory',
    options.iterations,
    () => _pixerFullDecodeMemory(bytes),
  );
  await _runEngine(
    'pixer_full_decode_file',
    options.iterations,
    () => _pixerFullDecodeFile(file.path),
  );
  await _runEngine(
    'package_image_start_decode',
    options.iterations,
    () => _packageImageStartDecode(bytes),
  );
}

Future<void> _runEngine(
  String engine,
  int iterations,
  _MetadataResult Function() task,
) async {
  final timings = <int>[];
  final rssDeltas = <int>[];
  _MetadataResult? last;

  // Warm up FFI bindings and decoder caches outside the timed sample.
  last = task();

  for (var index = 0; index < iterations; index++) {
    final beforeRss = ProcessInfo.currentRss;
    final stopwatch = Stopwatch()..start();
    last = task();
    stopwatch.stop();
    timings.add(stopwatch.elapsedMicroseconds);
    rssDeltas.add(ProcessInfo.currentRss - beforeRss);
  }

  final result = last;
  if (result == null) throw StateError('No benchmark result for $engine.');
  print(
    'result engine=$engine width=${result.width} height=${result.height} '
    'color=${result.color} p50Us=${_percentile(timings, 0.50)} '
    'p95Us=${_percentile(timings, 0.95)} '
    'avgUs=${_average(timings).toStringAsFixed(0)} '
    'maxRssDeltaMiB=${(_max(rssDeltas) / _mib).toStringAsFixed(2)}',
  );
}

_MetadataResult _pixerHeaderMemory(Uint8List bytes) {
  final metadata = Pixer.readMetadataFromMemory(bytes);
  return _MetadataResult(
    width: metadata.width,
    height: metadata.height,
    color: metadata.colorType.name,
  );
}

_MetadataResult _pixerHeaderFile(String path) {
  final metadata = Pixer.readMetadataFromFile(path);
  return _MetadataResult(
    width: metadata.width,
    height: metadata.height,
    color: metadata.colorType.name,
  );
}

_MetadataResult _pixerFullDecodeMemory(Uint8List bytes) {
  final image = Pixer.fromMemory(bytes);
  try {
    final metadata = image.getMetadata();
    return _MetadataResult(
      width: metadata.width,
      height: metadata.height,
      color: metadata.colorType.name,
    );
  } finally {
    image.dispose();
  }
}

_MetadataResult _pixerFullDecodeFile(String path) {
  final image = Pixer.fromFile(path);
  try {
    final metadata = image.getMetadata();
    return _MetadataResult(
      width: metadata.width,
      height: metadata.height,
      color: metadata.colorType.name,
    );
  } finally {
    image.dispose();
  }
}

_MetadataResult _packageImageStartDecode(Uint8List bytes) {
  final decoder = dart_image.findDecoderForData(bytes);
  final info = decoder?.startDecode(bytes);
  if (info == null) throw StateError('package:image could not read metadata.');
  return _MetadataResult(
    width: info.width,
    height: info.height,
    color: 'rgbaBudget',
  );
}

int _percentile(List<int> values, double percentile) {
  final sorted = values.toList()..sort();
  final index =
      (sorted.length * percentile).ceil().clamp(1, sorted.length).toInt() - 1;
  return sorted[index];
}

double _average(List<int> values) {
  return values.reduce((left, right) => left + right) / values.length;
}

int _max(List<int> values) {
  return values.fold<int>(values.first, math.max);
}

final class _MetadataResult {
  const _MetadataResult({
    required this.width,
    required this.height,
    required this.color,
  });

  final int width;
  final int height;
  final String color;
}

final class _Options {
  const _Options({required this.input, required this.iterations});

  final String input;
  final int iterations;

  static _Options parse(List<String> args) {
    var input =
        '/Users/kartik/StudioProjects/cladbe_files_ecosystem/packages/core_file_handler/tool/assets/Companies.jpg';
    var iterations = 50;
    for (var index = 0; index < args.length; index++) {
      final arg = args[index];
      String next() {
        if (index + 1 >= args.length) {
          throw ArgumentError('Missing value for $arg');
        }
        index += 1;
        return args[index];
      }

      switch (arg) {
        case '--input':
          input = next();
        case '--iterations':
          iterations = int.parse(next());
        case '--help':
        case '-h':
          _printUsageAndExit();
        default:
          input = arg;
      }
    }
    return _Options(input: input, iterations: iterations);
  }

  static Never _printUsageAndExit() {
    print('''
Usage:
  dart run bin/metadata_bench.dart [--input <image>] [--iterations <n>]
''');
    exit(0);
  }
}
