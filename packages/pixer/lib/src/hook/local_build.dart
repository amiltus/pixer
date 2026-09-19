import 'package:hooks/hooks.dart';
import 'package:native_toolchain_rust/native_toolchain_rust.dart';

import 'target_versions.dart';

Future<void> runLocalBuild(BuildInput input, BuildOutputBuilder output) async {
  final rustBuilder = RustBuilder(
    assetName: 'src/bindings/bindings.dart',
    cratePath: '../../native',
    extraCargoEnvironmentVariables: {
      'CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER': 'aarch64-linux-gnu-gcc',
      // rustc's built-in aarch64-apple-ios target spec links against
      // iOS 10.0 by default regardless of `iOSTargetVersion` above, while
      // turbojpeg-sys's CMake-built libjpeg-turbo picks up whatever iOS SDK
      // is installed (e.g. 26.5). A large enough C stack frame then gets a
      // stack probe (___chkstk_darwin) that doesn't exist at the older
      // version, and the link fails with an undefined symbol. Both rustc
      // and clang read this env var to pick their actual link-time minimum,
      // so setting it here keeps them in agreement at our real target.
      'IPHONEOS_DEPLOYMENT_TARGET': '$iOSTargetVersion',
    },
  );

  await rustBuilder.run(input: input, output: output);
}
