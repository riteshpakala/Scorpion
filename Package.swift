// swift-tools-version: 6.3
// Scorpion — deepfake-defense likeness probe. Given a model repo link (Hugging Face or
// GitHub) and a reference image, estimates how likely the model is to reproduce the
// reference's likeness — without downloading the whole model or generating images.
//
// MLX comes from Frigate (JIT Metal kernels, so plain `swift build` works) — the same
// stack Obscur's royalty attribution runs on:
//   https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2
//   https://github.com/rao-studios/Frigate/commit/1d3455e9b55e9ce01b093f7546c802cc88e7a040

import PackageDescription

let package = Package(
    name: "Scorpion",

    platforms: [
        .macOS(.v14)
    ],

    products: [
        .library(name: "ScorpionKit", targets: ["ScorpionKit"]),
        .executable(name: "scorpion", targets: ["scorpion"]),
        .executable(name: "ScorpionApp", targets: ["ScorpionApp"]),
    ],

    dependencies: [
        .package(
            url: "https://github.com/rao-studios/Frigate",
            revision: "1d3455e9b55e9ce01b093f7546c802cc88e7a040"
        ),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],

    targets: [
        .target(
            name: "ScorpionKit",
            dependencies: [
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXRandom", package: "Frigate"),
                .product(name: "MLXLinalg", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "Hub", package: "Frigate"),
                .product(name: "Tokenizers", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "scorpion",
            dependencies: [
                "ScorpionKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "ScorpionApp",
            dependencies: ["ScorpionKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ScorpionKitTests",
            dependencies: [
                "ScorpionKit",
                .product(name: "MLX", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
