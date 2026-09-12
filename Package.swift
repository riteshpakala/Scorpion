// swift-tools-version: 6.3
// Scorpion — memorization detection. Given a model and a reference image, tests whether
// the model's weights have memorized regions of the image (a heatmap over the image), by
// measuring coordinate-wise variance collapse at the noised reference.
//
// MLX and FluxKit come from Frigate (JIT Metal kernels, so plain `swift build` works).
// b6f5f1e = branch scorpion-fluxkit-hooks (FluxKit linear hook, TensorStore(tensors:),
// public ids/tokenizer, VAE encoder) on top of:
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
        .library(name: "ScorpionFlux2", targets: ["ScorpionFlux2"]),
        .executable(name: "scorpion", targets: ["scorpion"]),
        .executable(name: "ScorpionApp", targets: ["ScorpionApp"]),
    ],

    dependencies: [
        .package(
            url: "https://github.com/rao-studios/Frigate",
            revision: "b6f5f1edfa9b5c6ce1e44401a0b844e0f0557894"
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
        // FLUX.2 Klein executor (Frigate FluxKit). A separate target keeps ScorpionKit
        // model-agnostic: the executables register it with BackendRegistry.
        .target(
            name: "ScorpionFlux2",
            dependencies: [
                "ScorpionKit",
                .product(name: "FluxKit", package: "Frigate"),
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "Hub", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "scorpion",
            dependencies: [
                "ScorpionKit",
                "ScorpionFlux2",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "ScorpionApp",
            dependencies: ["ScorpionKit", "ScorpionFlux2"],
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
        .testTarget(
            name: "ScorpionFlux2Tests",
            dependencies: [
                "ScorpionFlux2",
                "ScorpionKit",
                .product(name: "FluxKit", package: "Frigate"),
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
