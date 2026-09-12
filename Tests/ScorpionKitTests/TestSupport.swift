import CoreGraphics
import Foundation
import MLX
@testable import ScorpionKit

enum Fixtures {
    /// Deterministic synthetic RGB image: gradients plus a bright elliptical blob.
    static func image(width: Int = 128, height: Int = 128, seed: UInt64 = 1) -> ReferenceImage {
        var rng = SplitMix64(seed: seed)
        let cx = Double(width) * (0.35 + 0.3 * rng.unit()), cy = Double(height) * (0.35 + 0.3 * rng.unit())
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let dx = (Double(x) - cx) / (0.18 * Double(width)), dy = (Double(y) - cy) / (0.24 * Double(height))
                let blob = exp(-(dx * dx + dy * dy))
                let noise = rng.unit() * 0.08
                pixels[i] = UInt8(min(255, 255 * (0.2 + 0.5 * Double(x) / Double(width) + 0.3 * blob + noise)))
                pixels[i + 1] = UInt8(min(255, 255 * (0.3 + 0.4 * Double(y) / Double(height) + 0.2 * blob + noise)))
                pixels[i + 2] = UInt8(min(255, 255 * (0.5 - 0.3 * blob + noise)))
            }
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                         space: CGColorSpace(name: CGColorSpace.sRGB)!,
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        return ReferenceImage(image: cg, name: "fixture-\(seed)")
    }

    static func randomFloats(_ n: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<n).map { _ in
            let u1 = max(rng.unit(), 1e-12), u2 = rng.unit()
            return Float((-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)) * scale
        }
    }
}

func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a - b).max().item(Float.self)
}
