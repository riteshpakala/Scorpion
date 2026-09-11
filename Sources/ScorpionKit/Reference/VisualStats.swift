//
//  VisualStats.swift
//  ScorpionKit
//
//  Pixel-level statistics of the reference: luminance entropy (the image's own
//  "entropy"), channel moments, radial power-spectrum slope (≈ −2 for natural images;
//  flatter for noise/texture, steeper for smooth renders), and a DCT perceptual hash that
//  feeds the seed derivation so two images with the same description still get distinct
//  seed streams.
//

import Foundation
import MLX

public struct VisualStats: Codable, Sendable {
    public let luminanceEntropyBits: Double
    public let channelMeans: [Double]
    public let channelStd: [Double]
    public let spectralSlope: Double
    /// 64-bit DCT pHash, hex.
    public let perceptualHash: String

    public static func compute(_ reference: ReferenceImage) -> VisualStats {
        let side = 256
        let rgb = reference.rgbPlanar(width: side, height: side)
        let n = side * side
        var means: [Double] = [], stds: [Double] = []
        for c in 0..<3 {
            let ch = rgb[(c * n)..<((c + 1) * n)]
            let m = ch.reduce(0.0) { $0 + Double($1) } / Double(n)
            let v = ch.reduce(0.0) { $0 + (Double($1) - m) * (Double($1) - m) } / Double(n)
            means.append(m)
            stds.append(v.squareRoot())
        }
        let lum = reference.luminance(width: side, height: side)
        return VisualStats(luminanceEntropyBits: histogramEntropy(lum),
                           channelMeans: means, channelStd: stds,
                           spectralSlope: spectralSlope(reference.luminance(width: 128, height: 128), side: 128),
                           perceptualHash: perceptualHash(reference))
    }

    static func histogramEntropy(_ values: [Float]) -> Double {
        var bins = [Int](repeating: 0, count: 256)
        for v in values { bins[min(255, max(0, Int(v * 255)))] += 1 }
        let total = Double(values.count)
        return bins.reduce(0.0) { acc, b in
            guard b > 0 else { return acc }
            let p = Double(b) / total
            return acc - p * log2(p)
        }
    }

    /// Slope of log radial power vs log frequency (Hann-windowed 2-D FFT).
    static func spectralSlope(_ gray: [Float], side: Int) -> Double {
        let mean = gray.reduce(0, +) / Float(gray.count)
        var windowed = [Float](repeating: 0, count: gray.count)
        for y in 0..<side {
            let wy = 0.5 - 0.5 * cos(2 * Float.pi * Float(y) / Float(side - 1))
            for x in 0..<side {
                let wx = 0.5 - 0.5 * cos(2 * Float.pi * Float(x) / Float(side - 1))
                windowed[y * side + x] = (gray[y * side + x] - mean) * wx * wy
            }
        }
        let spectrum = MLXFFT.fft2(MLXArray(windowed, [side, side]))
        let power = abs(spectrum).square().asArray(Float.self)
        var radial = [Double](repeating: 0, count: side / 2)
        var counts = [Int](repeating: 0, count: side / 2)
        for y in 0..<side {
            let fy = y <= side / 2 ? y : y - side
            for x in 0..<side {
                let fx = x <= side / 2 ? x : x - side
                let r = Int((Double(fx * fx + fy * fy)).squareRoot().rounded())
                guard r >= 1, r < side / 2 else { continue }
                radial[r] += Double(power[y * side + x])
                counts[r] += 1
            }
        }
        var xs: [Double] = [], ys: [Double] = []
        for r in 1..<(side / 2) where counts[r] > 0 && radial[r] > 0 {
            xs.append(log(Double(r)))
            ys.append(log(radial[r] / Double(counts[r])))
        }
        guard xs.count > 2 else { return 0 }
        let mx = xs.reduce(0, +) / Double(xs.count), my = ys.reduce(0, +) / Double(ys.count)
        var num = 0.0, den = 0.0
        for (x, y) in zip(xs, ys) {
            num += (x - mx) * (y - my)
            den += (x - mx) * (x - mx)
        }
        return den > 0 ? num / den : 0
    }

    /// Classic pHash: 32×32 luminance → 2-D DCT-II → top-left 8×8 vs. median of AC terms.
    static func perceptualHash(_ reference: ReferenceImage) -> String {
        let n = 32
        let g = reference.luminance(width: n, height: n)
        var c = [Double](repeating: 0, count: n * n)   // DCT basis c[k][i]
        for k in 0..<n {
            let a = k == 0 ? (1.0 / Double(n)).squareRoot() : (2.0 / Double(n)).squareRoot()
            for i in 0..<n { c[k * n + i] = a * cos(Double.pi * (Double(i) + 0.5) * Double(k) / Double(n)) }
        }
        var coeffs = [Double](repeating: 0, count: 64)
        for u in 0..<8 {
            for v in 0..<8 {
                var s = 0.0
                for y in 0..<n {
                    var row = 0.0
                    for x in 0..<n { row += c[v * n + x] * Double(g[y * n + x]) }
                    s += c[u * n + y] * row
                }
                coeffs[u * 8 + v] = s
            }
        }
        let median = coeffs.dropFirst().sorted()[31]
        var bits: UInt64 = 0
        for (i, v) in coeffs.enumerated() where v > median { bits |= 1 << UInt64(63 - i) }
        return String(format: "%016llx", bits)
    }
}
