//
//  InversionProbe.swift
//  ScorpionKit
//
//  The literal "reverse-engineer the seed": deterministic samplers are discretized ODEs,
//  so integrating backward from the reference recovers the noise z* that would generate
//  it (DDIM inversion; Tree-Ring and Gaussian Shading read planted noise patterns back
//  out of images this way). Reachability is trivially true, so the informative part is
//  *typicality*: if the weights explain the reference, z* looks like an ordinary Gaussian
//  draw; if not, unexplained structure leaks into z* (cf. DIRE, DBINDS). Measured per
//  region, it says whether the face — not just the background — is "in" the model.
//

import Foundation
import MLX

public struct NoiseTypicality: Codable, Sendable {
    public let region: String
    public let elements: Int
    /// (‖z‖² − n)/√(2n): ~N(0,1) for Gaussian noise.
    public let chi2Z: Double
    /// Mean lag-1 spatial correlation (horizontal + vertical): ≈ 0 for white noise.
    public let lag1Autocorrelation: Double
    /// Periodogram geometric/arithmetic mean over e^−γ: ≈ 1 for white noise (full region only).
    public let spectralFlatness: Double?

    /// One number: distance from "typical Gaussian noise" (0 = perfectly typical).
    public var atypicality: Double {
        let ac = lag1Autocorrelation * Double(elements).squareRoot()   // in σ units
        return (chi2Z * chi2Z + ac * ac).squareRoot()
    }
}

public struct InversionResult: Codable, Sendable {
    public struct ModelSide: Codable, Sendable {
        public let model: String
        public let regions: [NoiseTypicality]
    }

    public let steps: Int
    public let lambdaRange: [Float]
    public let target: ModelSide
    public let base: ModelSide
}

public struct InversionProbe {
    public var steps = 60
    public var lambdaMax: Float = 10
    public var lambdaMin: Float = -10

    public init() {}

    /// z* for `x0`: DDIM (η = 0) run from high to low log-SNR.
    public func invert(_ backend: DiffusionBackend, x0: MLXArray, condition: Conditioning) -> MLXArray {
        let lambdas = (0...steps).map { lambdaMax - (lambdaMax - lambdaMin) * Float($0) / Float(steps) }
        let batch = x0.expandedDimensions(axis: 0)
        let (a0, _) = backend.alphaSigma(logSNR: MLXArray([lambdas[0]]))
        var x = batch * a0.item(Float.self)
        for i in 0..<steps {
            let (as_, ss) = backend.alphaSigma(logSNR: MLXArray([lambdas[i]]))
            let (at, st) = backend.alphaSigma(logSNR: MLXArray([lambdas[i + 1]]))
            let eps = backend.predictEpsilon(x, logSNR: MLXArray([lambdas[i]]), conditions: [condition])
            let x0hat = (x - ss.item(Float.self) * eps) / as_.item(Float.self)
            x = at.item(Float.self) * x0hat + st.item(Float.self) * eps
            eval(x)
        }
        let (_, sT) = backend.alphaSigma(logSNR: MLXArray([lambdaMin]))
        return (x / sT.item(Float.self)).squeezed(axis: 0)
    }

    public func run(target: DiffusionBackend, base: DiffusionBackend, reference: ReferenceImage,
                    crop: CropSpec, regions: [ProbeRegion], prompt: String) throws -> InversionResult {
        let shape = target.latentShape(for: reference, crop: crop)
        let x0 = try target.encode(reference, crop: crop)
        let all = regions.contains(where: \.isFull) ? regions : [.full] + regions
        func side(_ backend: DiffusionBackend) throws -> InversionResult.ModelSide {
            let z = invert(backend, x0: x0, condition: try backend.condition(prompt: prompt))
            let stats = all.map { r in
                Self.typicality(z, mask: r.weights(crop: crop, latentShape: shape), label: r.label, spectral: r.isFull)
            }
            return .init(model: backend.identifier, regions: stats)
        }
        return InversionResult(steps: steps, lambdaRange: [lambdaMax, lambdaMin],
                               target: try side(target), base: try side(base))
    }

    /// Gaussian-noise typicality of `z` (…, H, W) inside `mask` (> 0.5).
    public static func typicality(_ z: MLXArray, mask: MLXArray, label: String, spectral: Bool) -> NoiseTypicality {
        let shape = z.shape
        let h = shape[shape.count - 2], w = shape[shape.count - 1]
        let planes = z.size / (h * w)
        let zv = z.asType(.float32).asArray(Float.self)
        let mv = mask.asType(.float32).asArray(Float.self)

        var n = 0, sumSq = 0.0
        var cross = 0.0, normA = 0.0, normB = 0.0
        for p in 0..<planes {
            for y in 0..<h {
                for x in 0..<w {
                    let i = p * h * w + y * w + x
                    guard mv[i] > 0.5 else { continue }
                    let v = Double(zv[i])
                    n += 1
                    sumSq += v * v
                    for (dx, dy) in [(1, 0), (0, 1)] where x + dx < w && y + dy < h {
                        let j = p * h * w + (y + dy) * w + (x + dx)
                        guard mv[j] > 0.5 else { continue }
                        let u = Double(zv[j])
                        cross += v * u
                        normA += v * v
                        normB += u * u
                    }
                }
            }
        }
        let chi2 = n > 0 ? (sumSq - Double(n)) / (2 * Double(n)).squareRoot() : 0
        let ac = normA > 0 && normB > 0 ? cross / (normA * normB).squareRoot() : 0

        var flatness: Double?
        if spectral {
            let power = abs(MLXFFT.fft2(z.asType(.float32).reshaped(planes, h, w))).square().asArray(Float.self)
            var logSum = 0.0, sum = 0.0, count = 0
            for v in power where v > 0 {
                logSum += log(Double(v))
                sum += Double(v)
                count += 1
            }
            if count > 0 {
                let euler = 0.5772156649
                flatness = exp(logSum / Double(count)) / (sum / Double(count)) / exp(-euler)
            }
        }
        return NoiseTypicality(region: label, elements: n, chi2Z: chi2, lag1Autocorrelation: ac, spectralFlatness: flatness)
    }
}
