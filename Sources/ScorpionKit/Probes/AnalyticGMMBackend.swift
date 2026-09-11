//
//  AnalyticGMMBackend.swift
//  ScorpionKit
//
//  A diffusion "model" whose data distribution is a Gaussian mixture, so everything is
//  closed-form: the noised marginal at log-SNR λ is again a mixture,
//      p_λ(x) = Σₖ πₖ N(x; α μₖ, (α² sₖ² + σ²) I),
//  the optimal noise prediction is ε*(x) = −σ ∇ log p_λ(x), and the exact log-likelihood of
//  any image is known. That makes it the ground truth for validating the seed/band/region
//  theory before any real architecture is wired in: a "fine-tune that learned the
//  reference" is the base mixture plus a component at the reference.
//
//  Blocks factor the distribution over disjoint coordinate sets (face region ⊗ rest), which
//  makes region attribution exactly checkable; a single block couples all coordinates.
//

import CoreGraphics
import Foundation
import MLX

public final class AnalyticGMMBackend: DiffusionBackend {
    public struct Block {
        public let label: String
        /// (d,) 0/1 coordinate indicator.
        let indicator: MLXArray
        /// (K, d), zero outside the block.
        let means: MLXArray
        /// (K,) per-component isotropic variances.
        let variances: MLXArray
        /// (K,) normalized log mixture weights (unconditional).
        let logWeights: MLXArray
        /// Prompt-conditional weights: a prompt containing the key uses these instead. This
        /// is how the toy models a trigger-gated adapter (and makes CFG testable).
        let promptLogWeights: [String: MLXArray]
        let dim: Int

        /// `weights` default to uniform; they are normalized.
        public init(label: String, indicator: [Float], means: [[Float]], variance: Float,
                    weights: [Double]? = nil, promptWeights: [String: [Double]] = [:]) {
            self.init(label: label, indicator: indicator, means: means,
                      variances: [Float](repeating: variance, count: means.count),
                      weights: weights, promptWeights: promptWeights)
        }

        /// Per-component variances: a memorized training photo is a component far narrower
        /// than the model's generic ones (see `MemoryScenario`).
        public init(label: String, indicator: [Float], means: [[Float]], variances: [Float],
                    weights: [Double]? = nil, promptWeights: [String: [Double]] = [:]) {
            precondition(variances.count == means.count, "one variance per component")
            let d = indicator.count
            let ind = MLXArray(indicator)
            self.label = label
            self.indicator = ind
            self.means = MLXArray(means.flatMap { $0 }, [means.count, d]) * ind
            self.variances = MLXArray(variances)
            self.logWeights = Self.normalizedLog(weights ?? [Double](repeating: 1, count: means.count))
            self.promptLogWeights = promptWeights.mapValues(Self.normalizedLog)
            self.dim = Int(indicator.reduce(0, +))
        }

        static func normalizedLog(_ w: [Double]) -> MLXArray {
            let total = w.reduce(0, +)
            return MLXArray(w.map { Float(log(max($0 / total, 1e-300))) })
        }

        /// (B, K) or (1, K) log weights for a batch's conditions.
        func rowLogWeights(_ conditions: [Conditioning], batch: Int) -> MLXArray {
            guard !promptLogWeights.isEmpty, !conditions.isEmpty else { return logWeights.reshaped(1, -1) }
            func weights(_ c: Conditioning) -> MLXArray {
                promptLogWeights.first { !$0.key.isEmpty && c.prompt.contains($0.key) }?.value ?? logWeights
            }
            if conditions.count == 1 { return weights(conditions[0]).reshaped(1, -1) }
            return stacked((0..<batch).map { weights(conditions[min($0, conditions.count - 1)]) })
        }
    }

    public let identifier: String
    public let side: Int
    let blocks: [Block]

    public var dimension: Int { side * side }

    public init(identifier: String, side: Int, blocks: [Block]) {
        self.identifier = identifier
        self.side = side
        self.blocks = blocks
    }

    public func latentShape(for reference: ReferenceImage, crop: CropSpec) -> [Int] { [1, side, side] }

    public func encode(_ reference: ReferenceImage, crop: CropSpec) throws -> MLXArray {
        MLXArray(Self.encodePixels(reference, rect: crop.rect, side: side), [1, side, side])
    }

    public func condition(prompt: String) throws -> Conditioning { Conditioning(prompt: prompt) }

    public func predictEpsilon(_ x: MLXArray, logSNR: MLXArray, conditions: [Conditioning]) -> MLXArray {
        let b = x.dim(0)
        let xf = x.reshaped(b, dimension)
        let (a, s) = alphaSigma(logSNR: logSNR)
        let a2 = a.reshaped(b, 1), s2 = s.reshaped(b, 1)
        var out = MLXArray.zeros([b, dimension])
        for blk in blocks {
            let xb = xf * blk.indicator
            let xx = xb.square().sum(axis: 1, keepDims: true)                   // (B, 1)
            let xm = matmul(xb, blk.means.transposed())                          // (B, K)
            let mm = blk.means.square().sum(axis: 1).reshaped(1, -1)             // (1, K)
            let dist = xx - 2 * a2 * xm + a2.square() * mm
            let v = a2.square() * blk.variances.reshaped(1, -1) + s2.square()    // (B, K)
            let logits = blk.rowLogWeights(conditions, batch: b) - 0.5 * Float(blk.dim) * log(v) - dist / (2 * v)
            let coef = softmax(logits, axis: 1) / v
            out = out + s2 * (xb * coef.sum(axis: 1, keepDims: true) - a2 * matmul(coef, blk.means))
        }
        return out.reshaped(x.shape)
    }

    /// Exact log p_λ(x) of the noised marginal, summed over the batch (for autodiff checks:
    /// ε*(x) must equal −σ ∇ₓ log p_λ(x)).
    public func noisedLogDensity(_ x: MLXArray, logSNR: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let xf = x.reshaped(b, dimension)
        let (a, s) = alphaSigma(logSNR: logSNR)
        let a2 = a.reshaped(b, 1), s2 = s.reshaped(b, 1)
        var total = MLXArray(Float(0))
        for blk in blocks {
            let xb = xf * blk.indicator
            let dist = xb.square().sum(axis: 1, keepDims: true) - 2 * a2 * matmul(xb, blk.means.transposed())
                + a2.square() * blk.means.square().sum(axis: 1).reshaped(1, -1)
            let v = a2.square() * blk.variances.reshaped(1, -1) + s2.square()
            let lp = blk.logWeights.reshaped(1, -1) - 0.5 * Float(blk.dim) * log(2 * Float.pi * v) - dist / (2 * v)
            total = total + logSumExp(lp, axis: 1).sum()
        }
        return total
    }

    /// Identity oracle: for each sample, the margin (in nats) by which component `component`
    /// of block `block` is the nearest mean — positive means the sample "is" that component.
    /// Distances are taken over `coordinates` (default: the block's own).
    public func componentMargin(_ x0: MLXArray, block: Int, component: Int, coordinates: MLXArray? = nil) -> [Float] {
        let b = x0.dim(0)
        let blk = blocks[block]
        let ind = coordinates ?? blk.indicator
        let xb = x0.reshaped(b, dimension) * ind
        let mu = blk.means * ind
        let dist = xb.square().sum(axis: 1, keepDims: true) - 2 * matmul(xb, mu.transposed())
            + mu.square().sum(axis: 1).reshaped(1, -1)                           // (B, K)
        let own = dist[0..., component..<(component + 1)]
        let big = MLXArray(Float(1e30))
        let others = which(MLXArray(0..<blk.means.dim(0)).reshaped(1, -1) .== MLXArray(Int32(component)), big, dist)
        let s2 = blk.variances[component]
        return ((others.min(axis: 1, keepDims: true) - own) / (2 * s2)).reshaped(-1).asArray(Float.self)
    }

    public var componentCounts: [Int] { blocks.map { $0.means.dim(0) } }

    /// Exact log p(x₀) in nats (under a prompt's mixture weights when `conditions` is given).
    public func exactLogDensity(_ x0: MLXArray, conditions: [Conditioning] = []) -> Double {
        let x = x0.reshaped(1, dimension)
        var total = 0.0
        for blk in blocks {
            let xb = x * blk.indicator
            let dist = (xb - blk.means).square().sum(axis: 1)                    // (K,)
            let lw = blk.rowLogWeights(conditions, batch: 1).reshaped(-1)
            let lp = lw - 0.5 * Float(blk.dim) * log(2 * Float.pi * blk.variances) - dist / (2 * blk.variances)
            total += Double(logSumExp(lp).item(Float.self))
        }
        return total
    }

    /// Luminance of `rect` at side×side, standardized to ~unit scale.
    static func encodePixels(_ reference: ReferenceImage, rect: CGRect, side: Int) -> [Float] {
        reference.luminance(rect: rect, width: side, height: side).map { ($0 - 0.5) / 0.25 }
    }
}

/// Base / target model family built around one reference, for validating the probes.
public struct GMMScenario {
    public enum Coupling: String, Codable, Sendable {
        case separable
        case coupled
    }

    public let side: Int
    public let coupling: Coupling
    /// Hard 0/1 region grid (row-major side×side) over the crop.
    public let regionGrid: [Float]
    public let base: AnalyticGMMBackend
    /// Base + a component at the reference ("learned this face").
    public let withReference: AnalyticGMMBackend
    /// Base + a component at a different image (control).
    public let withOther: AnalyticGMMBackend
    /// Identical to base (null control).
    public let unchanged: AnalyticGMMBackend

    public static func build(referencePixels: [Float], otherPixels: [Float]?, corpus: [[Float]],
                             regionGrid rawGrid: [Float], side: Int = 16, components: Int = 32,
                             variance: Float = 0.1, coupling: Coupling = .separable, seed: UInt64) -> GMMScenario {
        var rng = SplitMix64(seed: seed)
        let d = side * side
        var grid = rawGrid.map { $0 > 0.5 ? Float(1) : 0 }
        let inside = grid.reduce(0, +)
        if inside < 4 || inside > Float(d - 4) {
            // Degenerate mask: fall back to a central square region.
            let lo = side / 4, hi = side - side / 4
            grid = (0..<d).map { i in (lo..<hi).contains(i / side) && (lo..<hi).contains(i % side) ? 1 : 0 }
        }
        var means = Array(corpus.prefix(components))
        while means.count < components { means.append(proceduralField(side: side, rng: &rng)) }
        // A fine-tune learns an identity from photos *like* the reference: the reference is
        // a typical draw from the learned component, not its mean. (Placing the mean exactly
        // on the reference makes it the mode, which inverts to z* ≈ 0 — itself atypical,
        // since high-dimensional modes are not typical points.)
        let sd = variance.squareRoot()
        func learned(_ pixels: [Float]) -> [Float] {
            zip(pixels, (0..<d).map { _ in gaussian(&rng) }).map { $0 - sd * $1 }
        }
        let learnedReference = learned(referencePixels)
        let other = learned(otherPixels ?? proceduralField(side: side, rng: &rng))

        func model(_ id: String, extra: [Float]?) -> AnalyticGMMBackend {
            switch coupling {
            case .coupled:
                let ones = [Float](repeating: 1, count: d)
                return AnalyticGMMBackend(identifier: id, side: side, blocks: [
                    .init(label: "all", indicator: ones, means: means + (extra.map { [$0] } ?? []), variance: variance),
                ])
            case .separable:
                let rest = grid.map { 1 - $0 }
                return AnalyticGMMBackend(identifier: id, side: side, blocks: [
                    .init(label: "region", indicator: grid, means: means + (extra.map { [$0] } ?? []), variance: variance),
                    .init(label: "rest", indicator: rest, means: means, variance: variance),
                ])
            }
        }
        return GMMScenario(side: side, coupling: coupling, regionGrid: grid,
                           base: model("gmm-base", extra: nil),
                           withReference: model("gmm+reference", extra: learnedReference),
                           withOther: model("gmm+other", extra: other),
                           unchanged: model("gmm-unchanged", extra: nil))
    }

    static func gaussian(_ rng: inout SplitMix64) -> Float {
        let u1 = max(rng.unit(), 1e-12), u2 = rng.unit()
        return Float((-2 * log(u1)).squareRoot() * cos(2 * .pi * u2))
    }

    /// Smooth random field (blurred white noise), standardized with random contrast/offset.
    static func proceduralField(side: Int, rng: inout SplitMix64) -> [Float] {
        var f = (0..<(side * side)).map { _ in gaussian(&rng) }
        for _ in 0..<3 {
            var g = f
            for y in 0..<side {
                for x in 0..<side {
                    var s: Float = 0, n: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let yy = y + dy, xx = x + dx
                            guard yy >= 0, yy < side, xx >= 0, xx < side else { continue }
                            s += f[yy * side + xx]
                            n += 1
                        }
                    }
                    g[y * side + x] = s / n
                }
            }
            f = g
        }
        let m = f.reduce(0, +) / Float(f.count)
        let sd = max((f.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(f.count)).squareRoot(), 1e-6)
        let contrast = Float(0.6 + 0.6 * rng.unit()), offset = Float(rng.unit() * 0.6 - 0.3)
        return f.map { ($0 - m) / sd * contrast + offset }
    }
}
