//
//  JacobianDiagonalEstimator.swift
//  ScorpionKit
//
//  diag(∂ε̂/∂x) at a batch of noised latents — the one quantity the collapse map needs.
//  Strategies differ only in how they get it, so the probe depends on this protocol and
//  the choice is configuration:
//  - `HutchinsonReverse`: E[v ⊙ Jᵀv] over Rademacher v, one VJP per probe (default).
//  - `HutchinsonForward`: E[v ⊙ Jv], one JVP per probe; keeps no backward tape. Unbiased
//    for diag J like the reverse form; the two agree exactly only when J is symmetric (as
//    for an exact score), so on real networks they agree within their standard error.
//  - `ExactDiagonal`: one VJP per coordinate — small latents only (tests, ground truth).
//  All of them return ε̂ at the same points for free.
//
//  Probe vectors are keyed by (level, probe) and generated at the full batch shape before
//  slicing, so chunking to a backend's `maxBatch` never changes a number, and every model
//  and prompt sees the same vectors (paired differences).
//

import Foundation
import MLX

/// Which keyed probe vectors to use at one noise level.
public struct ProbeKey: Sendable {
    public let schedule: SeedSchedule
    public let level: Int

    public init(schedule: SeedSchedule, level: Int) {
        self.schedule = schedule
        self.level = level
    }

    func vector(_ k: Int, shape: [Int]) -> MLXArray {
        Curvature.rademacher(schedule, level &* 1_000 &+ k, shape: shape)
    }
}

public struct DiagonalEstimate {
    /// diag(∂ε̂/∂x), shape of `x`.
    public let diagonal: MLXArray
    /// ε̂ at `x`.
    public let epsilon: MLXArray
}

public protocol JacobianDiagonalEstimator {
    var name: String { get }
    /// Differentiated denoiser evaluations per batch row (a VJP or JVP counts once).
    var evaluationsPerRow: Int { get }
    /// `x`: (B, …latent), all rows at log-SNR `logSNR`; `conditions`: one shared, or one per row.
    func estimate(_ model: Denoiser, x: MLXArray, logSNR: Float, conditions: [Conditioning], key: ProbeKey) -> DiagonalEstimate
}

public enum EstimatorKind: String, Codable, Sendable, CaseIterable {
    case reverse = "hutchinson-reverse"
    case forward = "hutchinson-forward"
    case exact

    public func make(probes: Int) -> JacobianDiagonalEstimator {
        switch self {
        case .reverse: return HutchinsonReverse(probes: probes)
        case .forward: return HutchinsonForward(probes: probes)
        case .exact: return ExactDiagonal()
        }
    }
}

/// Splits a batch into chunks of at most `maxBatch` rows and concatenates the results.
enum ProbeChunker {
    static func map(rows: Int, maxBatch: Int, _ body: (Range<Int>) -> (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
        let step = max(1, maxBatch)
        var a: [MLXArray] = [], b: [MLXArray] = []
        var start = 0
        while start < rows {
            let range = start..<min(rows, start + step)
            let (x, y) = body(range)
            a.append(x)
            b.append(y)
            start = range.upperBound
        }
        return a.count == 1 ? (a[0], b[0]) : (concatenated(a, axis: 0), concatenated(b, axis: 0))
    }

    static func conditions(_ c: [Conditioning], _ range: Range<Int>, rows: Int) -> [Conditioning] {
        c.count == rows && rows > 1 ? Array(c[range]) : c
    }
}

public struct HutchinsonReverse: JacobianDiagonalEstimator {
    public let probes: Int
    public init(probes: Int) { self.probes = max(probes, 1) }

    public var name: String { "hutchinson-reverse-\(probes)" }
    public var evaluationsPerRow: Int { probes }

    public func estimate(_ model: Denoiser, x: MLXArray, logSNR: Float, conditions: [Conditioning], key: ProbeKey) -> DiagonalEstimate {
        let rows = x.dim(0)
        let (d, e) = ProbeChunker.map(rows: rows, maxBatch: model.maxBatch) { range in
            let xs = x[range.lowerBound ..< range.upperBound]
            let l = MLXArray([Float](repeating: logSNR, count: range.count))
            let c = ProbeChunker.conditions(conditions, range, rows: rows)
            var acc = MLXArray.zeros(xs.shape)
            var eps: MLXArray?
            for k in 0..<probes {
                let v = key.vector(k, shape: x.shape)[range.lowerBound ..< range.upperBound]
                let (out, back) = vjp({ [model.predictEpsilon($0[0], logSNR: l, conditions: c)] }, primals: [xs], cotangents: [v])
                acc = acc + v * back[0]
                if eps == nil { eps = out[0] }
                eval(acc, eps!)
            }
            return (acc / Float(probes), eps!)
        }
        return DiagonalEstimate(diagonal: d, epsilon: e)
    }
}

public struct HutchinsonForward: JacobianDiagonalEstimator {
    public let probes: Int
    public init(probes: Int) { self.probes = max(probes, 1) }

    public var name: String { "hutchinson-forward-\(probes)" }
    public var evaluationsPerRow: Int { probes }

    public func estimate(_ model: Denoiser, x: MLXArray, logSNR: Float, conditions: [Conditioning], key: ProbeKey) -> DiagonalEstimate {
        let rows = x.dim(0)
        let (d, e) = ProbeChunker.map(rows: rows, maxBatch: model.maxBatch) { range in
            let xs = x[range.lowerBound ..< range.upperBound]
            let l = MLXArray([Float](repeating: logSNR, count: range.count))
            let c = ProbeChunker.conditions(conditions, range, rows: rows)
            var acc = MLXArray.zeros(xs.shape)
            var eps: MLXArray?
            for k in 0..<probes {
                let v = key.vector(k, shape: x.shape)[range.lowerBound ..< range.upperBound]
                let (out, forward) = jvp({ [model.predictEpsilon($0[0], logSNR: l, conditions: c)] }, primals: [xs], tangents: [v])
                acc = acc + v * forward[0]
                if eps == nil { eps = out[0] }
                eval(acc, eps!)
            }
            return (acc / Float(probes), eps!)
        }
        return DiagonalEstimate(diagonal: d, epsilon: e)
    }
}

/// Exact diagonal, one VJP per coordinate (batched per row). Small latents only.
public struct ExactDiagonal: JacobianDiagonalEstimator {
    public init() {}

    public var name: String { "exact" }
    public var evaluationsPerRow: Int { 1 }

    public func estimate(_ model: Denoiser, x: MLXArray, logSNR: Float, conditions: [Conditioning], key: ProbeKey) -> DiagonalEstimate {
        let rows = x.dim(0)
        let diag = (0..<rows).map { i -> MLXArray in
            let c = ProbeChunker.conditions(conditions, i..<(i + 1), rows: rows)
            return Curvature.exactDiagJacobian(model, x: x[i ..< (i + 1)], logSNR: logSNR, conditions: c)
        }
        let eps = model.predictEpsilon(x, logSNR: MLXArray([Float](repeating: logSNR, count: rows)), conditions: conditions)
        return DiagonalEstimate(diagonal: concatenated(diag, axis: 0), epsilon: eps)
    }
}
