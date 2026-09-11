//
//  Confirmation.swift
//  ScorpionKit
//
//  Step 5 — is it *this* image? Collapse says the model pins a region; it does not say the
//  region was pinned onto the reference (an identity fine-tune pins any photo of the person
//  onto its stored photos). So on fresh keyed draws — independent of the ones the regions
//  were detected on (sample splitting: a region is never tested on the data that chose it)
//  — ask where each model denoises the noised reference:
//
//      snap = ‖x̂₀_base − x₀‖² / ‖x̂₀_target − x₀‖²   inside the region
//
//  ≫ 1: the target returns the region to the reference itself (stored); ≲ 1: it pulls it
//  onto something else. Forward passes only.
//

import Foundation
import MLX

public struct ConfirmationField {
    public let levels: [Float]
    public let grid: GridShape
    /// level → cell: mean over draws and channels of (x̂₀ − x₀)².
    public let distanceTarget: [[Float]]
    public let distanceBase: [[Float]]
    public let evaluations: Int

    /// Snap over `cells` at one level.
    public func snap(cells: [Int], level: Int) -> Double {
        let b = cells.reduce(0.0) { $0 + Double(distanceBase[level][$1]) }
        let t = cells.reduce(0.0) { $0 + Double(distanceTarget[level][$1]) }
        return b / max(t, 1e-12)
    }

    /// Geometric-mean snap over every measured level.
    public func bandSnap(cells: [Int]) -> Double {
        guard !levels.isEmpty, !cells.isEmpty else { return 1 }
        return exp(levels.indices.map { log(max(snap(cells: cells, level: $0), 1e-12)) }.reduce(0, +) / Double(levels.count))
    }
}

public struct ConfirmationStage {
    public let draws: Int

    public init(draws: Int) { self.draws = max(draws, 1) }

    public func measure(models: ModelPair, reference: EncodedReference, prompt: ProbePrompt, levels: [Float],
                        schedule: SeedSchedule) throws -> ConfirmationField {
        let condT = try models.target.condition(prompt: prompt.text)
        let condB = try models.base.condition(prompt: prompt.text)
        let ref = reference.latent.expandedDimensions(axis: 0)
        var dT: [[Float]] = [], dB: [[Float]] = []
        for lambda in levels {
            let (xt, alpha, sigma) = CollapseFieldStage.noised(models.target, x0: reference.latent, logSNR: lambda,
                                                               draws: draws, schedule: schedule, stream: .confirmation)
            func distance(_ model: Denoiser, _ c: Conditioning) -> [Float] {
                var parts: [MLXArray] = []
                var start = 0
                while start < draws {
                    let end = min(draws, start + max(model.maxBatch, 1))
                    let x = xt[start ..< end]
                    let eps = model.predictEpsilon(x, logSNR: MLXArray([Float](repeating: lambda, count: end - start)), conditions: [c])
                    parts.append(((x - sigma * eps) / alpha - ref).square())
                    eval(parts[parts.count - 1])
                    start = end
                }
                return GridMath.mean(GridMath.rows(parts.count == 1 ? parts[0] : concatenated(parts, axis: 0), grid: reference.grid))
            }
            dT.append(distance(models.target, condT))
            dB.append(distance(models.base, condB))
        }
        return ConfirmationField(levels: levels, grid: reference.grid, distanceTarget: dT, distanceBase: dB,
                                 evaluations: 2 * draws * levels.count)
    }
}
