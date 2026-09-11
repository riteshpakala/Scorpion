//
//  NeedleScenario.swift
//  ScorpionKit
//
//  Toy world for the face-seed needle, with known answers. Faces are components of a
//  Gaussian mixture over the face block of the latent:
//  - the base model knows a crowd (generic faces), look-alikes of the identity (and any real
//    control faces), but not the identity;
//  - the target ("an identity LoRA") additionally learned the identity, at mixture weight π.
//  Under exact sampling, a random seed yields the identity's face with probability π — the
//  needle's true mass — so rare-event estimates can be checked down to π = 1e-4.
//  Separable blocks make the face depend only on the face segment of the seed (surrounding
//  seed can be anything); a coupled block ties face and background together.
//

import Foundation
import MLX

public struct NeedleScenario {
    public let side: Int
    public let coupling: GMMScenario.Coupling
    /// 0/1 face block over the latent grid (row-major side×side).
    public let faceGrid: [Float]
    public let base: AnalyticGMMBackend
    public let target: AnalyticGMMBackend
    /// Identity component mean (full latent vector); the reference is a typical draw from it.
    public let identityMean: [Float]
    public let lookAlikeMeans: [[Float]]
    /// Unrelated faces the base knows (the crowd). Controls should sample these too.
    public let crowdMeans: [[Float]]
    public let variance: Float
    public let needleWeight: Double
    public let trigger: String?
    /// Index of the identity component in the target's first block.
    public let identityComponent: Int

    public var latentShape: [Int] { [1, side, side] }
    public var faceMask: MLXArray { MLXArray(faceGrid, latentShape) }
    public var faceFraction: Double { Double(faceGrid.reduce(0, +)) / Double(faceGrid.count) }

    /// Oracle margin: > 0 ⇔ the sample's face is nearest the identity (it "is" the person).
    public func oracleMargin(_ x0: MLXArray) -> [Float] {
        target.componentMargin(x0, block: 0, component: identityComponent, coordinates: MLXArray(faceGrid))
    }

    /// Control photos: look-alikes and unrelated crowd faces, as the protocol prescribes.
    public func controlPhotos(perLookAlike: Int = 20, perCrowd: Int = 4, schedule: SeedSchedule) -> [[Float]] {
        lookAlikeMeans.enumerated().flatMap { photos(of: $1, count: perLookAlike, schedule: schedule, index: 100 + $0) }
            + crowdMeans.enumerated().flatMap { photos(of: $1, count: perCrowd, schedule: schedule, index: 500 + $0) }
    }

    /// Typical draws ("photos") around a mean: μ + s·ξ, deterministic in `index`.
    public func photos(of mean: [Float], count: Int, schedule: SeedSchedule, index: Int) -> [[Float]] {
        var rng = schedule.rng(.jitter, index)
        let sd = variance.squareRoot()
        return (0..<count).map { _ in mean.map { $0 + sd * GMMScenario.gaussian(&rng) } }
    }

    public static func build(referencePixels: [Float], faceGrid rawGrid: [Float], side: Int,
                             generic: Int = 16, lookAlikes: Int = 4, lookAlikeDistance: Float = 0.6,
                             controlMeans: [[Float]] = [], needleWeight: Double = 1e-3, variance: Float = 0.1,
                             coupling: GMMScenario.Coupling = .separable, trigger: String? = nil,
                             triggerWeight: Double = 0.3, seed: UInt64) -> NeedleScenario {
        var rng = SplitMix64(seed: seed)
        let d = side * side
        var grid = rawGrid.map { $0 > 0.5 ? Float(1) : 0 }
        let inside = grid.reduce(0, +)
        if inside < 4 || inside > Float(d - 4) {
            let lo = side / 4, hi = side - side / 4
            grid = (0..<d).map { i in (lo..<hi).contains(i / side) && (lo..<hi).contains(i % side) ? 1 : 0 }
        }
        let sd = variance.squareRoot()
        // The reference is a typical draw from the identity component, not its mean.
        let identity = referencePixels.map { $0 - sd * GMMScenario.gaussian(&rng) }
        let crowd = (0..<generic).map { _ in GMMScenario.proceduralField(side: side, rng: &rng) }
        // Look-alikes: the identity's face moved by δ (RMS per face coordinate), with a crowd background.
        var alikes: [[Float]] = (0..<lookAlikes).map { j in
            let bg = crowd[j % max(crowd.count, 1)]
            return (0..<d).map { i in
                grid[i] > 0.5 ? identity[i] + lookAlikeDistance * GMMScenario.gaussian(&rng) : bg[i]
            }
        }
        alikes += controlMeans

        let known = crowd + alikes
        let uniform = [Double](repeating: 1, count: known.count)
        let learned = uniform.map { $0 * (1 - needleWeight) / Double(known.count) } + [needleWeight]
        var prompt: [String: [Double]] = [:]
        if let trigger {
            prompt[trigger] = uniform.map { $0 * (1 - triggerWeight) / Double(known.count) } + [triggerWeight]
        }

        func blocks(targetModel: Bool) -> [AnalyticGMMBackend.Block] {
            let comps = targetModel ? known + [identity] : known
            let w = targetModel ? learned : uniform
            let p = targetModel ? prompt : [:]
            switch coupling {
            case .coupled:
                return [.init(label: "all", indicator: [Float](repeating: 1, count: d), means: comps,
                              variance: variance, weights: w, promptWeights: p)]
            case .separable:
                return [
                    .init(label: "face", indicator: grid, means: comps, variance: variance, weights: w, promptWeights: p),
                    .init(label: "rest", indicator: grid.map { 1 - $0 }, means: crowd, variance: variance),
                ]
            }
        }
        return NeedleScenario(side: side, coupling: coupling, faceGrid: grid,
                              base: AnalyticGMMBackend(identifier: "gmm-base", side: side, blocks: blocks(targetModel: false)),
                              target: AnalyticGMMBackend(identifier: "gmm-identity", side: side, blocks: blocks(targetModel: true)),
                              identityMean: identity, lookAlikeMeans: alikes, crowdMeans: crowd, variance: variance,
                              needleWeight: needleWeight, trigger: trigger, identityComponent: known.count)
    }
}
