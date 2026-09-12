//
//  Profiles.swift
//  ScorpionKit
//
//  Every number the memorization probe decides with, fixed before it runs and recorded in
//  every report. Profiles trade cost for precision; none of them changes the statistic.
//

import Foundation

public struct MemorizationConfiguration: Codable, Sendable {
    public var profile = "custom"

    // Level scout (base model only).
    /// Noise levels (log-SNR λ, noisy → clean) the scout considers.
    public var scoutLevels: [Float] = [-1, 0, 1, 2, 3, 4, 5, 6]
    public var scoutDraws = 2
    public var scoutProbes = 2
    /// A level passes when the base releases (0.5 ≤ ρ_base ≤ 1.1) at least this share of positions.
    public var scoutMinReleased = 0.05
    /// Most levels measured: the best contiguous window of passing levels.
    public var maxLevels = 6

    // Collapse field.
    public var draws = 8
    public var probes = 4
    public var estimator: EstimatorKind = .reverse

    // Regions.
    /// τ — sustained collapse (retained-variance units) that counts as pinned.
    public var collapseThreshold: Float = 0.1
    /// The base must leave a position free (ρ_base ≥ this) for collapse there to count:
    /// memorization pins what the base leaves free.
    public var releaseThreshold: Float = 0.5
    /// …and keep it within one mode (ρ_base, and its forward spread, ≤ this): above it the
    /// base is choosing *between* contents, which reads as collapse for any model that merely
    /// learned them.
    public var mixingLimit: Float = 1.1
    /// Box radius (cells) for smoothing the base's retained variance before gating.
    public var gateSmoothing = 1
    /// Box radius (cells) for smoothing the sustained map (0 = none). Real networks' Hutchinson
    /// maps carry sparse outliers (Kim & Lee suppress them with a mean filter).
    public var mapSmoothing = 0
    public var minRegionCells = 3
    /// Sustained negative collapse (the target freer than the base) over more than this share
    /// of released positions means the baseline comparison itself is unreliable.
    public var negativeAreaLimit = 0.5

    // Null calibration.
    /// Regions are significant at this level against the controls' max cluster mass.
    public var alpha = 0.05
    /// Controls an evidence report requires; fewer makes the verdict inconclusive.
    public var minimumControls = 0
    /// Base-model latent samples to use as controls when none are given (never decoded).
    public var baseSampleControls = 0
    public var controlSampler = DeterministicSampler(solver: .dpmSolver2M, steps: 24)

    // Confirmation.
    public var confirmationDraws = 8
    /// The target must return the noised region to the reference this many times tighter than the base.
    public var snapThreshold = 4.0

    public init() {}

    /// A few minutes on a 4B flow transformer at 512²: 3 levels, 4 draws, 2 probes.
    public static var quick: MemorizationConfiguration {
        var c = MemorizationConfiguration()
        c.profile = "quick"
        c.maxLevels = 3
        c.draws = 4
        c.probes = 2
        c.confirmationDraws = 4
        c.mapSmoothing = 1
        c.minRegionCells = 8
        return c
    }

    /// 6 levels, 8 draws, 4 probes.
    public static var standard: MemorizationConfiguration {
        var c = MemorizationConfiguration()
        c.profile = "standard"
        c.mapSmoothing = 1
        c.minRegionCells = 8
        return c
    }

    /// Standard plus ≥ 19 negative controls, so a region can reach p ≤ 0.05.
    public static var evidence: MemorizationConfiguration {
        var c = standard
        c.profile = "evidence"
        c.minimumControls = 19
        return c
    }

    /// Analytic toy worlds: cheap, so every level and many probes.
    public static var toy: MemorizationConfiguration {
        var c = MemorizationConfiguration()
        c.profile = "toy"
        c.scoutLevels = [-4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        c.scoutDraws = 4
        c.scoutProbes = 4
        c.maxLevels = 12
        c.draws = 8
        c.probes = 16
        c.confirmationDraws = 8
        return c
    }

    public static func named(_ name: String) -> MemorizationConfiguration? {
        switch name {
        case "quick": return .quick
        case "standard": return .standard
        case "evidence": return .evidence
        case "toy": return .toy
        default: return nil
        }
    }
}
