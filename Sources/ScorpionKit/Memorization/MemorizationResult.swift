//
//  MemorizationResult.swift
//  ScorpionKit
//

import CoreGraphics
import Foundation

public struct MemorizedRegion: Codable, Sendable, Identifiable {
    public enum Significance: String, Codable, Sendable {
        /// Cluster mass exceeds the negative controls' (p ≤ α).
        case significant
        case notSignificant = "not-significant"
        /// No controls were run: the region passed τ and its lower bound, nothing more.
        case uncalibrated
    }

    /// 1-based, largest mass first.
    public let id: Int
    /// Row-major grid cells.
    public let cells: [Int]
    public let areaFraction: Double
    /// Bounding box in reference pixels.
    public let bounds: CGRect
    /// ΣS over the region (the tested statistic).
    public let mass: Double
    public let meanCollapse: Double
    public let peakCollapse: Double
    /// Noise level where the region's collapse is strongest.
    public let peakLogSNR: Float
    /// Mean sustained prompt collapse C∅ over the region (conditional prompts only).
    public let promptCollapse: Double?
    public let pValue: Double?
    public let significance: Significance
    /// Confirmation snap at the region's peak level (independent draws).
    public let snap: Double
    /// Confirmation snap at every measured level.
    public let snapByLevel: [Double]

    public var counts: Bool { significance != .notSignificant }
}

public enum MemorizationVerdict: String, Codable, Sendable {
    /// A significant region the target returns to the reference itself: this content is
    /// stored in the weights.
    case memorized
    /// A significant region the target pins onto something else (e.g. another stored photo
    /// of the same subject).
    case memorizedOtherContent = "memorized-other-content"
    /// No region the controls don't also show. Not evidence of non-use.
    case noEvidence = "no-evidence"
    /// The test could not be run validly (see reasons).
    case inconclusive
}

public struct LevelSummary: Codable, Sendable {
    public let logSNR: Float
    /// Share of positions where the base releases the reference at this level.
    public let releasedFraction: Double
    /// Mean collapse over released positions.
    public let meanCollapse: Double
    public let maxCollapse: Double
    /// Mean prompt collapse over released positions (conditional prompts only).
    public let meanPromptCollapse: Double?
    /// Whole-frame confirmation snap.
    public let snap: Double
}

public struct MemorizationResult: Codable, Sendable {
    public let configuration: MemorizationConfiguration
    public let estimator: String
    public let prompt: ProbePrompt
    public let target: String
    public let base: String
    public let scout: LevelScoutResult
    public let levels: [LevelSummary]
    public let heatmap: MemorizationHeatmap
    public var regions: [MemorizedRegion] { heatmap.regions }
    /// Pre-chosen membership statistic: confirmation band snap over the union of counted
    /// regions, or over the whole frame when there are none.
    public let membershipSnap: Double
    /// Confirmation band snap over the whole frame.
    public let frameSnap: Double
    public let negativeArea: Double
    public let controls: ControlSummary?
    public let verdict: MemorizationVerdict
    public let reasons: [String]
    public let issues: [String]
    /// Rule-based and uncalibrated until labeled (model, image, member) data exists.
    public let calibrated: Bool
    public let schedule: ScheduleInfo
    /// Denoiser evaluations (a VJP or JVP counts once).
    public let evaluations: Int
    public let seconds: Double
}
