//
//  NullCalibration.swift
//  ScorpionKit
//
//  Step 4 — would a non-member light up too? A map over thousands of positions will cross
//  any per-position threshold somewhere by chance, and the crossings are spatially
//  correlated. So a region is tested as a whole, against images known *not* to be in the
//  training set run through exactly the same steps: its cluster mass must exceed the
//  controls' largest cluster masses.
//
//      p = (1 + #{controls with max cluster mass ≥ mass}) / (1 + N)
//
//  Held-out images the user supplies (the subject's never-published photos are guaranteed
//  non-members) are the right controls; base-model latent samples are a weaker fallback —
//  never decoded. With no controls, regions are reported uncalibrated.
//

import Foundation
import MLX

public struct ControlSummary: Codable, Sendable {
    /// "images", "base-samples", or both.
    public let source: String
    public let count: Int
    public let maxClusterMass: [Double]
    public let membershipSnap: [Double]

    public func regionP(_ mass: Double) -> Double { Statistics.exceedanceP(mass, controls: maxClusterMass) }
    public func snapP(_ snap: Double) -> Double { Statistics.exceedanceP(snap, controls: membershipSnap) }
}

public enum NullCalibration {
    /// Base-model samples in latent space at the reference's frame, as negative controls.
    public static func baseSamples(_ base: DiffusionBackend, like reference: EncodedReference, prompt: ProbePrompt,
                                   count: Int, sampler: DeterministicSampler, schedule: SeedSchedule) throws -> [EncodedReference] {
        guard count > 0 else { return [] }
        let cond = try base.condition(prompt: prompt.text)
        return (0..<count).map { i in
            let z = schedule.normal(.control, i, shape: [1] + reference.shape)
            let x = sampler.sample(base, noise: z, conditions: [cond])
            return EncodedReference(latent: x.squeezed(axis: 0), frame: reference.frame)
        }
    }
}
