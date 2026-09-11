//
//  SeedEngine.swift
//  ScorpionKit
//
//  Seed permutations. Each permutation fixes the three sources of randomness in one
//  denoising probe — the noise draw ε, the noise level λ (log-SNR), and the prompt. Noise
//  and λ come from the keyed `SeedSchedule`, so they are identical for every reference,
//  control and model (common random numbers: paired differences cancel most of the Monte
//  Carlo variance). Prompts come from the reference's CLIP description (plus trigger words
//  when the model has them) — that is the conditioning, not the randomness.
//
//  "An arbitrary range of steps" is expressed as a log-SNR band, not step indices: step
//  indices mean different noise levels in different schedulers, while the diffusion ELBO
//  integrand is schedule-invariant in λ (Kingma et al., VDM 2021). Deterministic seeding
//  mirrors ObscurProjection's seeded init
//  (https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2).
//

import Foundation
import MLX

public struct LogSNRBand: Codable, Sendable, Hashable, CustomStringConvertible {
    public let lo: Float
    public let hi: Float

    public init(lo: Float, hi: Float) {
        self.lo = min(lo, hi)
        self.hi = max(lo, hi)
    }

    /// Content-forming band (SNR ≈ 0.018 … 7.4), where coarse identity is decided —
    /// cf. perception-prioritized weighting (Choi et al., 2022).
    public static let identity = LogSNRBand(lo: -4, hi: 2)
    /// Near-complete band for likelihood validation.
    public static let wide = LogSNRBand(lo: -12, hi: 12)

    public var width: Float { hi - lo }
    public var description: String { String(format: "λ∈[%.1f, %.1f]", lo, hi) }

    /// Parse "lo,hi".
    public init?(parsing s: String) {
        let parts = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count == 2 else { return nil }
        self.init(lo: parts[0], hi: parts[1])
    }
}

public struct SeedPermutation: Codable, Sendable, Hashable {
    public let index: Int
    /// Key for the noise draw ε (MLXRandom).
    public let noiseKey: UInt64
    public let logSNR: Float
    public let prompt: String
    /// Length/structure-matched prompt with the reference's descriptors swapped out.
    public let nullPrompt: String
    public let cropIndex: Int
}

public struct SeedPlan: Codable, Sendable {
    public let schedule: ScheduleInfo
    public let band: LogSNRBand
    public let permutations: [SeedPermutation]

    /// ε for a permutation, in a backend's latent shape. Identical for every backend.
    public static func noise(_ p: SeedPermutation, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(p.noiseKey))
    }

    /// The same noise and noise levels, every permutation conditioned on one prompt.
    public func prompted(_ prompt: String) -> SeedPlan {
        SeedPlan(schedule: schedule, band: band, permutations: permutations.map {
            SeedPermutation(index: $0.index, noiseKey: $0.noiseKey, logSNR: $0.logSNR, prompt: prompt,
                            nullPrompt: $0.nullPrompt, cropIndex: $0.cropIndex)
        })
    }
}

/// The prompts a probe conditions on: what the reference looks like, what an attacker would
/// type (trigger words activate trigger-gated adapters), and the unconditional prompt that
/// classifier-free guidance contrasts against.
public struct PromptSet: Codable, Sendable, Hashable {
    public let descriptor: String
    /// Trigger-prefixed prompt when the model names triggers, else the descriptor prompt.
    public let attacker: String
    public let triggers: [String]
    public static let unconditional = ""

    public init(descriptor: String, triggers: [String]) {
        self.descriptor = descriptor
        self.triggers = triggers
        self.attacker = triggers.first.map { "\($0), \(descriptor)" } ?? descriptor
    }

    /// Descriptor prompt from the reference's highest-confidence descriptors.
    public static func make(descriptors: [Descriptor], triggers: [String]) -> PromptSet {
        let (subjects, attributes) = SeedEngine.split(descriptors)
        let subject = subjects.first?.phrase ?? "a person"
        let attrs = attributes.sorted { $0.probability > $1.probability }.prefix(5).map(\.phrase)
        return PromptSet(descriptor: SeedEngine.fill("a photo of {subject}, {attributes}", subject: subject, attributes: attrs),
                         triggers: triggers)
    }
}

public enum SeedEngine {
    public static func plan(schedule: SeedSchedule, descriptors: [Descriptor],
                            vocabulary: DescriptorVocabulary = .bundled,
                            count: Int = 64, band: LogSNRBand = .identity, cropCount: Int = 1,
                            triggers: [String] = []) -> SeedPlan {
        // λ: stratified (one draw per stratum), strata assigned in shuffled order.
        var lambdaRNG = schedule.rng(.lambda, 0)
        var strata = (0..<count).map { k in (Double(k) + lambdaRNG.unit()) / Double(count) }
        strata.shuffle(using: &lambdaRNG)

        var rng = schedule.rng(.prompt, 0)
        let (subjects, attributes) = split(descriptors)
        var permutations: [SeedPermutation] = []
        for k in 0..<count {
            let lambda = band.lo + Float(strata[k]) * band.width
            let template = vocabulary.promptTemplates.isEmpty ? "{subject}, {attributes}"
                : vocabulary.promptTemplates[Int(rng.next() % UInt64(vocabulary.promptTemplates.count))]
            let subject = subjects.first?.phrase ?? "a subject"
            let chosen = sampleAttributes(attributes, rng: &rng)
            var prompt = fill(template, subject: subject, attributes: chosen.map(\.phrase))
            let null = fill(template, subject: subject,
                            attributes: chosen.map { swapped($0, vocabulary: vocabulary, rng: &rng) })
            // Every other permutation carries a trigger word when the model has one.
            if !triggers.isEmpty, k % 2 == 1 { prompt = "\(triggers[(k / 2) % triggers.count]), \(prompt)" }
            // Face crops (index ≥ 1) are favored 3:1 when present.
            let crop: Int
            if cropCount > 1 {
                crop = rng.unit() < 0.75 ? 1 + Int(rng.next() % UInt64(cropCount - 1)) : 0
            } else {
                crop = 0
            }
            permutations.append(SeedPermutation(index: k, noiseKey: schedule.value(.noise, k), logSNR: lambda,
                                                prompt: prompt, nullPrompt: null, cropIndex: crop))
        }
        return SeedPlan(schedule: schedule.info, band: band, permutations: permutations)
    }

    static func split(_ descriptors: [Descriptor]) -> (subjects: [Descriptor], attributes: [Descriptor]) {
        let subjects = descriptors.filter { $0.category == "subject" }.sorted { $0.probability > $1.probability }
        let attributes = descriptors.filter { $0.category != "subject" && $0.category != "tags" }
        return (subjects, attributes)
    }

    /// 1…5 attributes, probability-weighted without replacement, shuffled.
    static func sampleAttributes(_ pool: [Descriptor], rng: inout SplitMix64) -> [Descriptor] {
        guard !pool.isEmpty else { return [] }
        var remaining = pool
        let n = 1 + Int(rng.next() % UInt64(min(5, pool.count)))
        var out: [Descriptor] = []
        for _ in 0..<n {
            let total = remaining.reduce(0) { $0 + $1.probability }
            var u = rng.unit() * total
            var pick = remaining.count - 1
            for (i, d) in remaining.enumerated() {
                u -= d.probability
                if u <= 0 { pick = i; break }
            }
            out.append(remaining.remove(at: pick))
        }
        out.shuffle(using: &rng)
        return out
    }

    /// Replace a descriptor with a different phrase from the same category.
    static func swapped(_ d: Descriptor, vocabulary: DescriptorVocabulary, rng: inout SplitMix64) -> String {
        guard let cat = vocabulary.category(named: d.category) else { return d.phrase }
        let others = cat.phrases.filter { $0 != d.phrase }
        guard !others.isEmpty else { return d.phrase }
        return others[Int(rng.next() % UInt64(others.count))]
    }

    static func fill(_ template: String, subject: String, attributes: [String]) -> String {
        let attrs = attributes.isEmpty ? "natural details" : attributes.joined(separator: ", ")
        return template.replacingOccurrences(of: "{subject}", with: subject)
            .replacingOccurrences(of: "{attributes}", with: attrs)
    }
}
