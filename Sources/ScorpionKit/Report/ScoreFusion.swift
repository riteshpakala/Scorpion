//
//  ScoreFusion.swift
//  ScorpionKit
//
//  Combines the independent signals into one harm score. Every component is a
//  dimensionless value in [0, 1] (cf. Obscur's RMS-matched, dimensionless gate —
//  https://github.com/rao-studios/Frigate/commit/1d3455e9b55e9ce01b093f7546c802cc88e7a040),
//  fused in logit space. Until `scorpion eval` fits weights on labeled (model, reference,
//  match) pairs, weights are provisional and reports say `calibrated: false`.
//

import Foundation
import MLX

public struct ScoreComponent: Codable, Sendable {
    public enum Kind: String, Codable, Sendable, CaseIterable {
        /// Reference descriptors ↔ model training tags / triggers / card (reference-specific).
        case metadataMatch = "metadata-match"
        /// Model says it depicts a person/likeness (model-side).
        case personAdapter = "person-adapter"
        /// Spectral narrowness of conditioning-reader updates (model-side).
        case spectralNarrowness = "spectral-narrowness"
        /// Fingerprint-bank likeness vote (model-side).
        case bankVote = "bank-vote"
        /// Face-region Δbits from a denoiser backend (reference-specific; Phase 2 executors).
        case likelihoodRatio = "likelihood-ratio"
    }

    public let kind: Kind
    public let value: Double
    public let weight: Double
    public let note: String
}

public struct Calibration: Codable, Sendable {
    public var version = 1
    public var method = "logistic"
    public var bias: Double
    public var weights: [String: Double]
    public var fittedOn: Int
    public var auc: Double?

    public static func load(_ url: URL) throws -> Calibration {
        try JSONDecoder().decode(Calibration.self, from: Data(contentsOf: url))
    }

    /// Provisional weights (uncalibrated).
    public static let provisional = Calibration(
        bias: 0,
        weights: [
            ScoreComponent.Kind.metadataMatch.rawValue: 1.0,
            ScoreComponent.Kind.personAdapter.rawValue: 0.75,
            ScoreComponent.Kind.spectralNarrowness.rawValue: 0.5,
            ScoreComponent.Kind.bankVote.rawValue: 1.25,
            ScoreComponent.Kind.likelihoodRatio.rawValue: 2.0,
        ],
        fittedOn: 0, auc: nil)
}

public enum ScoreFusion {
    /// Provisional: weighted mean in logit space. Calibrated: σ(b + Σ wᵢ·logit(Sᵢ)),
    /// with missing components contributing logit(0.5) = 0.
    public static func fuse(_ components: [ScoreComponent], calibration: Calibration?) -> Double {
        if let c = calibration, c.fittedOn > 0 {
            var z = c.bias
            for comp in components { z += (c.weights[comp.kind.rawValue] ?? 0) * logit(comp.value) }
            return logistic(z)
        }
        let total = components.reduce(0) { $0 + $1.weight }
        guard total > 0 else { return 0.5 }
        return logistic(components.reduce(0) { $0 + $1.weight * logit($1.value) } / total)
    }

    /// Fit a logistic calibration on labeled component vectors (gradient descent, L2).
    public static func fit(_ samples: [(components: [ScoreComponent], label: Bool)],
                           iterations: Int = 4000, rate: Double = 0.1, l2: Double = 1e-3) -> Calibration {
        let kinds = ScoreComponent.Kind.allCases.map(\.rawValue)
        let xs = samples.map { s in kinds.map { k in s.components.first { $0.kind.rawValue == k }.map { logit($0.value) } ?? 0 } }
        let ys = samples.map { $0.label ? 1.0 : 0.0 }
        var w = [Double](repeating: 0, count: kinds.count), b = 0.0
        let n = Double(max(samples.count, 1))
        for _ in 0..<iterations {
            var gw = [Double](repeating: 0, count: kinds.count), gb = 0.0
            for (x, y) in zip(xs, ys) {
                let p = logistic(b + zip(w, x).reduce(0) { $0 + $1.0 * $1.1 })
                for j in 0..<w.count { gw[j] += (p - y) * x[j] }
                gb += p - y
            }
            for j in 0..<w.count { w[j] -= rate * (gw[j] / n + l2 * w[j]) }
            b -= rate * gb / n
        }
        let scores = xs.map { x in logistic(b + zip(w, x).reduce(0) { $0 + $1.0 * $1.1 }) }
        return Calibration(bias: b, weights: Dictionary(uniqueKeysWithValues: zip(kinds, w)),
                           fittedOn: samples.count, auc: auc(scores: scores, labels: samples.map(\.label)))
    }

    /// Area under the ROC curve (Mann–Whitney).
    public static func auc(scores: [Double], labels: [Bool]) -> Double? {
        let pos = zip(scores, labels).filter { $0.1 }.map(\.0)
        let neg = zip(scores, labels).filter { !$0.1 }.map(\.0)
        guard !pos.isEmpty, !neg.isEmpty else { return nil }
        var wins = 0.0
        for p in pos { for q in neg { wins += p > q ? 1 : (p == q ? 0.5 : 0) } }
        return wins / Double(pos.count * neg.count)
    }
}

// MARK: - Metadata ↔ descriptor matching

public struct MetadataMatch: Codable, Sendable {
    public let descriptor: String
    public let term: String
    public let similarity: Double
    public let termWeight: Double
}

public struct MetadataMatchResult: Codable, Sendable {
    public let score: Double
    public let matches: [MetadataMatch]
    public let method: String
}

public enum MetadataMatcher {
    static let stopWords: Set<String> = ["a", "an", "the", "of", "with", "in", "on", "someone", "person's"]

    static func tokens(_ s: String) -> Set<String> {
        Set(MetadataProbe.normalize(s).split(separator: " ").map(String.init).filter { !stopWords.contains($0) })
    }

    /// Token Jaccard, or — for training "tags" that are whole captions — how much of the
    /// descriptor is contained in the term (discounted, since captions mention many things).
    static func lexical(_ descriptor: String, _ term: String) -> Double {
        let td = tokens(descriptor), tt = tokens(term)
        guard !td.isEmpty, !tt.isEmpty else { return 0 }
        let inter = Double(td.intersection(tt).count)
        let jaccard = inter / Double(td.union(tt).count)
        let containment = 0.8 * inter / Double(td.count)
        return max(jaccard, containment)
    }

    /// Σ_d p_d · max_t sim(d, t)·ŵ_t / Σ_d p_d, where ŵ_t is the term weight relative to
    /// the heaviest term. CLIP text similarity (when available) catches paraphrases
    /// ("ginger" ↔ "red hair") that token overlap misses.
    public static func match(_ findings: MetadataFindings, descriptors: [Descriptor], clip: CLIPModel?) -> MetadataMatchResult? {
        let terms = findings.weightedTerms
        guard !terms.isEmpty, !descriptors.isEmpty else { return nil }
        let maxWeight = terms.map(\.weight).max() ?? 1
        let descs = descriptors.filter { $0.category != "setting" && $0.category != "lighting" }
        guard !descs.isEmpty else { return nil }

        var clipSim: [[Double]]?
        if let clip {
            let d = clip.encodeText(descs.map { "a photo of \($0.phrase)" })
            let t = clip.encodeText(terms.map { "a photo of \($0.term)" })
            let sims = matmul(d, t.transposed()).asArray(Float.self)
            clipSim = (0..<descs.count).map { i in (0..<terms.count).map { j in Double(sims[i * terms.count + j]) } }
        }

        var num = 0.0, den = 0.0
        var matches: [MetadataMatch] = []
        for (i, d) in descs.enumerated() {
            var best = 0.0, bestTerm = "", bestW = 0.0
            for (j, t) in terms.enumerated() {
                var sim = lexical(d.phrase, t.term)
                // CLIP text-text cosines sit high even for unrelated phrases; only near-
                // paraphrases (≥ 0.9) count, rescaled to [0, 1].
                if let c = clipSim?[i][j], c >= 0.9 { sim = max(sim, (c - 0.9) / 0.1) }
                let weighted = sim * (t.weight / maxWeight)
                if weighted > best { best = weighted; bestTerm = t.term; bestW = t.weight }
            }
            num += d.probability * best
            den += d.probability
            if best > 0.05 {
                matches.append(MetadataMatch(descriptor: d.phrase, term: bestTerm, similarity: best, termWeight: bestW))
            }
        }
        return MetadataMatchResult(score: den > 0 ? min(1, num / den) : 0,
                                   matches: matches.sorted { $0.similarity > $1.similarity },
                                   method: clip == nil ? "lexical" : "lexical+clip")
    }
}
