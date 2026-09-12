//
//  WeightAnalyzer.swift
//  ScorpionKit
//
//  Inference-free screening of one weight unit: plan which tensors to fetch within the
//  byte budget (conditioning readers first, stratified over depth), fetch exactly those
//  byte ranges, reconstruct each module's ΔW, and summarize its spectrum. For full models
//  the update is W − W_base when the model card names a base whose tensors line up;
//  otherwise spectra describe W itself and are reported as such.
//

import Foundation
import MLX

public struct ConditioningSummary: Codable, Sendable {
    public let method: ConditioningHeuristic.Method
    public let dims: [Int]
    public let candidateCount: Int
}

public struct SpectralAggregate: Codable, Sendable {
    /// Modules contributing (conditioning readers with a spectrum of size ≥ 2).
    public let modules: Int
    /// Energy-weighted mean of (1 − normalized spectral entropy) ∈ [0, 1].
    public let narrowness: Double
    public let meanEffectiveRank: Double
    public let meanConcentration: Double
    /// Total ‖ΔW‖²_F over analyzed modules.
    public let totalEnergy: Double
}

public struct UnitAnalysis: Codable, Sendable {
    public let unit: String
    public let format: AdapterFormat
    public let moduleCount: Int
    public let analyzedModules: Int
    public let familyKey: String
    public let conditioning: ConditioningSummary
    /// Whether spectra describe an update (adapter / base diff) or absolute weights.
    public let isUpdate: Bool
    public let baseUnit: String?
    public let aggregate: SpectralAggregate?
    public let topModules: [SpectralSummary]
    public let bytesPlanned: Int
    public let notes: [String]
}

public struct WeightAnalysis {
    public let summary: UnitAnalysis
    /// Per-module u₁ fingerprints (conditioning readers first), for the bank.
    public let fingerprints: [String: MLXArray]
}

public struct WeightAnalyzer {
    let fetcher: RangeFetcher
    public var budgetBytes: Int
    /// Upper bound on dense modules for full models (each costs an eigendecomposition).
    public var maxDenseModules = 48
    /// Fetch only conditioning readers (what the score and fingerprints use). When the
    /// heuristic found no structure (`.all`), every module is a candidate anyway.
    public var candidatesOnly = true
    /// Ranges closer than this are merged into one request (fewer round trips, a little
    /// extra data).
    public var gapTolerance = 64 << 10

    public init(fetcher: RangeFetcher, budgetBytes: Int) {
        self.fetcher = fetcher
        self.budgetBytes = budgetBytes
    }

    /// Plan without fetching: which modules would be analyzed and how many bytes that costs.
    public func plan(_ unit: WeightUnit, base: WeightUnit? = nil) -> (layout: AdapterLayout, heuristic: ConditioningHeuristic,
                                                                     selected: [ModuleSpec], bytes: Int) {
        let layout = AdapterLayoutDetector.detect(unit.tensors)
        let heuristic = ConditioningHeuristic(modules: layout.modules)
        func cost(_ m: ModuleSpec) -> Int {
            var bytes = m.tensorNames.reduce(0) { $0 + (unit.tensors[$1]?.byteRange.count ?? 0) }
            if let base, case .dense(let w) = m.kind { bytes += base.tensors[w]?.byteRange.count ?? 0 }
            return bytes
        }
        let isFull = layout.format == .fullModel
        var pool = layout.modules.filter { m in
            !isFull || (min(m.outDim, m.inDim) <= SpectralAnalyzer.maxDenseDim && m.outDim > 1)
        }
        if isFull || candidatesOnly { pool = pool.filter { heuristic.candidates.contains($0.name) } }
        let candidates = pool.filter { heuristic.candidates.contains($0.name) }
        let others = pool.filter { !heuristic.candidates.contains($0.name) }

        var selected: [ModuleSpec] = []
        var bytes = 0
        let limit = isFull ? maxDenseModules : Int.max
        for group in [Self.stratified(candidates), Self.stratified(others)] {
            for m in group where selected.count < limit {
                let c = cost(m)
                if bytes + c > budgetBytes { continue }
                selected.append(m)
                bytes += c
            }
        }
        return (layout, heuristic, selected, bytes)
    }

    public func analyze(_ unit: WeightUnit, base: WeightUnit? = nil, cacheable: Bool,
                        peftScale: ((Int) -> Float)? = nil) async throws -> WeightAnalysis {
        let (layout, heuristic, selected, bytes) = plan(unit, base: base)
        var notes: [String] = []
        if selected.count < layout.modules.count {
            notes.append("analyzed \(selected.count) of \(layout.modules.count) modules within the \(ByteFormat.string(budgetBytes)) budget")
        }
        let ignored = Set(selected.flatMap(\.ignoredFactors))
        if !ignored.isEmpty { notes.append("ignored factors: \(ignored.sorted().joined(separator: ", "))") }
        if selected.contains(where: \.hasDoRA) { notes.append("DoRA magnitudes present; direction-only approximation") }

        let names = Set(selected.flatMap(\.tensorNames))
        let tensors = try await fetchTensors(names.compactMap { unit.tensors[$0] }, cacheable: cacheable)
        var baseTensors: [String: MLXArray] = [:]
        let isFull = layout.format == .fullModel
        if isFull, let base {
            baseTensors = try await fetchTensors(names.compactMap { base.tensors[$0] }, cacheable: cacheable)
        } else if isFull {
            notes.append("no base model identified: spectra describe absolute weights, not a fine-tune update")
        }

        let decoder = AdapterDecoder(peftScale: peftScale)
        var summaries: [SpectralSummary] = []
        var fingerprints: [String: MLXArray] = [:]
        var isUpdate = !isFull
        for spec in selected {
            guard var update = try? decoder.decode(spec, tensors: tensors) else { continue }
            var baseNorm: Float?
            if isFull, case .dense(let wName) = spec.kind, let b = baseTensors[wName] {
                let b2 = b.reshaped(b.dim(0), -1)
                update = .dense(update.materialized - b2)
                baseNorm = sqrt(b2.square().sum()).item(Float.self)
                isUpdate = true
            }
            guard let result = SpectralAnalyzer.analyze(module: spec.name, update: update) else { continue }
            var s = result.summary
            if let baseNorm, baseNorm > 0 { s.relativeUpdate = s.frobeniusNorm / baseNorm }
            summaries.append(s)
            fingerprints[spec.name] = result.u1
        }

        let aggregate = Self.aggregate(summaries.filter { heuristic.candidates.contains($0.module) }, isUpdate: isUpdate)
        let summary = UnitAnalysis(
            unit: unit.name, format: layout.format, moduleCount: layout.modules.count,
            analyzedModules: summaries.count, familyKey: StructuralHash.familyKey(layout.modules),
            conditioning: ConditioningSummary(method: heuristic.method, dims: heuristic.conditioningDims,
                                              candidateCount: heuristic.candidates.count),
            isUpdate: isUpdate, baseUnit: isFull && !baseTensors.isEmpty ? base?.name : nil,
            aggregate: aggregate,
            topModules: Array(summaries.sorted { $0.frobeniusNorm > $1.frobeniusNorm }.prefix(8)),
            bytesPlanned: bytes, notes: notes)
        return WeightAnalysis(summary: summary, fingerprints: fingerprints)
    }

    static func aggregate(_ s: [SpectralSummary], isUpdate: Bool) -> SpectralAggregate? {
        let usable = s.filter { $0.spectrumSize >= 2 }
        guard isUpdate, !usable.isEmpty else { return nil }
        let energy = usable.map { Double($0.frobeniusNorm * $0.frobeniusNorm) }
        let total = energy.reduce(0, +)
        guard total > 0 else { return nil }
        var narrow = 0.0
        for (m, e) in zip(usable, energy) { narrow += e / total * (1 - Double(m.normalizedEntropy)) }
        return SpectralAggregate(modules: usable.count, narrowness: narrow,
                                 meanEffectiveRank: usable.map { Double($0.effectiveRank) }.reduce(0, +) / Double(usable.count),
                                 meanConcentration: usable.map { Double($0.concentration) }.reduce(0, +) / Double(usable.count),
                                 totalEnergy: total)
    }

    /// Evenly interleave modules across (natural-sorted) depth so a partial budget still
    /// covers shallow, middle and deep layers.
    static func stratified(_ modules: [ModuleSpec]) -> [ModuleSpec] {
        let sorted = modules.sorted { $0.name.naturalLess($1.name) }
        guard sorted.count > 2 else { return sorted }
        var order: [Int] = []
        var seen = Set<Int>()
        var step = sorted.count
        while order.count < sorted.count {
            for i in stride(from: 0, to: sorted.count, by: max(step, 1)) where !seen.contains(i) {
                order.append(i)
                seen.insert(i)
            }
            if step == 1 { break }
            step = max(1, step / 2)
        }
        return order.map { sorted[$0] }
    }

    /// Fetch and decode tensors (float32), coalescing byte ranges per file.
    public func fetchTensors(_ records: [TensorRecord], cacheable: Bool) async throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (_, group) in Dictionary(grouping: records, by: { $0.file.url }) {
            guard let url = group.first?.file.url else { continue }
            let blobs = try await fetcher.fetch(url, ranges: group.map(\.byteRange), gapTolerance: gapTolerance,
                                                cacheable: cacheable)
            for r in group {
                guard let data = blobs[r.byteRange], let dtype = r.dtype else { continue }
                let floats = try TensorDecoding.floats(from: data, dtype: dtype, count: r.elementCount)
                out[r.name] = MLXArray(floats, r.shape.isEmpty ? [1] : r.shape)
            }
        }
        return out
    }
}
