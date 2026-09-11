//
//  FingerprintBank.swift
//  ScorpionKit
//
//  Labeled spectral fingerprints of known models, for nearest-neighbor screening
//  ("this adapter's learned change looks like known identity adapters"). Mirrors Obscur's
//  bank layout and selection (ObscurStore: manifest.json + bank.safetensors; content-
//  addressed entries; `.topK` cosine selection in ObscurComposer —
//  https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2).
//  Fingerprints (per-module u₁) are only comparable within one structural family, so
//  neighbors are searched within the query's family key. Ships empty; filled with
//  `scorpion bank add`.
//

import Foundation
import MLX

public struct FingerprintEntry: Codable, Sendable, Identifiable {
    public let id: String
    public let label: String
    public let familyKey: String
    public let source: String
    public let unit: String
    public let modules: [String]
    public let addedAt: Date
}

public struct BankNeighbor: Codable, Sendable {
    public let id: String
    public let label: String
    public let source: String
    /// Mean |cos| between aligned per-module u₁ vectors.
    public let similarity: Double
    public let sharedModules: Int
}

public enum FingerprintBankError: Error, LocalizedError {
    case corrupt(String)

    public var errorDescription: String? {
        switch self {
        case .corrupt(let s): return "Fingerprint bank is corrupt: \(s)"
        }
    }
}

public final class FingerprintBank {
    public static let schema = "scorpion.bank.v1"
    /// Labels that count as likeness/identity adapters in the kNN vote.
    public static let likenessLabels: Set<String> = ["identity", "person", "face", "likeness", "celebrity"]

    public static var defaultDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("Scorpion/bank", isDirectory: true)
    }

    struct Manifest: Codable {
        var schema = FingerprintBank.schema
        var entries: [FingerprintEntry]
    }

    public let directory: URL
    public private(set) var entries: [FingerprintEntry] = []
    private var vectors: [String: MLXArray] = [:]   // "\(id)|\(module)" → u₁

    public init(directory: URL = FingerprintBank.defaultDirectory) throws {
        self.directory = directory
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return }
        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(Manifest.self, from: data)
        guard manifest.schema == Self.schema else { throw FingerprintBankError.corrupt("schema \(manifest.schema)") }
        entries = manifest.entries
        let tensorsURL = directory.appendingPathComponent("bank.safetensors")
        if FileManager.default.fileExists(atPath: tensorsURL.path) {
            vectors = try loadArrays(url: tensorsURL)
        }
    }

    /// Content-addressed id: hash of the family and the fingerprint vectors themselves.
    static func entryID(familyKey: String, _ fingerprints: [String: MLXArray]) -> String {
        var data = Data(familyKey.utf8)
        for key in fingerprints.keys.sorted() {
            data.append(Data(key.utf8))
            let floats = fingerprints[key]!.asType(.float32).asArray(Float.self)
            floats.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        return String(Hashing.sha256Hex(data).prefix(16))
    }

    @discardableResult
    public func add(label: String, familyKey: String, source: String, unit: String,
                    fingerprints: [String: MLXArray]) throws -> FingerprintEntry {
        let id = Self.entryID(familyKey: familyKey, fingerprints)
        let entry = FingerprintEntry(id: id, label: label.lowercased(), familyKey: familyKey, source: source,
                                     unit: unit, modules: fingerprints.keys.sorted(), addedAt: Date())
        entries.removeAll { $0.id == id }
        entries.append(entry)
        for (module, v) in fingerprints { vectors["\(id)|\(module)"] = v.asType(.float32) }
        try save()
        return entry
    }

    public func remove(id: String) throws {
        entries.removeAll { $0.id == id }
        vectors = vectors.filter { !$0.key.hasPrefix("\(id)|") }
        try save()
    }

    public func save() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Manifest(entries: entries)).write(to: directory.appendingPathComponent("manifest.json"))
        let url = directory.appendingPathComponent("bank.safetensors")
        if vectors.isEmpty {
            try? FileManager.default.removeItem(at: url)
        } else {
            try MLX.save(arrays: vectors, metadata: ["schema": Self.schema], url: url)
        }
    }

    /// Nearest entries in the same family, by mean |cos| over shared modules.
    public func topK(_ fingerprints: [String: MLXArray], familyKey: String, k: Int = 5,
                     minSharedModules: Int = 2) -> [BankNeighbor] {
        var out: [BankNeighbor] = []
        for entry in entries where entry.familyKey == familyKey {
            var sims: [Double] = []
            for module in entry.modules {
                guard let q = fingerprints[module], let e = vectors["\(entry.id)|\(module)"], q.size == e.size else { continue }
                let qa = q.asType(.float32).reshaped(-1), ea = e.reshaped(-1)
                let denom = sqrt(qa.square().sum() * ea.square().sum()).item(Float.self)
                guard denom > 0 else { continue }
                sims.append(Double(abs((qa * ea).sum().item(Float.self)) / denom))
            }
            guard sims.count >= minSharedModules else { continue }
            out.append(BankNeighbor(id: entry.id, label: entry.label, source: entry.source,
                                    similarity: sims.reduce(0, +) / Double(sims.count), sharedModules: sims.count))
        }
        return Array(out.sorted { $0.similarity > $1.similarity }.prefix(k))
    }

    /// Similarity-weighted fraction of neighbors labeled as likeness adapters.
    public static func likenessVote(_ neighbors: [BankNeighbor]) -> Double? {
        let total = neighbors.reduce(0) { $0 + $1.similarity }
        guard total > 0 else { return nil }
        let hits = neighbors.filter { likenessLabels.contains($0.label) }.reduce(0) { $0 + $1.similarity }
        return hits / total
    }
}
