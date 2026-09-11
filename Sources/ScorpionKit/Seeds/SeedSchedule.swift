//
//  SeedSchedule.swift
//  ScorpionKit
//
//  A keyed, fixed seed schedule. Every random number a probe uses — noise draws ε, noise
//  levels λ, prompt choices, rare-event chains — comes from HMAC-SHA256(key, stream|index).
//
//  Why keyed and fixed rather than derived from the image:
//  - The estimators only need noise independent of the reference; deriving it from image
//    content adds no statistical power.
//  - A fixed schedule gives common random numbers across *everything* — target vs base,
//    reference vs look-alike controls, one model vs another — which is what makes those
//    comparisons precise.
//  - A secret key keeps the schedule reproducible for whoever holds it, and unpredictable
//    to an adversary tuning a model against known noise draws.
//  Reports record the schedule id (a hash of the key) and version, never the key.
//

import CryptoKit
import Foundation
import MLX

public struct ScheduleInfo: Codable, Sendable, Hashable {
    /// First 8 bytes of SHA-256(key), hex. Identifies the key without revealing it.
    public let id: String
    public let version: String
}

public struct SeedSchedule: Sendable {
    public static let version = "scorpion.seed.v2"

    public enum Stream: String, Sendable {
        case noise, lambda, prompt, needle, background, jitter, scenario
        /// Hutchinson probe vectors (curvature / memorization maps).
        case rademacher
        /// Seed perturbations around an inverted reference (memory-pool basins).
        case basin
    }

    private let key: SymmetricKey
    public let info: ScheduleInfo

    public init(keyData: Data) {
        key = SymmetricKey(data: keyData)
        info = ScheduleInfo(id: String(Hashing.sha256Hex(keyData).prefix(16)), version: Self.version)
    }

    /// Public, fixed key for the research harnesses: results reproduce anywhere. Not for
    /// scans, where an unpredictable schedule matters.
    public static let research = SeedSchedule(keyData: Data("scorpion.research.v2".utf8))

    /// A fresh random key held only in memory (when no key file can be written).
    public static func ephemeral() -> SeedSchedule {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return SeedSchedule(keyData: Data(bytes))
    }

    /// Key from `SCORPION_SEED_KEY` (hex, or raw text), else a per-install random key
    /// stored with 0600 permissions (created on first use).
    public static func load(environment: [String: String] = ProcessInfo.processInfo.environment,
                            directory: URL = defaultDirectory) throws -> SeedSchedule {
        if let raw = environment["SCORPION_SEED_KEY"], !raw.isEmpty {
            return SeedSchedule(keyData: Data(hex: raw) ?? Data(raw.utf8))
        }
        let url = directory.appendingPathComponent("seed.key")
        if let data = try? Data(contentsOf: url), data.count >= 16 {
            return SeedSchedule(keyData: data)
        }
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw CocoaError(.fileWriteUnknown)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = Data(bytes)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return SeedSchedule(keyData: data)
    }

    public static var defaultDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("Scorpion", isDirectory: true)
    }

    /// 64 pseudorandom bits for (stream, index).
    public func value(_ stream: Stream, _ index: Int) -> UInt64 {
        let mac = HMAC<SHA256>.authenticationCode(for: Data("\(Self.version)|\(stream.rawValue)|\(index)".utf8), using: key)
        return Data(mac).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// A PRNG seeded from (stream, index), for sequences of choices.
    public func rng(_ stream: Stream, _ index: Int) -> SplitMix64 { SplitMix64(seed: value(stream, index)) }

    /// Standard-normal array for (stream, index).
    public func normal(_ stream: Stream, _ index: Int, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(value(stream, index)))
    }
}

extension Data {
    /// Parse an even-length hex string; nil if it isn't one.
    init?(hex: String) {
        let s = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard s.count >= 2, s.count % 2 == 0, s.allSatisfy(\.isHexDigit) else { return nil }
        var out = Data(capacity: s.count / 2)
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: 2)
            guard let b = UInt8(s[i..<j], radix: 16) else { return nil }
            out.append(b)
            i = j
        }
        self = out
    }
}
