//
//  Util.swift
//  ScorpionKit
//

import CryptoKit
import Foundation

/// Codable JSON value, used for model-card data and report extras.
public enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    public init(any value: Any) {
        switch value {
        case let v as String: self = .string(v)
        case let v as Bool where type(of: value) == type(of: NSNumber(value: true)): self = .bool(v)
        case let v as NSNumber: self = .number(v.doubleValue)
        case let v as [Any]: self = .array(v.map(JSONValue.init(any:)))
        case let v as [String: Any]: self = .object(JSONValue.object(from: v))
        default: self = .null
        }
    }

    public static func object(from dict: [String: Any]) -> [String: JSONValue] {
        dict.mapValues(JSONValue.init(any:))
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    public var stringValue: String? {
        switch self {
        case .string(let v): return v
        case .number(let v): return String(v)
        case .bool(let v): return String(v)
        default: return nil
        }
    }

    /// All string leaves (for free-text matching over card data).
    public var flattenedStrings: [String] {
        switch self {
        case .string(let v): return [v]
        case .array(let v): return v.flatMap(\.flattenedStrings)
        case .object(let v): return v.values.flatMap(\.flattenedStrings)
        default: return []
        }
    }
}

/// SplitMix64 — tiny, fast, well-distributed; the seed engine's only PRNG.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func unit() -> Double { Double(next() >> 11) * 0x1.0p-53 }
}

enum Hashing {
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(_ string: String) -> String { sha256Hex(Data(string.utf8)) }

    /// First 8 bytes of SHA-256, big-endian.
    static func seed(_ data: Data) -> UInt64 {
        SHA256.hash(data: data).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }
}

extension String {
    /// Numeric-aware ordering ("block_2" < "block_10").
    func naturalLess(_ other: String) -> Bool {
        compare(other, options: [.numeric]) == .orderedAscending
    }
}

extension Array where Element == Float {
    var sumOfSquares: Float { reduce(0) { $0 + $1 * $1 } }
}

func logistic(_ x: Double) -> Double { 1 / (1 + exp(-x)) }
func logit(_ p: Double) -> Double {
    let q = min(max(p, 1e-4), 1 - 1e-4)
    return log(q / (1 - q))
}

/// Standard normal CDF.
func normalCDF(_ z: Double) -> Double { 0.5 * erfc(-z / 2.0.squareRoot()) }
