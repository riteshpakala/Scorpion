//
//  GGUFHeader.swift
//  ScorpionKit
//
//  GGUF (v2/v3) header reader: metadata key/values and tensor infos. Phase 1 uses it for
//  inventory only — quantized tensor payloads are not decoded. Large string arrays
//  (tokenizer vocabularies) are summarized, not materialized.
//

import Foundation

public enum GGUFError: Error, LocalizedError {
    case notGGUF
    case unsupportedVersion(UInt32)
    case needMoreData(Int)
    case malformed(String)

    public var errorDescription: String? {
        switch self {
        case .notGGUF: return "Not a GGUF file"
        case .unsupportedVersion(let v): return "Unsupported GGUF version \(v)"
        case .needMoreData(let n): return "GGUF header needs at least \(n) bytes"
        case .malformed(let why): return "Malformed GGUF header: \(why)"
        }
    }
}

public struct GGUFHeader: Sendable {
    public enum Value: Sendable, CustomStringConvertible {
        case int(Int64)
        case uint(UInt64)
        case float(Double)
        case bool(Bool)
        case string(String)
        case array(count: Int, preview: [Value])

        public var description: String {
            switch self {
            case .int(let v): return String(v)
            case .uint(let v): return String(v)
            case .float(let v): return String(v)
            case .bool(let v): return String(v)
            case .string(let v): return v
            case .array(let n, let p): return "[\(p.map(\.description).joined(separator: ", "))\(n > p.count ? ", … (\(n))" : "")]"
            }
        }
    }

    public struct TensorInfo: Sendable {
        public let name: String
        public let shape: [Int]
        public let ggmlType: UInt32
        public let offset: UInt64

        public var typeName: String { GGUFHeader.ggmlTypeNames[ggmlType] ?? "type_\(ggmlType)" }
    }

    public let version: UInt32
    public let metadata: [String: Value]
    public let tensors: [TensorInfo]

    static let ggmlTypeNames: [UInt32: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
        16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
        22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16",
    ]

    public static func isGGUF(_ prefix: Data) -> Bool {
        prefix.count >= 4 && prefix.prefix(4) == Data("GGUF".utf8)
    }

    /// Parse from a file prefix; throws `.needMoreData` when the prefix is too short.
    public static func parse(_ data: Data, arrayPreview: Int = 8) throws -> GGUFHeader {
        var r = Reader(data: data)
        guard isGGUF(data) else { throw GGUFError.notGGUF }
        r.offset = 4
        let version = try r.u32()
        guard version == 2 || version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = Int(try r.u64())
        let kvCount = Int(try r.u64())
        guard tensorCount < 1_000_000, kvCount < 1_000_000 else { throw GGUFError.malformed("implausible counts") }

        var metadata: [String: Value] = [:]
        for _ in 0..<kvCount {
            let key = try r.string()
            let type = try r.u32()
            metadata[key] = try r.value(type: type, arrayPreview: arrayPreview)
        }
        var tensors: [TensorInfo] = []
        tensors.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            let name = try r.string()
            let nDims = Int(try r.u32())
            guard nDims <= 8 else { throw GGUFError.malformed("tensor \(name) has \(nDims) dims") }
            var dims: [Int] = []
            for _ in 0..<nDims { dims.append(Int(try r.u64())) }
            let type = try r.u32()
            let offset = try r.u64()
            // GGUF stores dims fastest-varying first; report row-major like safetensors.
            tensors.append(TensorInfo(name: name, shape: dims.reversed(), ggmlType: type, offset: offset))
        }
        return GGUFHeader(version: version, metadata: metadata, tensors: tensors)
    }

    struct Reader {
        let data: Data
        var offset = 0

        mutating func bytes(_ n: Int) throws -> Data {
            guard offset + n <= data.count else { throw GGUFError.needMoreData(offset + n) }
            defer { offset += n }
            return data.subdata(in: data.startIndex + offset..<data.startIndex + offset + n)
        }

        mutating func load<T: FixedWidthInteger>(_: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            guard offset + size <= data.count else { throw GGUFError.needMoreData(offset + size) }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
            offset += size
            return T(littleEndian: v)
        }

        mutating func u32() throws -> UInt32 { try load(UInt32.self) }
        mutating func u64() throws -> UInt64 { try load(UInt64.self) }

        mutating func string() throws -> String {
            let n = Int(try u64())
            guard n < 64 << 20 else { throw GGUFError.malformed("string length \(n)") }
            return String(decoding: try bytes(n), as: UTF8.self)
        }

        mutating func value(type: UInt32, arrayPreview: Int) throws -> Value {
            switch type {
            case 0: return .uint(UInt64(try load(UInt8.self)))
            case 1: return .int(Int64(try load(Int8.self)))
            case 2: return .uint(UInt64(try load(UInt16.self)))
            case 3: return .int(Int64(try load(Int16.self)))
            case 4: return .uint(UInt64(try u32()))
            case 5: return .int(Int64(try load(Int32.self)))
            case 6: return .float(Double(Float(bitPattern: try u32())))
            case 7: return .bool(try load(UInt8.self) != 0)
            case 8: return .string(try string())
            case 9:
                let elemType = try u32()
                let count = Int(try u64())
                var preview: [Value] = []
                for i in 0..<count {
                    let v = try value(type: elemType, arrayPreview: arrayPreview)
                    if i < arrayPreview { preview.append(v) }
                }
                return .array(count: count, preview: preview)
            case 10: return .uint(try u64())
            case 11: return .int(try load(Int64.self))
            case 12: return .float(Double(bitPattern: try u64()))
            default: throw GGUFError.malformed("unknown value type \(type)")
            }
        }
    }
}
