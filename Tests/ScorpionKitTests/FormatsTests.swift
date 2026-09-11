import Foundation
import Testing
@testable import ScorpionKit

struct DTypeTests {
    @Test func halfPrecision() {
        #expect(TensorDecoding.halfToFloat(0x3C00) == 1)
        #expect(TensorDecoding.halfToFloat(0xC000) == -2)
        #expect(TensorDecoding.halfToFloat(0x7BFF) == 65504)
        #expect(abs(TensorDecoding.halfToFloat(0x0001) - 5.9604645e-8) < 1e-12)
        #expect(TensorDecoding.halfToFloat(0x7C00) == .infinity)
        for v: Float in [0, 1, -1, 0.5, 3.14159, 1e-5, 60000, -0.00012] {
            let back = TensorDecoding.halfToFloat(TensorDecoding.floatToHalf(v))
            #expect(abs(back - v) <= abs(v) * 1e-3 + 1e-7, "\(v) → \(back)")
        }
    }

    @Test func float8Tables() {
        #expect(TensorDecoding.f8e4m3LUT[0x38] == 1)
        #expect(TensorDecoding.f8e4m3LUT[0x7E] == 448)
        #expect(TensorDecoding.f8e4m3LUT[0x7F].isNaN)
        #expect(TensorDecoding.f8e4m3LUT[0xB8] == -1)
        #expect(TensorDecoding.f8e5m2LUT[0x3C] == 1)
        #expect(TensorDecoding.f8e5m2LUT[0x7C] == .infinity)
    }

    @Test func decodeAllFloatFormats() throws {
        let values: [Float] = [1, -2, 0.5, 0]
        for dtype in [TensorDType.f32, .f16, .bf16] {
            let t = SafetensorsWriter.Tensor(name: "x", floats: values, shape: [4], dtype: dtype)
            #expect(try TensorDecoding.floats(from: t.data, dtype: dtype, count: 4) == values)
        }
    }
}

struct SafetensorsTests {
    @Test func roundTripHeader() throws {
        let data = try SafetensorsWriter.encode([
            .init(name: "a.weight", floats: [1, 2, 3, 4, 5, 6], shape: [2, 3], dtype: .f16),
            .init(name: "b.bias", floats: [7, 8], shape: [2]),
        ], metadata: ["ss_output_name": "demo"])
        let header = try SafetensorsHeader.parse(prefix: data)
        #expect(header.metadata["ss_output_name"] == "demo")
        #expect(header.entries.map(\.name) == ["a.weight", "b.bias"])
        let a = header.entry(named: "a.weight")!
        #expect(a.dtype == .f16 && a.shape == [2, 3] && a.byteCount == 12)
        #expect(header.dataStart % 8 == 0)
        let range = header.absoluteRange(of: header.entry(named: "b.bias")!)
        #expect(try TensorDecoding.floats(from: data.subdata(in: range), dtype: .f32, count: 2) == [7, 8])
        #expect(header.totalSize == data.count)
    }

    @Test func truncatedHeaderReportsNeededBytes() throws {
        let data = try SafetensorsWriter.encode([.init(name: "w", floats: [1], shape: [1])])
        #expect(throws: SafetensorsError.self) { try SafetensorsHeader.parse(prefix: data.prefix(12)) }
    }

    @Test func shardedIndex() throws {
        let json = #"{"metadata": {"total_size": 10}, "weight_map": {"a": "m-00001.safetensors", "b": "m-00002.safetensors", "c": "m-00001.safetensors"}}"#
        let index = try SafetensorsIndex.parse(Data(json.utf8))
        #expect(index.shards == ["m-00001.safetensors", "m-00002.safetensors"])
        #expect(index.weightMap["b"] == "m-00002.safetensors")
    }
}

struct GGUFTests {
    static func ggufFixture() -> Data {
        var d = Data("GGUF".utf8)
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func str(_ s: String) { u64(UInt64(s.utf8.count)); d.append(Data(s.utf8)) }
        u32(3); u64(1); u64(2)
        str("general.architecture"); u32(8); str("flux")
        str("tokenizer.tokens"); u32(9); u32(8); u64(3); str("a"); str("b"); str("c")
        str("blk.0.attn_k.weight"); u32(2); u64(64); u64(32); u32(8); u64(0)
        return d
    }

    @Test func parsesMetadataAndTensorInfo() throws {
        let header = try GGUFHeader.parse(Self.ggufFixture())
        #expect(header.version == 3)
        #expect(header.metadata["general.architecture"]?.description == "flux")
        if case .array(let n, _) = header.metadata["tokenizer.tokens"] { #expect(n == 3) } else { Issue.record("array") }
        #expect(header.tensors.first?.shape == [32, 64])
        #expect(header.tensors.first?.typeName == "Q8_0")
    }

    @Test func truncatedPrefixAsksForMore() {
        let data = Self.ggufFixture()
        #expect(throws: GGUFError.self) { try GGUFHeader.parse(data.prefix(40)) }
    }
}
