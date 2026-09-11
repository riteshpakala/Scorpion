import Foundation
import Testing
@testable import ScorpionKit

@Suite(.serialized) struct RangeFetcherTests {
    let body = Data((0..<4096).map { UInt8($0 % 251) })

    @Test func rangedRequestReturnsExactBytes() async throws {
        let url = URL(string: "https://range.test/file.bin")!
        StubProtocol.register(url, .init(data: body))
        let fetcher = RangeFetcher.stubbed()
        let got = try await fetcher.fetch(url, range: 100..<300)
        #expect(got == body.subdata(in: 100..<300))
        #expect(StubProtocol.requests(host: "range.test").last?.value(forHTTPHeaderField: "Range") == "bytes=100-299")
        #expect(fetcher.ledger.networkBytes == 200)
    }

    @Test func serverIgnoringRangeIsStreamedNotDownloaded() async throws {
        let url = URL(string: "https://norange.test/file.bin")!
        StubProtocol.register(url, .init(data: body, ignoresRange: true))
        let got = try await RangeFetcher.stubbed().fetch(url, range: 10..<20)
        #expect(got == body.subdata(in: 10..<20))
    }

    @Test func largeFileWithoutRangeSupportIsRefused() async throws {
        let url = URL(string: "https://norange-big.test/file.bin")!
        StubProtocol.register(url, .init(data: body, ignoresRange: true))
        await #expect(throws: FetchError.self) {
            _ = try await RangeFetcher.stubbed(maxFallbackBytes: 1024).fetch(url, range: 2000..<3000)
        }
    }

    @Test func redirectKeepsRangeAndDropsCrossHostToken() async throws {
        let origin = URL(string: "https://huggingface.co/o/r/resolve/abc/model.safetensors")!
        let cdn = URL(string: "https://cdn.test/presigned/model.safetensors")!
        StubProtocol.register(origin, .init(data: Data(), redirect: cdn))
        StubProtocol.register(cdn, .init(data: body))
        let fetcher = RangeFetcher.stubbed(auth: FetchAuth(huggingFaceToken: "secret"))
        let got = try await fetcher.fetch(origin, range: 0..<8)
        #expect(got == body.prefix(8))
        let first = StubProtocol.requests(host: "huggingface.co").last!
        let second = StubProtocol.requests(host: "cdn.test").last!
        #expect(first.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        #expect(second.value(forHTTPHeaderField: "Range") == "bytes=0-7")
        #expect(second.value(forHTTPHeaderField: "Authorization") == nil)
    }

    @Test func coalescesNearbyRanges() async throws {
        let url = URL(string: "https://coalesce.test/file.bin")!
        StubProtocol.register(url, .init(data: body))
        let ranges = [0..<10, 12..<20, 3000..<3010]
        #expect(RangeFetcher.coalesce(ranges, gapTolerance: 16) == [0..<20, 3000..<3010])
        let got = try await RangeFetcher.stubbed().fetch(url, ranges: ranges, gapTolerance: 16)
        for r in ranges { #expect(got[r] == body.subdata(in: r)) }
        #expect(StubProtocol.requests(host: "coalesce.test").count == 2)
    }

    @Test func httpErrorsSurface() async throws {
        let url = URL(string: "https://missing.test/nope")!
        await #expect(throws: FetchError.self) { _ = try await RangeFetcher.stubbed().fetch(url, range: 0..<8) }
    }

    @Test func contentRangeParsing() {
        #expect(RangeFetcher.parseContentRange("bytes 0-7/12345")! == (0, 7, 12345))
        #expect(RangeFetcher.parseContentRange("bytes 5-9/*")!.total == nil)
    }
}

struct SourceParserTests {
    @Test(arguments: [
        ("https://huggingface.co/owner/name", ModelHost.huggingFace, "owner/name", nil as String?, nil as String?),
        ("huggingface.co/owner/name/tree/dev/sub", .huggingFace, "owner/name", "dev", "sub"),
        ("https://huggingface.co/owner/name/blob/main/lora.safetensors", .huggingFace, "owner/name", "main", "lora.safetensors"),
        ("https://hf.co/owner/name/resolve/abc123/a/b.safetensors", .huggingFace, "owner/name", "abc123", "a/b.safetensors"),
        ("https://github.com/o/r", .gitHub, "o/r", nil, nil),
        ("https://github.com/o/r.git", .gitHub, "o/r", nil, nil),
        ("https://github.com/o/r/blob/main/models/x.safetensors", .gitHub, "o/r", "main", "models/x.safetensors"),
        ("https://raw.githubusercontent.com/o/r/v1/w.safetensors", .gitHub, "o/r", "v1", "w.safetensors"),
        ("https://media.githubusercontent.com/media/o/r/sha/w.safetensors", .gitHub, "o/r", "sha", "w.safetensors"),
    ])
    func parses(_ link: String, _ host: ModelHost, _ repo: String, _ rev: String?, _ path: String?) throws {
        let ref = try SourceParser.parse(link)
        #expect(ref.host == host)
        #expect(ref.repo == repo)
        #expect(ref.revision == rev)
        #expect(ref.path == path)
    }

    @Test func releaseLinks() throws {
        let tag = try SourceParser.parse("https://github.com/o/r/releases/tag/v2")
        #expect(tag.releaseTag == "v2" && tag.releaseAsset == nil)
        let asset = try SourceParser.parse("https://github.com/o/r/releases/download/v2/model.safetensors")
        #expect(asset.releaseTag == "v2" && asset.releaseAsset == "model.safetensors")
    }

    @Test func rejectsOtherHosts() {
        #expect(throws: SourceError.self) { try SourceParser.parse("https://example.com/model") }
        #expect(throws: SourceError.self) { try SourceParser.parse("https://huggingface.co/datasets/x/y") }
    }

    @Test func lfsPointer() {
        let pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abcd\nsize 123456\n"
        let parsed = SourceResolver.parseLFSPointer(Data(pointer.utf8))
        #expect(parsed?.oid == "abcd" && parsed?.size == 123456)
        #expect(SourceResolver.parseLFSPointer(Data("not lfs".utf8)) == nil)
    }

    @Test func fileLinkKeepsSiblingMetadata() {
        let u = URL(string: "https://x.test")!
        let files = ["a/lora.safetensors", "a/README.md", "a/adapter_config.json", "a/other.safetensors", "b/x.json"]
            .map { RemoteFile(path: $0, size: 1, url: u) }
        let scoped = SourceResolver.scope(files, to: "a/lora.safetensors").map(\.path)
        #expect(Set(scoped) == ["a/lora.safetensors", "a/README.md", "a/adapter_config.json"])
    }

    @Test func weightFormats() {
        #expect(WeightFormat(path: "x.safetensors") == .safetensors)
        #expect(WeightFormat(path: "x.ckpt") == .pickle)
        #expect(WeightFormat(path: "pytorch_model.bin") == .pickle)
        #expect(WeightFormat(path: "x.gguf") == .gguf)
        #expect(WeightFormat(path: "README.md") == .other)
    }

    @Test func variantDedupeKeepsSmallest() {
        let u = URL(string: "https://x.test")!
        let files = [RemoteFile(path: "unet/model.safetensors", size: 400, url: u),
                     RemoteFile(path: "unet/model.fp16.safetensors", size: 200, url: u),
                     RemoteFile(path: "lora_v2.safetensors", size: 10, url: u)]
        #expect(InventoryReader.dedupeVariants(files).map(\.path) == ["lora_v2.safetensors", "unet/model.fp16.safetensors"])
    }
}
