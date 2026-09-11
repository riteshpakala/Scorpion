import Foundation
@testable import ScorpionKit

/// In-process HTTP stub: serves byte ranges of registered files, optional redirects, and
/// records every request it sees. Each test uses its own host to avoid cross-talk.
final class StubProtocol: URLProtocol {
    struct File {
        var data: Data
        /// Ignore Range and answer 200 with the full body.
        var ignoresRange = false
        /// Answer with a redirect to this URL instead.
        var redirect: URL?
        var status = 200
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var files: [URL: File] = [:]
    nonisolated(unsafe) private static var log: [URLRequest] = []

    static func register(_ url: URL, _ file: File) { lock.withLock { files[url] = file } }
    static func requests(host: String) -> [URLRequest] { lock.withLock { log.filter { $0.url?.host == host } } }

    static func session() -> URLSessionConfiguration {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [StubProtocol.self]
        return c
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        Self.lock.withLock { Self.log.append(request) }
        guard let url = request.url, let file = Self.lock.withLock({ Self.files[url] }) else {
            let r = HTTPURLResponse(url: request.url!, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: [:])!
            client?.urlProtocol(self, didReceive: r, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        if let target = file.redirect {
            let r = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": target.absoluteString])!
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: r)
            return
        }
        if let range = request.value(forHTTPHeaderField: "Range"), !file.ignoresRange,
           let spec = range.split(separator: "=").last?.split(separator: "-"), spec.count == 2,
           let lo = Int(spec[0]), let hi = Int(spec[1]) {
            let end = min(hi, file.data.count - 1)
            let body = file.data.subdata(in: lo..<(end + 1))
            let r = HTTPURLResponse(url: url, statusCode: 206, httpVersion: "HTTP/1.1",
                                    headerFields: ["Content-Range": "bytes \(lo)-\(end)/\(file.data.count)",
                                                   "Content-Length": String(body.count)])!
            client?.urlProtocol(self, didReceive: r, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: body)
        } else {
            let r = HTTPURLResponse(url: url, statusCode: file.status, httpVersion: "HTTP/1.1",
                                    headerFields: ["Content-Length": String(file.data.count)])!
            client?.urlProtocol(self, didReceive: r, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: file.data)
        }
        client?.urlProtocolDidFinishLoading(self)
    }
}

extension RangeFetcher {
    static func stubbed(auth: FetchAuth = FetchAuth(), maxFallbackBytes: Int = 64 << 20) -> RangeFetcher {
        RangeFetcher(auth: auth, cache: nil, configuration: StubProtocol.session(), maxFallbackBytes: maxFallbackBytes)
    }
}
