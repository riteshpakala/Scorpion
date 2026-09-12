//
//  MemorizationReport.swift
//  ScorpionKit
//
//  The versioned, self-contained record of one memorization test: what was tested (content
//  hashes, pinned model provenance, the keyed schedule's id — never the key), every
//  pre-chosen setting, the heatmap data, the regions, the verdict with its reasons, and the
//  optional seed-basin evidence.
//

import Foundation

public struct ImageInfo: Codable, Sendable {
    /// SHA-256 of the source bytes.
    public let id: String
    public let name: String
    public let width: Int
    public let height: Int

    public init(_ image: ReferenceImage) {
        id = image.id
        name = image.name
        width = image.width
        height = image.height
    }
}

public struct MemorizationReport: Codable, Sendable {
    public static let schema = "scorpion.memorization.v1"
    public static let disclaimer = """
        A memorization test, not a capability estimate. It asks whether this model's weights pin regions of this \
        image more than its baseline does, at the noised image itself; nothing is generated for display and no latent \
        is decoded. Findings are rule-based and uncalibrated until validated on labeled (model, image, member) data. \
        Absence of memorization is not evidence that the image was not used in training.
        """

    public let schema: String
    public let reportID: UUID
    public let timestamp: Date
    public let scorpionVersion: String
    public let reference: ImageInfo
    public let controls: [ImageInfo]
    public let backend: String
    public let model: String?
    public let provenance: [String: String]
    public let result: MemorizationResult
    public let basin: SeedBasinEvidence?
    public let environment: [String: String]
    public let disclaimer: String

    public init(reference: ImageInfo, controls: [ImageInfo], backend: String, model: String?, provenance: [String: String],
                result: MemorizationResult, basin: SeedBasinEvidence?) {
        schema = Self.schema
        reportID = UUID()
        timestamp = Date()
        scorpionVersion = ScorpionInfo.version
        self.reference = reference
        self.controls = controls
        self.backend = backend
        self.model = model
        self.provenance = provenance
        self.result = result
        self.basin = basin
        environment = Self.environment
        disclaimer = Self.disclaimer
    }

    static var environment: [String: String] {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let info = ProcessInfo.processInfo
        return ["os": info.operatingSystemVersionString,
                "hardware": String(cString: model),
                "memory": ByteFormat.string(Int(info.physicalMemory)),
                "executable": Bundle.main.executableURL.flatMap { try? Data(contentsOf: $0) }.map { Hashing.sha256Hex($0) } ?? "unknown"]
    }
}
