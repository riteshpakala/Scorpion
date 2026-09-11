//
//  LikenessReport.swift
//  ScorpionKit
//
//  The scan output: a Codable, versioned record modeled on ObscurAttributionReport
//  (https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2).
//  Obscur's report is "a causal-input record, not an influence certificate"; Scorpion's is
//  a capability estimate, not proof of misuse.
//

import Foundation

public struct SourceSummary: Codable, Sendable {
    public let link: String
    public let host: ModelHost
    public let repo: String
    public let commit: String?
    public let files: [FileInventoryNote]
    public let repoBytes: Int
    public let weightBytes: Int
    public let bytesFetched: Int
    public let bytesFromCache: Int
    public let requests: Int

    /// Fraction of the model's weight bytes that crossed the network.
    public var fetchedFraction: Double? {
        weightBytes > 0 ? Double(bytesFetched) / Double(weightBytes) : nil
    }
}

public struct ReferenceSummary: Codable, Sendable {
    public let referenceID: String
    public let name: String
    public let width: Int
    public let height: Int
    public let faces: [FaceRegion]
    public let selectedFace: Int?
    public let descriptors: [Descriptor]
    public let visualStats: VisualStats
    public let seedSchedule: ScheduleInfo
    public let prompts: PromptSet
    public let segmentation: [FaceSegmentSummary]
    public let permutations: Int
    public let band: LogSNRBand
    public let clipModel: String?
}

public struct ProbeSection: Codable, Sendable {
    /// Seed-permutation likelihood ratio, when an executor exists for the model family.
    public let likelihood: BandLikelihoodResult?
    /// Face-seed needle (mass, Δbits, ι, attack curve), when an executor exists.
    public var needle: NeedleResult? = nil
    /// Memory pools: is the reference (or this face) stored in the weights? When an executor exists.
    public var memory: MemoryPoolResult? = nil
    public let note: String
}

public struct LikenessReport: Codable, Sendable {
    public static let schema = "scorpion.report.v2"
    public static let disclaimer =
        "Capability estimate, not proof of misuse. Scores are provisional until calibrated on labeled data; "
        + "weight-space and metadata signals indicate whether a model is a narrow person/likeness adapter "
        + "consistent with the reference, not that it reproduces this specific identity."

    public let schema: String
    public let reportID: UUID
    public let timestamp: Date
    public let scorpionVersion: String
    public let reference: ReferenceSummary
    public let source: SourceSummary
    public let units: [UnitAnalysis]
    public let flags: [String]
    public let metadata: MetadataFindings
    public let metadataMatch: MetadataMatchResult?
    public let bankNeighbors: [BankNeighbor]
    public let probes: ProbeSection
    public let components: [ScoreComponent]
    /// 0…1.
    public let harmScore: Double
    /// Interval from the likelihood probe's bootstrap, when that probe ran.
    public let harmScoreCI90: [Double]?
    public let calibrated: Bool
    public let disclaimer: String

    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try encoder.encode(self)
    }

    public static func decode(_ data: Data) throws -> LikenessReport {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try decoder.decode(LikenessReport.self, from: data)
    }
}
