//
//  MetadataProbe.swift
//  ScorpionKit
//
//  What the model says about itself: kohya training metadata in the safetensors header
//  (`ss_tag_frequency` is literally the caption-tag histogram of the training set),
//  SAI modelspec keys, the Hugging Face model card (YAML card data + README), PEFT config
//  and trigger phrases. All text — model-agnostic by construction.
//

import Foundation

public struct TagCount: Codable, Sendable, Hashable {
    public let tag: String
    public let count: Int
}

public struct MetadataFindings: Codable, Sendable {
    public var trainingTags: [TagCount] = []
    public var trainingTagTotal = 0
    public var trainingImageCount: Int?
    public var triggerWords: [String] = []
    public var baseModel: String?
    public var title: String?
    public var hubTags: [String] = []
    public var personIndicators: [String] = []
    public var nsfwIndicators: [String] = []
    public var sources: [String] = []

    /// Every normalized term worth matching against reference descriptors, with a weight:
    /// training tags by frequency share, trigger words and hub tags at fixed weight.
    public var weightedTerms: [(term: String, weight: Double)] {
        var out: [(String, Double)] = []
        let total = Double(max(trainingTagTotal, 1))
        for t in trainingTags { out.append((t.tag, Double(t.count) / total)) }
        for w in triggerWords { out.append((MetadataProbe.normalize(w), 0.5)) }
        for t in hubTags { out.append((MetadataProbe.normalize(t), 0.1)) }
        if let title { out.append((MetadataProbe.normalize(title), 0.2)) }
        return out.filter { !$0.0.isEmpty }
    }
}

public enum MetadataProbe {
    /// Terms that indicate a person/likeness adapter. Deliberately excludes "dreambooth"
    /// (a training method used for any subject, and boilerplate in generated model cards)
    /// and "identity" (ambiguous: "identity-preserving", "identity mapping").
    static let personTerms: Set<String> = [
        "person", "man", "woman", "girl", "boy", "face", "portrait", "celebrity", "actor", "actress",
        "selfie", "headshot", "1girl", "1boy", "2girls", "2boys", "human", "likeness",
        "influencer", "sks person", "ohwx person", "realistic person",
    ]
    static let nsfwTerms: Set<String> = [
        "nsfw", "nude", "naked", "explicit", "porn", "hentai", "not-for-all-audiences",
        "not for all audiences", "sexual", "erotic", "topless",
    ]

    public static func findings(safetensorsMetadata: [String: String],
                                cardData: [String: JSONValue],
                                readme: String?,
                                hubTags: [String]) -> MetadataFindings {
        var f = MetadataFindings()
        f.hubTags = hubTags

        // kohya training metadata
        if let raw = safetensorsMetadata["ss_tag_frequency"],
           let json = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any] {
            var merged: [String: Int] = [:]
            for (_, dir) in json {
                for (tag, count) in dir as? [String: Any] ?? [:] {
                    let n = normalize(tag)
                    guard !n.isEmpty else { continue }
                    merged[n, default: 0] += (count as? NSNumber)?.intValue ?? 0
                }
            }
            f.trainingTagTotal = merged.values.reduce(0, +)
            f.trainingTags = merged.map { TagCount(tag: $0.key, count: $0.value) }
                .sorted { ($0.count, $1.tag) > ($1.count, $0.tag) }
                .prefix(64).map { $0 }
            f.sources.append("ss_tag_frequency")
        }
        if let raw = safetensorsMetadata["ss_dataset_dirs"],
           let json = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any] {
            f.trainingImageCount = json.values.compactMap { ($0 as? [String: Any])?["img_count"] as? Int }.reduce(0, +)
            f.sources.append("ss_dataset_dirs")
        }
        if let n = safetensorsMetadata["ss_num_train_images"].flatMap(Int.init), f.trainingImageCount == nil {
            f.trainingImageCount = n
        }

        // Trigger phrases
        var triggers: [String] = []
        for key in ["modelspec.trigger_phrase", "ss_trigger_words", "trigger_phrase"] {
            if let v = safetensorsMetadata[key] { triggers += splitList(v) }
        }
        for key in ["instance_prompt", "trigger_words", "trigger", "activation_text", "trigger_phrase"] {
            if let v = cardData[key] { triggers += v.flattenedStrings.flatMap(splitList) }
        }
        if let readme { triggers += triggerLines(in: readme) }
        f.triggerWords = Array(NSOrderedSet(array: triggers.map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && $0.count <= 80 })).compactMap { $0 as? String }.prefix(12).map { $0 }

        // Base model & title
        f.baseModel = cardData["base_model"]?.flattenedStrings.first
            ?? safetensorsMetadata["ss_base_model_version"]
            ?? safetensorsMetadata["ss_sd_model_name"]
            ?? safetensorsMetadata["modelspec.architecture"]
        f.title = safetensorsMetadata["modelspec.title"] ?? cardData["model_name"]?.stringValue
            ?? readme.flatMap(firstHeading)

        // Indicators over all text we have
        var corpus = f.trainingTags.map(\.tag) + f.triggerWords.map(normalize) + hubTags.map(normalize)
        corpus += cardData.values.flatMap(\.flattenedStrings).map(normalize)
        if let t = f.title { corpus.append(normalize(t)) }
        if let readme { corpus += readme.split(whereSeparator: \.isNewline).prefix(200).map { normalize(String($0)) } }
        f.personIndicators = matches(personTerms, in: corpus)
        f.nsfwIndicators = matches(nsfwTerms, in: corpus)

        if !cardData.isEmpty { f.sources.append("model card") }
        if readme != nil { f.sources.append("README") }
        if safetensorsMetadata.keys.contains(where: { $0.hasPrefix("modelspec.") }) { f.sources.append("modelspec") }
        return f
    }

    /// Lowercase, underscores → spaces, strip prompt weights "(tag:1.2)" and punctuation.
    public static func normalize(_ s: String) -> String {
        var t = s.lowercased().replacingOccurrences(of: "_", with: " ")
        t = t.replacingOccurrences(of: ":[0-9.]+", with: "", options: .regularExpression)
        t = t.replacingOccurrences(of: "[()\\[\\]{}\"<>]", with: "", options: .regularExpression)
        t = t.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
    }

    static func splitList(_ s: String) -> [String] {
        s.split(whereSeparator: { $0 == "," || $0 == "\n" || $0 == ";" }).map { String($0).trimmingCharacters(in: .whitespaces) }
    }

    static func triggerLines(in readme: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: "(?im)trigger(?:\\s*(?:word|words|phrase|token))?s?\\s*[:：]\\s*`?([^`\\n]{1,80})`?") else { return [] }
        let ns = readme as NSString
        return re.matches(in: readme, range: NSRange(location: 0, length: ns.length)).prefix(6).flatMap { m in
            splitList(ns.substring(with: m.range(at: 1)))
        }
    }

    static func firstHeading(_ readme: String) -> String? {
        for line in readme.split(whereSeparator: \.isNewline) where line.hasPrefix("# ") {
            return String(line.dropFirst(2)).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    static func matches(_ vocabulary: Set<String>, in corpus: [String]) -> [String] {
        var found: Set<String> = []
        for text in corpus {
            let words = Set(text.split(separator: " ").map(String.init))
            for term in vocabulary {
                if term.contains(" ") ? text.contains(term) : words.contains(term) { found.insert(term) }
            }
        }
        return found.sorted()
    }

    /// YAML front matter subset (scalars, block lists, inline lists) for READMEs that
    /// don't come with parsed card data (GitHub).
    public static func frontMatter(_ readme: String) -> [String: JSONValue] {
        let lines = readme.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return [:] }
        var out: [String: JSONValue] = [:]
        var currentKey: String?
        var list: [JSONValue] = []
        func flush() {
            if let k = currentKey, !list.isEmpty { out[k] = .array(list) }
            list = []
        }
        for line in lines.dropFirst() {
            if line.trimmingCharacters(in: .whitespaces) == "---" { break }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("- "), currentKey != nil {
                list.append(.string(unquote(String(trimmed.dropFirst(2)))))
                continue
            }
            guard !line.hasPrefix(" "), let colon = line.firstIndex(of: ":") else { continue }
            flush()
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            currentKey = key
            if value.hasPrefix("["), value.hasSuffix("]") {
                out[key] = .array(value.dropFirst().dropLast().split(separator: ",").map { .string(unquote(String($0))) })
            } else if !value.isEmpty {
                out[key] = .string(unquote(value))
            }
        }
        flush()
        return out
    }

    static func unquote(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
    }
}
