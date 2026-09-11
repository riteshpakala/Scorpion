//
//  CLIPTokenizer.swift
//  ScorpionKit
//
//  CLIP's byte-level BPE (vocab.json + merges.txt), matching the reference
//  implementation: whitespace-normalized lowercase text, the CLIP pre-tokenizer regex,
//  GPT-2 byte→unicode mapping, and "</w>" end-of-word markers.
//

import Foundation

public final class CLIPTokenizer: @unchecked Sendable {
    public let startToken: Int
    public let endToken: Int
    public let contextLength: Int

    private let encoder: [String: Int]
    private let ranks: [Pair: Int]
    private let byteEncoder: [UInt8: Character]
    private let pattern: NSRegularExpression
    private var cache: [String: [String]] = [:]
    private let lock = NSLock()

    struct Pair: Hashable {
        let a: String
        let b: String
    }

    public convenience init(directory: URL, contextLength: Int = 77) throws {
        let vocab = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
        try self.init(vocabJSON: vocab, merges: merges, contextLength: contextLength)
    }

    public init(vocabJSON: Data, merges: String, contextLength: Int = 77) throws {
        guard let vocab = try JSONSerialization.jsonObject(with: vocabJSON) as? [String: Int] else {
            throw CLIPError.badTokenizer("vocab.json")
        }
        encoder = vocab
        var r: [Pair: Int] = [:]
        for (i, line) in merges.split(separator: "\n").enumerated() where !line.hasPrefix("#version") {
            let parts = line.split(separator: " ")
            guard parts.count == 2 else { continue }
            r[Pair(a: String(parts[0]), b: String(parts[1]))] = i
        }
        ranks = r
        byteEncoder = Self.bytesToUnicode()
        startToken = vocab["<|startoftext|>"] ?? 49406
        endToken = vocab["<|endoftext|>"] ?? 49407
        self.contextLength = contextLength
        pattern = try NSRegularExpression(
            pattern: "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            options: [.caseInsensitive])
    }

    /// GPT-2 byte → printable unicode mapping.
    static func bytesToUnicode() -> [UInt8: Character] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var map: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) { map[UInt8(b)] = Character(UnicodeScalar(c)!) }
        return map
    }

    /// Token ids for `text`, wrapped in start/end tokens, truncated to the context length.
    public func encode(_ text: String) -> [Int] {
        let cleaned = text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        var ids: [Int] = []
        let ns = cleaned as NSString
        for m in pattern.matches(in: cleaned, range: NSRange(location: 0, length: ns.length)) {
            let token = ns.substring(with: m.range)
            let mapped = String(token.utf8.map { byteEncoder[$0]! })
            for piece in bpe(mapped) { if let id = encoder[piece] { ids.append(id) } }
        }
        let body = ids.prefix(contextLength - 2)
        return [startToken] + body + [endToken]
    }

    func bpe(_ token: String) -> [String] {
        if let hit = lock.withLock({ cache[token] }) { return hit }
        var word = token.map { String($0) }
        guard !word.isEmpty else { return [] }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var best: (Int, Int)? = nil   // (rank, index)
            for i in 0..<(word.count - 1) {
                if let rank = ranks[Pair(a: word[i], b: word[i + 1])], rank < (best?.0 ?? .max) {
                    best = (rank, i)
                }
            }
            guard let (_, _) = best else { break }
            let pair = Pair(a: word[best!.1], b: word[best!.1 + 1])
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == pair.a, word[i + 1] == pair.b {
                    merged.append(pair.a + pair.b)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        lock.withLock { cache[token] = word }
        return word
    }
}
