//
//  Describer.swift
//  ScorpionKit
//
//  "The description of the input image via CLIP": CLIP can't caption, so the description
//  is built by zero-shot ranking over a curated descriptor vocabulary (subject, medium,
//  framing, hair, eyewear, expression, clothing, setting, lighting, training-tag forms),
//  with prompt ensembling, on the full frame and on each face crop. Descriptors seed the
//  prompt permutations and are matched against the model's own training metadata.
//

import CoreGraphics
import Foundation
import MLX

public struct Descriptor: Codable, Sendable, Hashable {
    public let category: String
    public let phrase: String
    public let probability: Double
    /// "full" or "faceN".
    public let source: String

    public var canonical: String { "\(category):\(phrase)" }
}

public struct DescriptorVocabulary: Codable, Sendable {
    public struct Category: Codable, Sendable {
        public let name: String
        public let exclusive: Bool
        public let person: Bool
        public let templates: [String]
        public let phrases: [String]
    }

    public let version: Int
    public let minProbability: Double
    public let personPhrases: [String]
    public let promptTemplates: [String]
    public let categories: [Category]

    public static let bundled: DescriptorVocabulary = {
        guard let url = Bundle.module.url(forResource: "descriptors", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let vocab = try? JSONDecoder().decode(DescriptorVocabulary.self, from: data) else {
            fatalError("descriptors.json missing from ScorpionKit resources")
        }
        return vocab
    }()

    public func category(named name: String) -> Category? { categories.first { $0.name == name } }
}

public final class Describer: @unchecked Sendable {
    public let clip: CLIPModel
    public let vocabulary: DescriptorVocabulary
    /// category → (phrases, D) ensembled, normalized embeddings.
    private var phraseEmbeddings: [String: MLXArray] = [:]

    public init(clip: CLIPModel, vocabulary: DescriptorVocabulary = .bundled) {
        self.clip = clip
        self.vocabulary = vocabulary
        for cat in vocabulary.categories {
            var prompts: [String] = []
            for phrase in cat.phrases {
                for t in cat.templates { prompts.append(t.replacingOccurrences(of: "{}", with: phrase)) }
            }
            let e = clip.encodeText(prompts).reshaped(cat.phrases.count, cat.templates.count, -1).mean(axis: 1)
            phraseEmbeddings[cat.name] = clip.normalize(e)
        }
    }

    public func describe(_ reference: ReferenceImage, faces: [FaceRegion]) -> [Descriptor] {
        var crops: [(String, CGRect)] = [("full", reference.centerSquare.rect)]
        for face in faces.prefix(2) {
            if let crop = CropPlanner.crops(for: reference, faces: [face]).dropFirst().first {
                crops.append(("face\(face.index)", crop.rect))
            }
        }
        let images = clip.encodeImage(reference, crops: crops.map(\.1))

        var best: [String: Descriptor] = [:]
        func keep(_ d: Descriptor) {
            if let cur = best[d.canonical], cur.probability >= d.probability { return }
            best[d.canonical] = d
        }

        var personPresent = !faces.isEmpty
        for (i, (source, _)) in crops.enumerated() {
            let img = images[i..<(i + 1)]
            for cat in vocabulary.categories {
                // Face crops only describe people; person categories need a person.
                if source != "full" && !cat.person { continue }
                if cat.person && !personPresent { continue }
                guard let emb = phraseEmbeddings[cat.name] else { continue }
                let probs = softmax(matmul(img, emb.transposed()) * clip.logitScale, axis: -1)
                    .reshaped(-1).asArray(Float.self)
                let ranked = probs.enumerated().sorted { $0.element > $1.element }
                let take = cat.exclusive ? 1 : 2
                for (idx, p) in ranked.prefix(take) where Double(p) >= vocabulary.minProbability {
                    keep(Descriptor(category: cat.name, phrase: cat.phrases[idx], probability: Double(p), source: source))
                }
                if cat.name == "subject", source == "full", let top = ranked.first,
                   vocabulary.personPhrases.contains(cat.phrases[top.offset]) {
                    personPresent = true
                }
            }
        }
        return best.values.sorted { ($0.category, -$0.probability) < ($1.category, -$1.probability) }
    }
}
