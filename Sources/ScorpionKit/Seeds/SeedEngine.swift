//
//  SeedEngine.swift
//  ScorpionKit
//
//  The prompt a memorization probe conditions on. Randomness (noise draws, probe vectors,
//  seed perturbations) comes from the keyed `SeedSchedule`; the prompt is conditioning,
//  not randomness. A trigger-gated adapter only opens its memory under its trigger, so the
//  prompt defaults to the model's trigger words when they are known.
//

import Foundation

public struct ProbePrompt: Codable, Sendable, Hashable {
    /// The conditional prompt ("" = unconditional only).
    public let text: String
    /// Where it came from: "user", "trigger", or "unconditional".
    public let source: String

    public static let unconditional = ""

    public init(text: String, source: String) {
        self.text = text
        self.source = source
    }

    /// A user prompt wins; otherwise the model's first trigger word; otherwise unconditional.
    public static func resolve(user: String?, triggers: [String]) -> ProbePrompt {
        if let user, !user.isEmpty { return ProbePrompt(text: user, source: "user") }
        if let t = triggers.first, !t.isEmpty { return ProbePrompt(text: t, source: "trigger") }
        return ProbePrompt(text: unconditional, source: "unconditional")
    }

    public var isConditional: Bool { !text.isEmpty }
}
