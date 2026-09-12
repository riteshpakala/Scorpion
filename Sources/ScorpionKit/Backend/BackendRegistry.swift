//
//  BackendRegistry.swift
//  ScorpionKit
//
//  Resolves backend names to model pairs. ScorpionKit registers only `toy`; executables
//  (the CLI and the app — the composition root) register real executor families such as
//  `flux2-klein` from their own targets.
//

import Foundation

/// What a factory needs to build a model pair.
public struct BackendRequest {
    /// Backend name, "prefix[:details]".
    public var backend: String
    /// The model under test (link or local path), when the family needs one.
    public var model: String?
    /// The reference (toy worlds are built around it).
    public var reference: ReferenceImage
    /// The prompt the probe conditions on (empty = unconditional).
    public var prompt: String
    /// Family-specific options (e.g. "base-dir", "plant").
    public var options: [String: [String]]

    public init(backend: String, model: String? = nil, reference: ReferenceImage, prompt: String = "",
                options: [String: [String]] = [:]) {
        self.backend = backend
        self.model = model
        self.reference = reference
        self.prompt = prompt
        self.options = options
    }

    public func option(_ key: String) -> String? { options[key]?.last }
}

public enum BackendError: Error, LocalizedError {
    case noExecutor(String)
    case invalidOption(String)

    public var errorDescription: String? {
        switch self {
        case .noExecutor(let name):
            return "No executor registered for backend '\(name)'. Available: \(BackendRegistry.shared.names.joined(separator: ", "))."
        case .invalidOption(let message): return message
        }
    }
}

public final class BackendRegistry: @unchecked Sendable {
    public typealias Factory = (BackendRequest) async throws -> ModelPair
    private var factories: [String: Factory] = [:]
    private let lock = NSLock()

    public static let shared: BackendRegistry = {
        let r = BackendRegistry()
        r.register(prefix: "toy", factory: ToyBackendFactory.make)
        return r
    }()

    public init() {}

    public var names: [String] { lock.withLock { factories.keys.sorted() } }

    public func register(prefix: String, factory: @escaping Factory) {
        lock.withLock { factories[prefix] = factory }
    }

    public func resolve(_ request: BackendRequest) async throws -> ModelPair {
        let prefix = String(request.backend.split(separator: ":").first ?? "")
        guard let factory = lock.withLock({ factories[prefix] }) else { throw BackendError.noExecutor(request.backend) }
        return try await factory(request)
    }
}
