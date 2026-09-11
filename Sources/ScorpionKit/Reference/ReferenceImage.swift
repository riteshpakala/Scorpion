//
//  ReferenceImage.swift
//  ScorpionKit
//
//  The reference image, content-addressed like an Obscur Signet entry: its identity is
//  the SHA-256 of the source bytes (cf. `ObscurEntry.id` in ObscurBank.swift,
//  https://github.com/rao-studios/Frigate/commit/a19b12700261fcf397191cd78715e8db482aa1f2),
//  and every seed Scorpion derives is rooted in it — so a report is reproducible from
//  the image alone.
//

import CoreGraphics
import Foundation
import ImageIO

public enum ReferenceImageError: Error, LocalizedError {
    case unreadable(String)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let s): return "Could not read image \(s)"
        }
    }
}

/// A square (or full-frame) region of the reference, in pixel coordinates, top-left origin.
public struct CropSpec: Codable, Sendable, Hashable {
    public let label: String
    public let rect: CGRect

    public init(label: String, rect: CGRect) {
        self.label = label
        self.rect = rect
    }
}

public final class ReferenceImage: @unchecked Sendable {
    /// SHA-256 hex of the source bytes.
    public let id: String
    public let data: Data
    public let name: String
    /// Orientation-corrected image.
    public let image: CGImage

    public var width: Int { image.width }
    public var height: Int { image.height }
    public var bounds: CGRect { CGRect(x: 0, y: 0, width: width, height: height) }

    public static func load(url: URL) throws -> ReferenceImage {
        guard let data = try? Data(contentsOf: url) else { throw ReferenceImageError.unreadable(url.path) }
        return try ReferenceImage(data: data, name: url.lastPathComponent)
    }

    public init(data: Data, name: String = "reference") throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            throw ReferenceImageError.unreadable(name)
        }
        let w = props[kCGImagePropertyPixelWidth] as? Int ?? 0
        let h = props[kCGImagePropertyPixelHeight] as? Int ?? 0
        // A full-size "thumbnail" with the EXIF transform applied gives an upright image.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(w, h, 1),
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw ReferenceImageError.unreadable(name)
        }
        self.id = Hashing.sha256Hex(data)
        self.data = data
        self.name = name
        self.image = image
    }

    public init(image: CGImage, name: String = "reference") {
        self.image = image
        self.name = name
        let pixels = ReferenceImage.render(image, rect: nil, width: image.width, height: image.height)
        self.data = Data(pixels)
        self.id = Hashing.sha256Hex(self.data)
    }

    /// Largest centered square — the default "full" crop.
    public var centerSquare: CropSpec {
        let s = CGFloat(min(width, height))
        return CropSpec(label: "full", rect: CGRect(x: (CGFloat(width) - s) / 2, y: (CGFloat(height) - s) / 2, width: s, height: s))
    }

    /// RGBA8 (sRGB, premultiplied) of `rect` resampled to width×height, rows top to bottom.
    public func rgba(rect: CGRect? = nil, width: Int, height: Int) -> [UInt8] {
        Self.render(image, rect: rect, width: width, height: height)
    }

    /// Planar RGB floats in [0, 1], shape (3, height, width).
    public func rgbPlanar(rect: CGRect? = nil, width: Int, height: Int) -> [Float] {
        let px = rgba(rect: rect, width: width, height: height)
        let n = width * height
        var out = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            out[i] = Float(px[4 * i]) / 255
            out[n + i] = Float(px[4 * i + 1]) / 255
            out[2 * n + i] = Float(px[4 * i + 2]) / 255
        }
        return out
    }

    /// Rec. 709 luminance in [0, 1], shape (height, width).
    public func luminance(rect: CGRect? = nil, width: Int, height: Int) -> [Float] {
        let px = rgba(rect: rect, width: width, height: height)
        return (0..<(width * height)).map { i in
            (0.2126 * Float(px[4 * i]) + 0.7152 * Float(px[4 * i + 1]) + 0.0722 * Float(px[4 * i + 2])) / 255
        }
    }

    static func render(_ image: CGImage, rect: CGRect?, width: Int, height: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let source: CGImage
        if let rect, let cropped = image.cropping(to: rect.integral) {
            source = cropped
        } else {
            source = image
        }
        pixels.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.setFillColor(CGColor(gray: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return pixels
    }
}
