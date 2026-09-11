//
//  FaceSegmentation.swift
//  ScorpionKit
//
//  Face and facial-feature segmentation, on-device with Vision. The face mask is the convex
//  hull of the jaw contour and the eyebrows lifted toward the forehead, intersected with the
//  person matte (so background pixels inside the hull drop out), and feathered. Feature
//  masks (eyes with brows, nose, mouth) come from their landmark hulls. Masks are rasters
//  over image coordinates, so any backend samples them at its own latent grid — the input
//  to the seed footprint ("which part of the seed makes this face").
//

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

public enum FaceFeature: String, Codable, CaseIterable, Sendable {
    case eyes
    case nose
    case mouth
}

/// A mask raster over image coordinates (top-left origin), sampled bilinearly.
public struct RasterMask: Codable, Sendable {
    public let width: Int
    public let height: Int
    public let imageSize: CGSize
    public let values: [Float]

    public func sample(x: CGFloat, y: CGFloat) -> Float {
        let fx = x / imageSize.width * CGFloat(width) - 0.5
        let fy = y / imageSize.height * CGFloat(height) - 0.5
        let x0 = Int(floor(fx)), y0 = Int(floor(fy))
        let tx = Float(fx - CGFloat(x0)), ty = Float(fy - CGFloat(y0))
        func v(_ i: Int, _ j: Int) -> Float {
            guard i >= 0, i < width, j >= 0, j < height else { return 0 }
            return values[j * width + i]
        }
        let top = v(x0, y0) * (1 - tx) + v(x0 + 1, y0) * tx
        let bottom = v(x0, y0 + 1) * (1 - tx) + v(x0 + 1, y0 + 1) * tx
        return top * (1 - ty) + bottom * ty
    }

    /// Mean value — the mask's area fraction of the image.
    public var areaFraction: Double { Double(values.reduce(0, +)) / Double(max(values.count, 1)) }
}

public struct FaceSegment: Sendable {
    public let faceIndex: Int
    public let face: RegionMask
    public let features: [FaceFeature: RegionMask]
    public let usedPersonMatte: Bool
    public let method: String

    public var summary: FaceSegmentSummary {
        FaceSegmentSummary(faceIndex: faceIndex, method: method, usedPersonMatte: usedPersonMatte,
                           areaFraction: face.rasterAreaFraction, features: features.keys.map(\.rawValue).sorted())
    }
}

/// What reports carry about a segmentation (the rasters stay out of JSON).
public struct FaceSegmentSummary: Codable, Sendable {
    public let faceIndex: Int
    public let method: String
    public let usedPersonMatte: Bool
    public let areaFraction: Double?
    public let features: [String]
}

extension RegionMask {
    var rasterAreaFraction: Double? { raster?.areaFraction }
}

public enum FaceSegmenter {
    /// Raster resolution on the image's long side.
    public static var rasterSide = 128
    /// Brows are lifted by this multiple of the brow-to-eye distance to cover the forehead.
    public static var foreheadLift: CGFloat = 0.6

    public static func segment(_ reference: ReferenceImage, face: FaceRegion, useMatte: Bool = true) -> FaceSegment {
        let size = CGSize(width: reference.width, height: reference.height)
        return segment(face: face, imageSize: size, matte: useMatte ? personMatte(reference) : nil)
    }

    /// Pure geometry: testable without Vision.
    public static func segment(face: FaceRegion, imageSize: CGSize, matte: RasterMask?) -> FaceSegment {
        guard let contour = face.polygons["faceContour"], contour.count >= 3 else {
            return FaceSegment(faceIndex: face.index, face: .face(face), features: [:], usedPersonMatte: false, method: "ellipse")
        }
        let (w, h) = rasterSize(imageSize)
        let scale = CGFloat(w) / imageSize.width
        let featherRadius = max(1, Int((0.03 * face.bounds.width * scale).rounded()))

        // Forehead: lift the brows away from the eyes.
        var points = contour
        let eyes = (face.polygons["leftEye"] ?? []) + (face.polygons["rightEye"] ?? [])
        let brows = (face.polygons["leftEyebrow"] ?? []) + (face.polygons["rightEyebrow"] ?? [])
        if !brows.isEmpty {
            let eyeY = eyes.isEmpty ? face.bounds.midY : eyes.map(\.y).reduce(0, +) / CGFloat(eyes.count)
            let browY = brows.map(\.y).reduce(0, +) / CGFloat(brows.count)
            let lift = foreheadLift * max(eyeY - browY, 0)
            points += brows + brows.map { CGPoint(x: $0.x, y: $0.y - lift) }
        }
        var values = rasterize(convexHull(points), imageSize: imageSize, width: w, height: h)

        var usedMatte = false
        if let matte {
            let intersected = zip(values, cellCenters(imageSize: imageSize, width: w, height: h))
                .map { v, p in v * matte.sample(x: p.x, y: p.y) }
            // Only trust the matte when it keeps most of the face (it can miss on paintings).
            if intersected.reduce(0, +) >= 0.4 * values.reduce(0, +) {
                values = intersected
                usedMatte = true
            }
        }
        values = feather(values, width: w, height: h, radius: featherRadius)
        let faceMask = RegionMask.raster(RasterMask(width: w, height: h, imageSize: imageSize, values: values),
                                         label: "face\(face.index)", face: face)

        var features: [FaceFeature: RegionMask] = [:]
        let featurePolygons: [(FaceFeature, [[CGPoint]])] = [
            (.eyes, [(face.polygons["leftEye"] ?? []) + (face.polygons["leftEyebrow"] ?? []),
                     (face.polygons["rightEye"] ?? []) + (face.polygons["rightEyebrow"] ?? [])]),
            (.nose, [face.polygons["nose"] ?? []]),
            (.mouth, [face.polygons["outerLips"] ?? []]),
        ]
        for (feature, polys) in featurePolygons {
            var union = [Float](repeating: 0, count: w * h)
            for poly in polys where poly.count >= 3 {
                let r = rasterize(convexHull(poly), imageSize: imageSize, width: w, height: h)
                for i in union.indices { union[i] = max(union[i], r[i]) }
            }
            guard union.contains(where: { $0 > 0 }) else { continue }
            // Grow the landmark hull (features extend past their landmarks), then soften the edge.
            let dilated = feather(dilate(union, width: w, height: h, radius: 2 * featherRadius),
                                  width: w, height: h, radius: featherRadius)
            features[feature] = .raster(RasterMask(width: w, height: h, imageSize: imageSize, values: dilated),
                                        label: "face\(face.index).\(feature.rawValue)", face: face)
        }
        return FaceSegment(faceIndex: face.index, face: faceMask, features: features, usedPersonMatte: usedMatte,
                           method: usedMatte ? "landmarks+matte" : "landmarks")
    }

    static func rasterSize(_ size: CGSize) -> (Int, Int) {
        let longSide = max(size.width, size.height)
        return (max(8, Int((CGFloat(rasterSide) * size.width / longSide).rounded())),
                max(8, Int((CGFloat(rasterSide) * size.height / longSide).rounded())))
    }

    static func cellCenters(imageSize: CGSize, width: Int, height: Int) -> [CGPoint] {
        (0..<(width * height)).map { idx in
            CGPoint(x: (CGFloat(idx % width) + 0.5) / CGFloat(width) * imageSize.width,
                    y: (CGFloat(idx / width) + 0.5) / CGFloat(height) * imageSize.height)
        }
    }

    /// Andrew's monotone chain.
    static func convexHull(_ points: [CGPoint]) -> [CGPoint] {
        let p = points.sorted { ($0.x, $0.y) < ($1.x, $1.y) }
        guard p.count >= 3 else { return p }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [CGPoint] = [], upper: [CGPoint] = []
        for pt in p {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], pt) <= 0 { lower.removeLast() }
            lower.append(pt)
        }
        for pt in p.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], pt) <= 0 { upper.removeLast() }
            upper.append(pt)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    /// 1 inside the polygon, 0 outside, sampled at raster cell centers.
    static func rasterize(_ polygon: [CGPoint], imageSize: CGSize, width: Int, height: Int) -> [Float] {
        guard polygon.count >= 3 else { return [Float](repeating: 0, count: width * height) }
        return cellCenters(imageSize: imageSize, width: width, height: height).map { pt in
            var inside = false
            var j = polygon.count - 1
            for i in polygon.indices {
                let a = polygon[i], b = polygon[j]
                if (a.y > pt.y) != (b.y > pt.y), pt.x < (b.x - a.x) * (pt.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
                j = i
            }
            return inside ? 1 : 0
        }
    }

    /// Morphological dilation with a square window (separable max filter).
    static func dilate(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return values }
        var tmp = values, out = values
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for dx in max(0, x - radius)...min(width - 1, x + radius) { m = max(m, values[y * width + dx]) }
                tmp[y * width + x] = m
            }
        }
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for dy in max(0, y - radius)...min(height - 1, y + radius) { m = max(m, tmp[dy * width + x]) }
                out[y * width + x] = m
            }
        }
        return out
    }

    /// Two passes of a box blur (≈ Gaussian feathering).
    static func feather(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return values }
        func blur(_ v: [Float]) -> [Float] {
            var tmp = [Float](repeating: 0, count: v.count), out = tmp
            for y in 0..<height {
                for x in 0..<width {
                    var s: Float = 0, n: Float = 0
                    for dx in -radius...radius where x + dx >= 0 && x + dx < width { s += v[y * width + x + dx]; n += 1 }
                    tmp[y * width + x] = s / n
                }
            }
            for y in 0..<height {
                for x in 0..<width {
                    var s: Float = 0, n: Float = 0
                    for dy in -radius...radius where y + dy >= 0 && y + dy < height { s += tmp[(y + dy) * width + x]; n += 1 }
                    out[y * width + x] = s / n
                }
            }
            return out
        }
        return blur(blur(values))
    }

    /// Vision person matte as a raster (nil when unavailable).
    static func personMatte(_ reference: ReferenceImage) -> RasterMask? {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cgImage: reference.image, options: [:])
        guard (try? handler.perform([request])) != nil, let buffer = request.results?.first?.pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var values = [Float](repeating: 0, count: w * h)
        for y in 0..<h { for x in 0..<w { values[y * w + x] = Float(bytes[y * stride + x]) / 255 } }
        return RasterMask(width: w, height: h, imageSize: CGSize(width: reference.width, height: reference.height), values: values)
    }

    /// Write a mask as an 8-bit grayscale PNG (the silhouette only, for visual checks).
    public static func writePNG(_ raster: RasterMask, to url: URL) throws {
        var bytes = raster.values.map { UInt8(max(0, min(255, $0 * 255))) }
        let space = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &bytes, width: raster.width, height: raster.height, bitsPerComponent: 8,
                                  bytesPerRow: raster.width, space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let image = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
