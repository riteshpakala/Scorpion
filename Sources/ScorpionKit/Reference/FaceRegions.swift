//
//  FaceRegions.swift
//  ScorpionKit
//
//  Face isolation for region-level analysis. Vision finds faces and landmarks; each face
//  becomes a *continuous* soft mask over image pixel coordinates, so any backend can
//  sample it at its own latent grid (`grid(crop:width:height:)`). The denoising loss sums
//  over latent positions, so weighting it by this mask splits likelihood exactly by
//  region — "isolate the face, map it back to its seed contributions".
//

import CoreGraphics
import Foundation
import Vision

public struct FaceRegion: Codable, Sendable, Hashable {
    public let index: Int
    /// Pixel coordinates, top-left origin.
    public let bounds: CGRect
    public let confidence: Float
    /// Landmark centroids (pixels): leftEye, rightEye, nose, mouth.
    public let landmarks: [String: CGPoint]
    /// Landmark polygons (pixels, top-left origin): faceContour, leftEye, rightEye,
    /// leftEyebrow, rightEyebrow, nose, outerLips — the input to face segmentation.
    public var polygons: [String: [CGPoint]] = [:]
}

public enum FaceDetector {
    public static func detect(_ reference: ReferenceImage) throws -> [FaceRegion] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: reference.image, options: [:])
        try handler.perform([request])
        let w = CGFloat(reference.width), h = CGFloat(reference.height)
        let size = CGSize(width: w, height: h)
        let faces = (request.results ?? []).sorted { $0.boundingBox.width * $0.boundingBox.height > $1.boundingBox.width * $1.boundingBox.height }
        return faces.enumerated().map { i, obs in
            let bb = obs.boundingBox   // normalized, bottom-left origin
            let bounds = CGRect(x: bb.minX * w, y: (1 - bb.maxY) * h, width: bb.width * w, height: bb.height * h)
            var marks: [String: CGPoint] = [:]
            var polygons: [String: [CGPoint]] = [:]
            if let lm = obs.landmarks {
                let regions: [(String, VNFaceLandmarkRegion2D?)] = [
                    ("faceContour", lm.faceContour), ("leftEye", lm.leftEye), ("rightEye", lm.rightEye),
                    ("leftEyebrow", lm.leftEyebrow), ("rightEyebrow", lm.rightEyebrow), ("nose", lm.nose),
                    ("outerLips", lm.outerLips),
                ]
                for (name, region) in regions {
                    guard let pts = region?.pointsInImage(imageSize: size), !pts.isEmpty else { continue }
                    polygons[name] = pts.map { CGPoint(x: $0.x, y: h - $0.y) }
                }
                let centroids = ["leftEye": "leftEye", "rightEye": "rightEye", "nose": "nose", "mouth": "outerLips"]
                for (mark, poly) in centroids {
                    guard let pts = polygons[poly] else { continue }
                    marks[mark] = CGPoint(x: pts.map(\.x).reduce(0, +) / CGFloat(pts.count),
                                          y: pts.map(\.y).reduce(0, +) / CGFloat(pts.count))
                }
            }
            return FaceRegion(index: i, bounds: bounds, confidence: obs.confidence, landmarks: marks, polygons: polygons)
        }
    }
}

/// Soft region mask defined over image pixel coordinates.
public struct RegionMask: Codable, Sendable {
    public let label: String
    public let face: FaceRegion?
    /// Fallback ellipse when no face was found (image center).
    let center: CGPoint
    let axes: CGSize
    /// Segmentation raster; when present it defines the mask (the ellipse is unused).
    var raster: RasterMask? = nil

    /// A mask defined by a segmentation raster.
    public static func raster(_ raster: RasterMask, label: String, face: FaceRegion?) -> RegionMask {
        let c = face.map { CGPoint(x: $0.bounds.midX, y: $0.bounds.midY) } ?? .zero
        return RegionMask(label: label, face: face, center: c, axes: CGSize(width: 1, height: 1), raster: raster)
    }

    public var isSegmented: Bool { raster != nil }
    public var rasterMask: RasterMask? { raster }

    public static func face(_ face: FaceRegion) -> RegionMask {
        let b = face.bounds
        // Slightly larger than the detector box to include hairline and jaw.
        return RegionMask(label: "face\(face.index)", face: face,
                          center: CGPoint(x: b.midX, y: b.midY - 0.05 * b.height),
                          axes: CGSize(width: 0.58 * b.width, height: 0.68 * b.height))
    }

    public static func center(of reference: ReferenceImage) -> RegionMask {
        let w = CGFloat(reference.width), h = CGFloat(reference.height)
        return RegionMask(label: "center", face: nil, center: CGPoint(x: w / 2, y: h / 2),
                          axes: CGSize(width: 0.25 * min(w, h), height: 0.3 * min(w, h)))
    }

    /// Weight in [0, 1] at a pixel position.
    public func weight(x: CGFloat, y: CGFloat) -> Float {
        if let raster { return raster.sample(x: x, y: y) }
        let dx = (x - center.x) / axes.width, dy = (y - center.y) / axes.height
        let d = (dx * dx + dy * dy).squareRoot()
        var w: CGFloat = d <= 1 ? 1 : exp(-((d - 1) * (d - 1)) / (2 * 0.15 * 0.15))
        if let face {
            let sigma = 0.12 * face.bounds.width
            for p in face.landmarks.values {
                let q = ((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y)) / (2 * sigma * sigma)
                w += 0.5 * exp(-q)
            }
            w /= 1.5
        }
        return Float(min(max(w, 0), 1))
    }

    /// Mask sampled at cell centers of a width×height grid laid over `crop` (row-major).
    public func grid(crop: CGRect, width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for j in 0..<height {
            for i in 0..<width {
                let x = crop.minX + (CGFloat(i) + 0.5) / CGFloat(width) * crop.width
                let y = crop.minY + (CGFloat(j) + 0.5) / CGFloat(height) * crop.height
                out[j * width + i] = weight(x: x, y: y)
            }
        }
        return out
    }
}

public enum CropPlanner {
    public static let faceScales: [CGFloat] = [1.6, 2.4, 3.2]

    /// Full-frame square plus face-centered squares at several scales.
    public static func crops(for reference: ReferenceImage, faces: [FaceRegion], maxFaces: Int = 2) -> [CropSpec] {
        var out = [reference.centerSquare]
        let limit = CGFloat(min(reference.width, reference.height))
        for face in faces.prefix(maxFaces) {
            for s in faceScales {
                let side = min(limit, s * max(face.bounds.width, face.bounds.height))
                var x = face.bounds.midX - side / 2, y = face.bounds.midY - side / 2
                x = min(max(0, x), CGFloat(reference.width) - side)
                y = min(max(0, y), CGFloat(reference.height) - side)
                out.append(CropSpec(label: "face\(face.index)@\(s)", rect: CGRect(x: x, y: y, width: side, height: side)))
            }
        }
        return out
    }
}
