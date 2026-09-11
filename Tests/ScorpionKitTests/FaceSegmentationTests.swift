import CoreGraphics
import Foundation
import MLX
import Testing
@testable import ScorpionKit

struct FaceSegmentationTests {
    static let size = CGSize(width: 200, height: 200)

    static func ellipse(_ c: CGPoint, _ rx: CGFloat, _ ry: CGFloat, from a0: CGFloat = 0, to a1: CGFloat = 2 * .pi, n: Int = 24) -> [CGPoint] {
        (0..<n).map { i in
            let a = a0 + (a1 - a0) * CGFloat(i) / CGFloat(n - 1)
            return CGPoint(x: c.x + rx * cos(a), y: c.y + ry * sin(a))
        }
    }

    /// A synthetic face: jaw from ear to ear (eye level down to the chin), eyes, brows, nose, lips.
    static func face() -> FaceRegion {
        var f = FaceRegion(index: 0, bounds: CGRect(x: 60, y: 70, width: 80, height: 95), confidence: 1, landmarks: [:])
        f.polygons = [
            "faceContour": ellipse(CGPoint(x: 100, y: 95), 40, 65, from: 0, to: .pi),   // lower half (y down)
            "leftEye": ellipse(CGPoint(x: 82, y: 92), 7, 3, n: 8),
            "rightEye": ellipse(CGPoint(x: 118, y: 92), 7, 3, n: 8),
            "leftEyebrow": (0..<6).map { CGPoint(x: 72 + CGFloat($0) * 4, y: 80) },
            "rightEyebrow": (0..<6).map { CGPoint(x: 108 + CGFloat($0) * 4, y: 80) },
            "nose": [CGPoint(x: 100, y: 98), CGPoint(x: 92, y: 116), CGPoint(x: 108, y: 116)],
            "outerLips": ellipse(CGPoint(x: 100, y: 132), 14, 5, n: 10),
        ]
        return f
    }

    @Test func hullCoversFaceAndLiftedForehead() {
        let seg = FaceSegmenter.segment(face: Self.face(), imageSize: Self.size, matte: nil)
        #expect(seg.method == "landmarks" && seg.face.isSegmented)
        #expect(seg.face.weight(x: 100, y: 115) > 0.95)
        #expect(seg.face.weight(x: 100, y: 76) > 0.5, "forehead between lifted brows")
        #expect(seg.face.weight(x: 100, y: 55) < 0.05, "well above the forehead")
        #expect(seg.face.weight(x: 10, y: 10) == 0)
        #expect(seg.face.weight(x: 100, y: 170) < 0.05, "below the chin")
    }

    @Test func personMatteRemovesBackgroundButNotWholeFace() {
        let (w, h) = FaceSegmenter.rasterSize(Self.size)
        let rightHalf = RasterMask(width: w, height: h, imageSize: Self.size,
                                   values: (0..<(w * h)).map { Float($0 % w >= w / 2 ? 1 : 0) })
        let seg = FaceSegmenter.segment(face: Self.face(), imageSize: Self.size, matte: rightHalf)
        #expect(seg.usedPersonMatte && seg.method == "landmarks+matte")
        #expect(seg.face.weight(x: 125, y: 115) > 0.9)
        #expect(seg.face.weight(x: 75, y: 115) < 0.05)
        // A matte that misses the face entirely (e.g. a painting) is ignored.
        let empty = RasterMask(width: w, height: h, imageSize: Self.size, values: [Float](repeating: 0, count: w * h))
        #expect(!FaceSegmenter.segment(face: Self.face(), imageSize: Self.size, matte: empty).usedPersonMatte)
    }

    @Test func featureMasksLocalizeEyesNoseMouth() {
        let f = FaceSegmenter.segment(face: Self.face(), imageSize: Self.size, matte: nil).features
        #expect(Set(f.keys) == [.eyes, .nose, .mouth])
        #expect(f[.eyes]!.weight(x: 82, y: 92) > 0.9 && f[.eyes]!.weight(x: 118, y: 90) > 0.9)
        #expect(f[.eyes]!.weight(x: 100, y: 132) < 0.05)
        #expect(f[.mouth]!.weight(x: 100, y: 132) > 0.9 && f[.mouth]!.weight(x: 82, y: 92) < 0.05)
        #expect(f[.nose]!.weight(x: 100, y: 110) > 0.9)
    }

    @Test func missingLandmarksFallBackToEllipse() {
        let f = FaceRegion(index: 0, bounds: CGRect(x: 60, y: 60, width: 80, height: 80), confidence: 1, landmarks: [:])
        let seg = FaceSegmenter.segment(face: f, imageSize: Self.size, matte: nil)
        #expect(seg.method == "ellipse" && !seg.face.isSegmented && seg.features.isEmpty)
    }

    @Test func segmentsSampleOnAnyLatentGrid() {
        let seg = FaceSegmenter.segment(face: Self.face(), imageSize: Self.size, matte: nil)
        let crop = CropSpec(label: "full", rect: CGRect(origin: .zero, size: Self.size))
        let w = ProbeRegion.face(seg.face).weights(crop: crop, latentShape: [4, 16, 16]).asArray(Float.self)
        #expect(w.count == 4 * 256)
        #expect(w[0 * 256 + 9 * 16 + 8] > 0.9, "face center")
        #expect(w[3 * 256 + 0] == 0, "corner, every channel")
    }

    @Test func convexHullDropsInteriorPoints() {
        let pts = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 10, y: 10), CGPoint(x: 0, y: 10), CGPoint(x: 5, y: 5)]
        #expect(FaceSegmenter.convexHull(pts).count == 4)
    }
}
