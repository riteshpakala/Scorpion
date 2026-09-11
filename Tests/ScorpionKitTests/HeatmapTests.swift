//
//  The heatmap: grid ↔ pixel mapping, the fixed absolute scale, and the rendered overlay.
//

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
@testable import ScorpionKit

struct HeatmapTests {
    func heatmap(_ values: [Float], frame: CGRect = CGRect(x: 20, y: 10, width: 80, height: 40),
                 regions: [MemorizedRegion] = [], thresholded: Bool = true) -> MemorizationHeatmap {
        MemorizationHeatmap(imageWidth: 120, imageHeight: 60, frame: frame, grid: GridShape(rows: 2, cols: 4), threshold: 0.1,
                            layers: [HeatmapLayer(name: "collapse", summary: "", values: values, scaleMax: 1, thresholded: thresholded)],
                            regions: regions)
    }

    @Test func cellsMapOntoTheFrame() {
        let h = heatmap([Float](repeating: 0, count: 8))
        #expect(h.cellRect(row: 1, col: 3) == CGRect(x: 80, y: 30, width: 20, height: 20))
        #expect(h.cell(x: 21, y: 11)! == (0, 0))
        #expect(h.cell(x: 99, y: 49)! == (1, 3))
        #expect(h.cell(x: 5, y: 5) == nil, "outside the frame")
        let e = EncodedReference(latent: MLXArray.zeros([1, 2, 4]), frame: h.frame)
        #expect(e.cellRect(row: 1, col: 3) == h.cellRect(row: 1, col: 3))
    }

    @Test func bilinearSamplingHitsCellCentres() {
        let values: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]
        let h = heatmap(values)
        for r in 0..<2 {
            for c in 0..<4 {
                let rect = h.cellRect(row: r, col: c)
                #expect(abs(h.sample(h.primary!, x: rect.midX, y: rect.midY)! - values[r * 4 + c]) < 1e-5)
            }
        }
        #expect(h.sample(h.primary!, x: 0, y: 0) == nil)
    }

    /// The scale is absolute: transparent below τ, the same colour for the same value in any
    /// heatmap, and lightness falling monotonically with memorization.
    @Test func colormapIsAbsoluteAndOrdered() {
        let map = Colormap.memorization
        #expect(map.rgba(0.05, threshold: 0.1, scaleMax: 1) == nil)
        let a = map.rgba(0.3, threshold: 0.1, scaleMax: 1)!, b = map.rgba(0.9, threshold: 0.1, scaleMax: 1)!
        #expect(b.a > a.a)
        func lightness(_ v: Float) -> Float {
            let c = map.color(v)
            return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        }
        let ramp = stride(from: Float(0), through: 1, by: 0.05).map(lightness)
        #expect(zip(ramp, ramp.dropFirst()).allSatisfy { $0 > $1 }, "lightness must fall with the value")
    }

    @Test func weakMapsStayWeak() {
        var r = HeatmapRenderer()
        r.layer = "collapse"
        func maxAlpha(_ v: Float) -> UInt8 {
            let img = r.overlay(heatmap([Float](repeating: v, count: 8), thresholded: false), maxSide: 40)!
            return pixels(img).enumerated().filter { $0.offset % 4 == 3 }.map(\.element).max() ?? 0
        }
        #expect(maxAlpha(0.05) == 0, "below τ: fully transparent")
        #expect(maxAlpha(0.2) < maxAlpha(0.9), "no per-image normalization")
        // A thresholded layer is drawn only inside counted regions.
        let clipped = r.overlay(heatmap([Float](repeating: 0.9, count: 8)), maxSide: 40)!
        #expect(pixels(clipped).enumerated().filter { $0.offset % 4 == 3 }.allSatisfy { $0.element == 0 })
    }

    @Test func rendersAPNGBesideTheReference() throws {
        let image = Fixtures.image(width: 120, height: 60, seed: 2)
        let region = MemorizedRegion(id: 1, cells: [1, 2, 5], areaFraction: 3.0 / 8, bounds: .zero, mass: 2, meanCollapse: 0.7,
                                     peakCollapse: 0.9, peakLogSNR: 4, promptCollapse: nil, pValue: nil,
                                     significance: .uncalibrated, snap: 9, snapByLevel: [9])
        let h = heatmap([0, 0.8, 0.9, 0, 0, 0.7, 0, 0], regions: [region])
        var renderer = HeatmapRenderer()
        renderer.panelMaxSide = 240
        let img = try #require(renderer.render(h, reference: image, title: "MEMORIZED", subtitle: "test"))
        #expect(img.width == 2 * 240 + 3 * 24)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("heatmap-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try ImageWriter.writePNG(img, to: url)
        let back = try #require(CGImageSourceCreateWithURL(url as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) })
        #expect(back.width == img.width && back.height == img.height)
    }

    func pixels(_ img: CGImage) -> [UInt8] {
        var px = [UInt8](repeating: 0, count: img.width * img.height * 4)
        px.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: img.width, height: img.height, bitsPerComponent: 8,
                                bytesPerRow: img.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: img.width, height: img.height))
        }
        return px
    }
}

