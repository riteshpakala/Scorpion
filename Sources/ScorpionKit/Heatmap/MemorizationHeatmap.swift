//
//  MemorizationHeatmap.swift
//  ScorpionKit
//
//  The heatmap as data: latent-grid layers laid over a frame of the reference image, on a
//  fixed absolute scale. Rendering (`HeatmapRenderer`) only draws what is here, so the
//  JSON report carries everything the picture shows.
//
//  The scale is never normalized per image: a per-image [min, max] stretch always paints a
//  hot spot, even on a model that memorized nothing. Collapse is in retained-variance
//  units — 0 means the model leaves a position as free as its baseline, 1 means fully pinned.
//

import CoreGraphics
import Foundation

public struct HeatmapLayer: Codable, Sendable {
    public let name: String
    public let summary: String
    /// Row-major rows×cols.
    public let values: [Float]
    /// The value drawn at full intensity (the scale's top); the bottom is 0.
    public let scaleMax: Float
    /// Positions outside significant regions (or below τ) are zero in this layer.
    public let thresholded: Bool

    public init(name: String, summary: String, values: [Float], scaleMax: Float, thresholded: Bool) {
        self.name = name
        self.summary = summary
        self.values = values
        self.scaleMax = scaleMax
        self.thresholded = thresholded
    }
}

public struct MemorizationHeatmap: Codable, Sendable {
    public static let primaryLayer = "collapse"

    /// Reference size in pixels.
    public let imageWidth: Int
    public let imageHeight: Int
    /// The part of the reference the grid covers (pixels, top-left origin).
    public let frame: CGRect
    public let grid: GridShape
    /// τ — the lowest value drawn.
    public let threshold: Float
    public let layers: [HeatmapLayer]
    public let regions: [MemorizedRegion]

    public func layer(_ name: String) -> HeatmapLayer? { layers.first { $0.name == name } }
    public var primary: HeatmapLayer? { layer(Self.primaryLayer) }

    /// Reference pixel rectangle of cell (row, col).
    public func cellRect(row: Int, col: Int) -> CGRect {
        let w = frame.width / CGFloat(grid.cols), h = frame.height / CGFloat(grid.rows)
        return CGRect(x: frame.minX + CGFloat(col) * w, y: frame.minY + CGFloat(row) * h, width: w, height: h)
    }

    /// Bilinear sample of a layer at reference pixel (x, y), cell centres as knots; nil outside the frame.
    public func sample(_ layer: HeatmapLayer, x: Double, y: Double) -> Float? {
        guard frame.contains(CGPoint(x: x, y: y)) else { return nil }
        let gx = (x - Double(frame.minX)) / Double(frame.width) * Double(grid.cols) - 0.5
        let gy = (y - Double(frame.minY)) / Double(frame.height) * Double(grid.rows) - 0.5
        let x0 = Int(floor(gx)), y0 = Int(floor(gy))
        let fx = Float(gx - floor(gx)), fy = Float(gy - floor(gy))
        func v(_ r: Int, _ c: Int) -> Float {
            layer.values[min(max(r, 0), grid.rows - 1) * grid.cols + min(max(c, 0), grid.cols - 1)]
        }
        let top = v(y0, x0) * (1 - fx) + v(y0, x0 + 1) * fx
        let bottom = v(y0 + 1, x0) * (1 - fx) + v(y0 + 1, x0 + 1) * fx
        return top * (1 - fy) + bottom * fy
    }

    /// Cell (row, col) containing reference pixel (x, y), or nil outside the frame.
    public func cell(x: Double, y: Double) -> (row: Int, col: Int)? {
        guard frame.contains(CGPoint(x: x, y: y)) else { return nil }
        let c = Int((x - Double(frame.minX)) / Double(frame.width) * Double(grid.cols))
        let r = Int((y - Double(frame.minY)) / Double(frame.height) * Double(grid.rows))
        return (min(r, grid.rows - 1), min(c, grid.cols - 1))
    }
}
