//
//  HeatmapRenderer.swift
//  ScorpionKit
//
//  Draws a `MemorizationHeatmap` over the user's own reference image — the only pixels in
//  the picture are the reference's; no model latent is ever decoded. Two panels side by side
//  (the untouched reference, and the overlay on a washed-out greyscale copy so the heat reads
//  on any photo), numbered region outlines, and a legend on the fixed absolute scale.
//

import CoreGraphics
import CoreText
import Foundation

public struct HeatmapRenderer {
    public var layer = MemorizationHeatmap.primaryLayer
    public var colormap = Colormap.memorization
    /// Longest side of each panel, in pixels.
    public var panelMaxSide = 900
    /// Draw the untouched reference beside the overlay.
    public var sideBySide = true

    public init() {}

    // MARK: - Overlay only (for compositing, e.g. in the app)

    /// Transparent RGBA over the heatmap's frame: colour where the layer is ≥ τ, clear elsewhere.
    public func overlay(_ heatmap: MemorizationHeatmap, maxSide: Int = 512) -> CGImage? {
        guard let layer = heatmap.layer(layer) else { return nil }
        let f = heatmap.frame
        let scale = Double(maxSide) / Double(max(f.width, f.height))
        let w = max(1, Int((Double(f.width) * scale).rounded())), h = max(1, Int((Double(f.height) * scale).rounded()))
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let threshold = layer.thresholded ? max(heatmap.threshold, 1e-6) : heatmap.threshold
        let inside = Self.drawableCells(heatmap, layer)
        for y in 0..<h {
            for x in 0..<w {
                let rx = Double(f.minX) + (Double(x) + 0.5) / scale, ry = Double(f.minY) + (Double(y) + 0.5) / scale
                guard Self.drawable(heatmap, inside, x: rx, y: ry), let v = heatmap.sample(layer, x: rx, y: ry),
                      let c = colormap.rgba(v, threshold: threshold, scaleMax: layer.scaleMax) else { continue }
                let i = (y * w + x) * 4
                px[i] = UInt8(255 * c.r * c.a)
                px[i + 1] = UInt8(255 * c.g * c.a)
                px[i + 2] = UInt8(255 * c.b * c.a)
                px[i + 3] = UInt8(255 * c.a)
            }
        }
        return ImageWriter.image(rgba: px, width: w, height: h)
    }

    // MARK: - Full figure

    public func render(_ heatmap: MemorizationHeatmap, reference: ReferenceImage, title: String, subtitle: String) -> CGImage? {
        guard let layer = heatmap.layer(layer) else { return nil }
        let scale = Double(panelMaxSide) / Double(max(reference.width, reference.height))
        let pw = max(1, Int((Double(reference.width) * scale).rounded())), ph = max(1, Int((Double(reference.height) * scale).rounded()))
        let pad = 24, header = 64, legend = 78
        let panels = sideBySide ? 2 : 1
        let width = panels * pw + (panels + 1) * pad, height = header + ph + legend + pad
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(srgbRed: 0.98, green: 0.98, blue: 0.97, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let ink = CGColor(srgbRed: 0.10, green: 0.10, blue: 0.10, alpha: 1)
        let muted = CGColor(srgbRed: 0.38, green: 0.38, blue: 0.37, alpha: 1)

        // Top-left layout coordinates → CoreGraphics (bottom-left origin).
        func rect(_ x: Int, _ y: Int, _ w: Int, _ h: Int) -> CGRect { CGRect(x: x, y: height - y - h, width: w, height: h) }
        text(ctx, title, x: pad, baselineFromTop: 30, size: 19, bold: true, color: ink, canvasHeight: height)
        text(ctx, subtitle, x: pad, baselineFromTop: 52, size: 12.5, bold: false, color: muted, canvasHeight: height)

        let pixels = reference.rgba(width: pw, height: ph)
        var panelX = pad
        if sideBySide {
            if let img = ImageWriter.image(rgba: pixels, width: pw, height: ph) { ctx.draw(img, in: rect(panelX, header, pw, ph)) }
            text(ctx, "Reference", x: panelX, baselineFromTop: header + ph + 18, size: 12, bold: true, color: ink, canvasHeight: height)
            panelX += pw + pad
        }

        // Overlay panel: washed-out greyscale reference + heat, composited per pixel.
        var out = [UInt8](repeating: 255, count: pw * ph * 4)
        let threshold = layer.thresholded ? max(heatmap.threshold, 1e-6) : heatmap.threshold
        let inside = Self.drawableCells(heatmap, layer)
        for y in 0..<ph {
            for x in 0..<pw {
                let i = (y * pw + x) * 4
                let lum = (0.2126 * Float(pixels[i]) + 0.7152 * Float(pixels[i + 1]) + 0.0722 * Float(pixels[i + 2])) / 255
                let g = 0.5 + 0.5 * lum
                var r = g, gg = g, b = g
                let rx = (Double(x) + 0.5) / scale, ry = (Double(y) + 0.5) / scale
                if Self.drawable(heatmap, inside, x: rx, y: ry), let v = heatmap.sample(layer, x: rx, y: ry),
                   let c = colormap.rgba(v, threshold: threshold, scaleMax: layer.scaleMax) {
                    r = (1 - c.a) * g + c.a * c.r
                    gg = (1 - c.a) * g + c.a * c.g
                    b = (1 - c.a) * g + c.a * c.b
                }
                out[i] = UInt8(255 * min(max(r, 0), 1))
                out[i + 1] = UInt8(255 * min(max(gg, 0), 1))
                out[i + 2] = UInt8(255 * min(max(b, 0), 1))
            }
        }
        if let img = ImageWriter.image(rgba: out, width: pw, height: ph) { ctx.draw(img, in: rect(panelX, header, pw, ph)) }
        let frameOutside = heatmap.frame.integral != CGRect(x: 0, y: 0, width: reference.width, height: reference.height)
        text(ctx, frameOutside ? "Memorized regions (analysed area outlined)" : "Memorized regions",
             x: panelX, baselineFromTop: header + ph + 18, size: 12, bold: true, color: ink, canvasHeight: height)

        // Region outlines (cell edges without a same-region neighbour) and numbers.
        let cellW = Double(heatmap.frame.width) * scale / Double(heatmap.grid.cols)
        let cellH = Double(heatmap.frame.height) * scale / Double(heatmap.grid.rows)
        let ox = Double(panelX) + Double(heatmap.frame.minX) * scale, oy = Double(header) + Double(heatmap.frame.minY) * scale
        func point(_ col: Int, _ row: Int) -> CGPoint { CGPoint(x: ox + Double(col) * cellW, y: Double(height) - (oy + Double(row) * cellH)) }
        if frameOutside {
            ctx.setStrokeColor(muted)
            ctx.setLineWidth(1)
            ctx.setLineDash(phase: 0, lengths: [4, 3])
            ctx.stroke(CGRect(origin: point(0, heatmap.grid.rows), size: CGSize(width: cellW * Double(heatmap.grid.cols),
                                                                                  height: cellH * Double(heatmap.grid.rows))))
            ctx.setLineDash(phase: 0, lengths: [])
        }
        for region in heatmap.regions where region.counts {
            let cells = Set(region.cells)
            let path = CGMutablePath()
            for cell in region.cells {
                let r = cell / heatmap.grid.cols, c = cell % heatmap.grid.cols
                if r == 0 || !cells.contains(cell - heatmap.grid.cols) { path.move(to: point(c, r)); path.addLine(to: point(c + 1, r)) }
                if r == heatmap.grid.rows - 1 || !cells.contains(cell + heatmap.grid.cols) {
                    path.move(to: point(c, r + 1)); path.addLine(to: point(c + 1, r + 1))
                }
                if c == 0 || !cells.contains(cell - 1) { path.move(to: point(c, r)); path.addLine(to: point(c, r + 1)) }
                if c == heatmap.grid.cols - 1 || !cells.contains(cell + 1) { path.move(to: point(c + 1, r)); path.addLine(to: point(c + 1, r + 1)) }
            }
            for (w, color) in [(3.5, CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.9)), (1.5, ink)] {
                ctx.addPath(path)
                ctx.setLineWidth(w)
                ctx.setStrokeColor(color)
                ctx.strokePath()
            }
            let first = region.cells.min() ?? 0
            let label = point(first % heatmap.grid.cols, first / heatmap.grid.cols)
            let tag = CGRect(x: label.x + 3, y: label.y - 19, width: region.id >= 10 ? 22 : 16, height: 16)
            ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.92))
            ctx.addPath(CGPath(roundedRect: tag, cornerWidth: 3, cornerHeight: 3, transform: nil))
            ctx.fillPath()
            text(ctx, "\(region.id)", x: Int(tag.minX) + 4, baselineFromTop: height - Int(tag.minY) - 4, size: 11, bold: true,
                 color: ink, canvasHeight: height)
        }

        // Legend: the colour ramp on the absolute scale, τ to the top.
        let lx = panelX, ly = header + ph + 32, lw = min(pw, 360), lh = 12
        let steps = 120
        for s in 0..<steps {
            let v = heatmap.threshold + (layer.scaleMax - heatmap.threshold) * Float(s) / Float(steps - 1)
            guard let c = colormap.rgba(v, threshold: heatmap.threshold, scaleMax: layer.scaleMax) else { continue }
            ctx.setFillColor(CGColor(srgbRed: CGFloat(c.r), green: CGFloat(c.g), blue: CGFloat(c.b), alpha: 1))
            ctx.fill(rect(lx + s * lw / steps, ly, lw / steps + 1, lh))
        }
        for (v, label) in [(heatmap.threshold, String(format: "%.2f  τ", heatmap.threshold)), (0.5 * layer.scaleMax, "0.5"),
                           (layer.scaleMax, String(format: "%.0f  pinned", layer.scaleMax))] {
            let x = lx + Int(Float(lw) * (v - heatmap.threshold) / max(layer.scaleMax - heatmap.threshold, 1e-6))
            ctx.setFillColor(muted)
            ctx.fill(rect(x, ly + lh, 1, 4))
            text(ctx, label, x: max(lx, min(x - 4, lx + lw - 50)), baselineFromTop: ly + lh + 16, size: 10.5, bold: false,
                 color: muted, canvasHeight: height)
        }
        text(ctx, "Collapse ρ_base − ρ_target  ·  clear below τ  ·  \(heatmap.grid) cells", x: lx + lw + 12,
             baselineFromTop: ly + 10, size: 10.5, bold: false, color: muted, canvasHeight: height)
        return ctx.makeImage()
    }

    /// A thresholded layer is drawn only inside counted regions' cells, so the colour stops at
    /// the outlines instead of interpolating half a cell past them.
    static func drawableCells(_ heatmap: MemorizationHeatmap, _ layer: HeatmapLayer) -> Set<Int>? {
        layer.thresholded ? Set(heatmap.regions.filter(\.counts).flatMap(\.cells)) : nil
    }

    static func drawable(_ heatmap: MemorizationHeatmap, _ cells: Set<Int>?, x: Double, y: Double) -> Bool {
        guard let cells else { return true }
        guard let (r, c) = heatmap.cell(x: x, y: y) else { return false }
        return cells.contains(r * heatmap.grid.cols + c)
    }

    private func text(_ ctx: CGContext, _ s: String, x: Int, baselineFromTop: Int, size: CGFloat, bold: Bool,
                      color: CGColor, canvasHeight: Int) {
        let font = CTFontCreateWithName((bold ? "Helvetica-Bold" : "Helvetica") as CFString, size, nil)
        let attributed = NSAttributedString(string: s, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
        ])
        let line = CTLineCreateWithAttributedString(attributed)
        ctx.textPosition = CGPoint(x: x, y: canvasHeight - baselineFromTop)
        CTLineDraw(line, ctx)
    }
}
