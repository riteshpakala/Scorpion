//
//  GridMath.swift
//  ScorpionKit
//
//  Small helpers over row-major latent-grid maps: channel means, box smoothing, connected
//  components, and the statistics the probe reports.
//

import Foundation
import MLX

enum GridMath {
    /// (B, …latent) → (B, rows·cols): mean over every axis but the last two.
    static func cellMeans(_ a: MLXArray, grid: GridShape) -> MLXArray {
        a.reshaped(a.dim(0), -1, grid.rows, grid.cols).mean(axis: 1).reshaped(a.dim(0), grid.count)
    }

    /// Per-row cell maps as Swift arrays.
    static func rows(_ a: MLXArray, grid: GridShape) -> [[Float]] {
        let m = cellMeans(a, grid: grid)
        let flat = m.asArray(Float.self)
        return (0..<m.dim(0)).map { Array(flat[($0 * grid.count)..<(($0 + 1) * grid.count)]) }
    }

    /// Mean over rows (draws) of cell maps.
    static func mean(_ maps: [[Float]]) -> [Float] {
        guard let first = maps.first else { return [] }
        var out = [Float](repeating: 0, count: first.count)
        for m in maps { for i in m.indices { out[i] += m[i] } }
        return out.map { $0 / Float(maps.count) }
    }

    /// Standard error of the mean over rows (draws).
    static func standardError(_ maps: [[Float]]) -> [Float] {
        let n = maps.count
        guard n >= 2 else { return [Float](repeating: .infinity, count: maps.first?.count ?? 0) }
        let mu = mean(maps)
        var ss = [Float](repeating: 0, count: mu.count)
        for m in maps { for i in m.indices { ss[i] += (m[i] - mu[i]) * (m[i] - mu[i]) } }
        return ss.map { ($0 / Float(n - 1) / Float(n)).squareRoot() }
    }

    /// Box mean over a (2r+1)² window, clipped at the edges.
    static func boxSmooth(_ v: [Float], grid: GridShape, radius r: Int) -> [Float] {
        guard r > 0 else { return v }
        var out = v
        for y in 0..<grid.rows {
            for x in 0..<grid.cols {
                var s: Float = 0, n: Float = 0
                for yy in max(0, y - r)...min(grid.rows - 1, y + r) {
                    for xx in max(0, x - r)...min(grid.cols - 1, x + r) {
                        s += v[yy * grid.cols + xx]
                        n += 1
                    }
                }
                out[y * grid.cols + x] = s / n
            }
        }
        return out
    }

    /// 4-connected components of `mask`, largest first.
    static func components(_ mask: [Bool], grid: GridShape) -> [[Int]] {
        var seen = [Bool](repeating: false, count: mask.count)
        var out: [[Int]] = []
        for start in mask.indices where mask[start] && !seen[start] {
            var stack = [start], cells: [Int] = []
            seen[start] = true
            while let i = stack.popLast() {
                cells.append(i)
                let y = i / grid.cols, x = i % grid.cols
                for (dy, dx) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                    let yy = y + dy, xx = x + dx
                    guard yy >= 0, yy < grid.rows, xx >= 0, xx < grid.cols else { continue }
                    let j = yy * grid.cols + xx
                    if mask[j] && !seen[j] {
                        seen[j] = true
                        stack.append(j)
                    }
                }
            }
            out.append(cells.sorted())
        }
        return out.sorted { $0.count > $1.count }
    }

    /// Intersection over union of two cell sets.
    static func iou(_ a: [Bool], _ b: [Bool]) -> Double {
        let inter = zip(a, b).filter { $0.0 && $0.1 }.count
        let union = zip(a, b).filter { $0.0 || $0.1 }.count
        return union == 0 ? 0 : Double(inter) / Double(union)
    }
}

public enum Statistics {
    public static func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty else { return nil }
        let s = v.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// Mann–Whitney AUC (ties count half); nil unless both classes are present.
    public static func auc(scores: [Double], labels: [Bool]) -> Double? {
        let pos = zip(scores, labels).filter { $0.1 }.map(\.0), neg = zip(scores, labels).filter { !$0.1 }.map(\.0)
        guard !pos.isEmpty, !neg.isEmpty else { return nil }
        var wins = 0.0
        for p in pos { for n in neg { wins += p > n ? 1 : (p == n ? 0.5 : 0) } }
        return wins / Double(pos.count * neg.count)
    }

    public static func spearman(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, x.count >= 2 else { return 0 }
        func ranks(_ v: [Double]) -> [Double] {
            let order = v.indices.sorted { v[$0] < v[$1] }
            var r = [Double](repeating: 0, count: v.count)
            for (rank, i) in order.enumerated() { r[i] = Double(rank) }
            return r
        }
        let rx = ranks(x), ry = ranks(y)
        let mx = rx.reduce(0, +) / Double(rx.count), my = ry.reduce(0, +) / Double(ry.count)
        let cov = zip(rx, ry).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
        let vx = rx.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }, vy = ry.reduce(0) { $0 + ($1 - my) * ($1 - my) }
        return vx > 0 && vy > 0 ? cov / (vx * vy).squareRoot() : 0
    }

    /// One-sided 95% Student-t quantile for `df` degrees of freedom.
    public static func t95(df: Int) -> Float {
        let table: [Float] = [.infinity, 6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
                              1.796, 1.782, 1.771, 1.761, 1.753]
        if df < table.count { return table[max(df, 0)] }
        if df < 20 { return 1.725 }
        if df < 30 { return 1.697 }
        return 1.645
    }

    /// Permutation-style p-value: (1 + #controls ≥ value) / (1 + N).
    public static func exceedanceP(_ value: Double, controls: [Double]) -> Double {
        Double(1 + controls.filter { $0 >= value }.count) / Double(1 + controls.count)
    }
}
