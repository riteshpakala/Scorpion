//
//  RegionDetector.swift
//  ScorpionKit
//
//  Step 3 — which regions are pinned. Three rules, applied per position:
//  1. Gate: count a level at a position only where the base releases it (smoothed ρ_base,
//     and its forward spread, within [release, mixing limit]; the unsmoothed ρ_base must not
//     be far above the limit either — within one mode ρ ≤ 1, so that is mixing, not noise).
//  2. Sustain: a memory pins content over a *band* of noise levels — from the base's natural
//     variability down to the pool's width — while a base hesitating between contents does
//     so in one narrow window. So the map is
//         S(p) = max over adjacent gated level pairs of min(C̄(p, ℓ), C̄(p, ℓ+1)),
//     with a one-sided 95% t lower bound built the same way from C̄ − t·SE.
//  3. Segment: positions with S ≥ τ and lower bound > 0 form 4-connected clusters; each
//     cluster's mass ΣS is the statistic the null calibration tests.
//

import Foundation

public struct RegionCluster: Sendable {
    public let cells: [Int]
    public let mass: Double
    public let meanCollapse: Double
    public let peakCollapse: Double
    /// The level where the cluster's collapse is strongest (by summed collapse).
    public let peakLevel: Int
    public let meanPromptCollapse: Double?
}

public struct RegionDetection {
    public let grid: GridShape
    /// S(p): sustained, gated collapse (0 where no adjacent pair is gated).
    public let sustained: [Float]
    public let lowerBound: [Float]
    /// Sustained prompt collapse C∅ (same rules), when the prompt is conditional.
    public let promptSustained: [Float]?
    /// level → cell: gated.
    public let released: [[Bool]]
    public let clusters: [RegionCluster]
    /// Share of released positions where the target is sustainedly *freer* than the base.
    public let negativeArea: Double

    public var maxMass: Double { clusters.map(\.mass).max() ?? 0 }
    public var releasedAnywhere: Bool { released.contains { $0.contains(true) } }
}

public struct RegionDetector {
    public let configuration: MemorizationConfiguration

    public init(configuration: MemorizationConfiguration) { self.configuration = configuration }

    public func detect(_ field: CollapseField) -> RegionDetection {
        let c = configuration, grid = field.grid, cells = grid.count
        let t = Statistics.t95(df: field.draws - 1)
        let levels = field.levels.count

        let released: [[Bool]] = (0..<levels).map { l in
            let rho = GridMath.boxSmooth(field.retainedBase[l], grid: grid, radius: c.gateSmoothing)
            let spread = GridMath.boxSmooth(field.spreadBase[l], grid: grid, radius: c.gateSmoothing)
            let raw = field.retainedBase[l]
            return (0..<cells).map {
                rho[$0] >= c.releaseThreshold && rho[$0] <= c.mixingLimit && spread[$0] <= c.mixingLimit && raw[$0] <= 2 * c.mixingLimit
            }
        }
        let mean = field.collapse.map(GridMath.mean)
        let lower = zip(mean, field.collapse.map(GridMath.standardError)).map { m, se in zip(m, se).map { $0 - t * $1 } }

        func sustain(_ maps: [[Float]]) -> (value: [Float], level: [Int]) {
            var value = [Float](repeating: 0, count: cells), level = [Int](repeating: -1, count: cells)
            var best = [Float](repeating: -.infinity, count: cells)
            let pairs = (0..<max(levels - 1, 0)).filter { field.adjacency.indices.contains($0) && field.adjacency[$0] }
            for p in 0..<cells {
                if pairs.isEmpty {
                    for l in 0..<levels where released[l][p] && maps[l][p] > best[p] { best[p] = maps[l][p]; level[p] = l }
                } else {
                    for l in pairs where released[l][p] && released[l + 1][p] {
                        let v = min(maps[l][p], maps[l + 1][p])
                        if v > best[p] { best[p] = v; level[p] = maps[l][p] >= maps[l + 1][p] ? l : l + 1 }
                    }
                }
                value[p] = best[p].isFinite ? best[p] : 0
            }
            return (value, level)
        }
        let (rawS, peak) = sustain(mean)
        let s = GridMath.boxSmooth(rawS, grid: grid, radius: c.mapSmoothing)
        let lb = GridMath.boxSmooth(sustain(lower).value, grid: grid, radius: c.mapSmoothing)
        let promptMean = field.promptCollapse.map { $0.map(GridMath.mean) }
        let promptS = promptMean.map { GridMath.boxSmooth(sustain($0).value, grid: grid, radius: c.mapSmoothing) }

        let negative = sustain(mean.map { $0.map { -$0 } }).value
        let releasedCells = (0..<cells).filter { p in released.contains { $0[p] } }
        let negativeArea = releasedCells.isEmpty ? 0
            : Double(releasedCells.filter { negative[$0] >= c.collapseThreshold }.count) / Double(releasedCells.count)

        let candidate = (0..<cells).map { s[$0] >= c.collapseThreshold && lb[$0] > 0 }
        let clusters = GridMath.components(candidate, grid: grid).filter { $0.count >= c.minRegionCells }.map { cellsIn -> RegionCluster in
            let values = cellsIn.map { Double(s[$0]) }
            var byLevel = [Double](repeating: 0, count: levels)
            for p in cellsIn { for l in 0..<levels { byLevel[l] += Double(mean[l][p]) } }
            let peakLevel = byLevel.indices.max { byLevel[$0] < byLevel[$1] } ?? max(peak[cellsIn[0]], 0)
            return RegionCluster(cells: cellsIn, mass: values.reduce(0, +), meanCollapse: values.reduce(0, +) / Double(values.count),
                                 peakCollapse: values.max() ?? 0, peakLevel: peakLevel,
                                 meanPromptCollapse: promptS.map { p in cellsIn.reduce(0.0) { $0 + Double(p[$1]) } / Double(cellsIn.count) })
        }.sorted { $0.mass > $1.mass }
        return RegionDetection(grid: grid, sustained: s, lowerBound: lb, promptSustained: promptS, released: released,
                               clusters: clusters, negativeArea: negativeArea)
    }
}
