//
//  DeltaLinear.swift
//  ScorpionFlux2
//
//  A fine-tune applied at run time: y = base(x) + x·ΔWᵀ, ΔW kept factored (low rank) when it
//  is. One transformer serves both models of the comparison — the switch is on for the
//  target and off for the base — so the base is *exactly* the base (the delta branch is
//  skipped, not multiplied by zero) and every other weight is shared bit for bit.
//

import Foundation
import MLX
import MLXNN
import ScorpionKit

/// Shared by every delta layer of one transformer: target = on, base = off. Read when a
/// forward graph is built, so each call to the backend sets it first.
public final class AdapterSwitch: @unchecked Sendable {
    public var enabled = false
    public init() {}
}

final class DeltaLinear: Linear {
    let base: Linear
    /// (r, in) and (out, r) with the adapter scale folded in; or a dense (out, in) delta.
    let down: MLXArray?
    let up: MLXArray?
    let dense: MLXArray?
    let adapterSwitch: AdapterSwitch

    init(base: Linear, update: WeightUpdate, switch s: AdapterSwitch) {
        switch update {
        case .lowRank(let u, let d, let scale):
            up = (u * scale).asType(.float32)
            down = d.asType(.float32)
            dense = nil
        case .dense(let w):
            dense = w.asType(.float32)
            up = nil
            down = nil
        }
        self.base = base
        adapterSwitch = s
        super.init(weight: base.weight, bias: base.bias)
    }

    /// (out, in) of the delta.
    var deltaShape: (out: Int, in: Int) {
        if let up, let down { return (up.dim(0), down.dim(1)) }
        return (dense!.dim(0), dense!.dim(1))
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = base(x)
        guard adapterSwitch.enabled else { return y }
        let xf = x.asType(.float32)
        let delta = (up != nil && down != nil) ? matmul(matmul(xf, down!.transposed()), up!.transposed())
                                               : matmul(xf, dense!.transposed())
        return y + delta.asType(y.dtype)
    }
}

extension Linear {
    /// (out, in) of the layer as used (quantized layers store packed weights).
    var logicalShape: (out: Int, in: Int) {
        if let q = self as? QuantizedLinear { return (q.weight.dim(0), q.scales.dim(1) * q.groupSize) }
        return (weight.dim(0), weight.dim(1))
    }
}
