var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/Codegen.res.mjs
var Codegen_res_exports = {};
__export(Codegen_res_exports, {
  at: () => at2,
  binaryExpr: () => binaryExpr,
  binding: () => binding,
  computeDispatch: () => computeDispatch,
  computeDispatch2D: () => computeDispatch2D,
  computeDispatchInt4Opt: () => computeDispatchInt4Opt,
  genArangeKernel: () => genArangeKernel,
  genArgMaxKernel: () => genArgMaxKernel,
  genArgMinKernel: () => genArgMinKernel,
  genAttentionKernel: () => genAttentionKernel,
  genBatchNormKernel: () => genBatchNormKernel,
  genBatchedMatMulKernel: () => genBatchedMatMulKernel,
  genBinaryBroadcastKernel: () => genBinaryBroadcastKernel,
  genBroadcastKernel: () => genBroadcastKernel,
  genCastKernel: () => genCastKernel,
  genClipKernel: () => genClipKernel,
  genConcatKernel: () => genConcatKernel,
  genConv1DKernel: () => genConv1DKernel,
  genConv2DKernel: () => genConv2DKernel,
  genCumprodKernel: () => genCumprodKernel,
  genCumsumKernel: () => genCumsumKernel,
  genDenseKernel: () => genDenseKernel,
  genDropoutKernel: () => genDropoutKernel,
  genEmbeddingKernel: () => genEmbeddingKernel,
  genGQAAttentionKernel: () => genGQAAttentionKernel,
  genGRUCellKernel: () => genGRUCellKernel,
  genGatherKernel: () => genGatherKernel,
  genGlobalPoolKernel: () => genGlobalPoolKernel,
  genInt4MatMulKernel: () => genInt4MatMulKernel,
  genInt4MatMulTiledKernel: () => genInt4MatMulTiledKernel,
  genLSTMCellKernel: () => genLSTMCellKernel,
  genLayerNormKernel: () => genLayerNormKernel,
  genLogSoftmaxKernel: () => genLogSoftmaxKernel,
  genMatMulKernel: () => genMatMulKernel,
  genOneHotKernel: () => genOneHotKernel,
  genPadKernel: () => genPadKernel,
  genPool2DKernel: () => genPool2DKernel,
  genRMSNormKernel: () => genRMSNormKernel,
  genReduceKernel: () => genReduceKernel,
  genReshapeKernel: () => genReshapeKernel,
  genReverseKernel: () => genReverseKernel,
  genRoPEKernel: () => genRoPEKernel,
  genScatterKernel: () => genScatterKernel,
  genSliceKernel: () => genSliceKernel,
  genSmallMMatMulKernel: () => genSmallMMatMulKernel,
  genSoftmaxKernel: () => genSoftmaxKernel,
  genSortKernel: () => genSortKernel,
  genSplitKernel: () => genSplitKernel,
  genSqueezeKernel: () => genSqueezeKernel,
  genStackKernel: () => genStackKernel,
  genTileKernel: () => genTileKernel,
  genTopKKernel: () => genTopKKernel,
  genTransposeKernel: () => genTransposeKernel,
  genUnaryKernel: () => genUnaryKernel,
  genWhereKernel: () => genWhereKernel,
  generate: () => generate,
  mainEnd: () => mainEnd,
  mainSignature: () => mainSignature,
  reduceFinalize: () => reduceFinalize,
  reduceIdentity: () => reduceIdentity,
  reduceOp: () => reduceOp,
  shaderHeader: () => shaderHeader,
  storageBuffer: () => storageBuffer,
  unaryExpr: () => unaryExpr,
  uniformsStruct: () => uniformsStruct,
  workgroupSize: () => workgroupSize
});

// src/Shape.res.mjs
var Shape_res_exports = {};
__export(Shape_res_exports, {
  at: () => at,
  broadcast: () => broadcast,
  infer: () => infer,
  inferConcat: () => inferConcat,
  inferConv2D: () => inferConv2D,
  inferMatMul: () => inferMatMul,
  inferPool2D: () => inferPool2D,
  inferReduce: () => inferReduce,
  inferReshape: () => inferReshape,
  len: () => len,
  normalizeAxis: () => normalizeAxis,
  numElements: () => numElements
});

// node_modules/@rescript/runtime/lib/es6/Primitive_option.js
function some(x) {
  if (x === void 0) {
    return {
      BS_PRIVATE_NESTED_SOME_NONE: 0
    };
  } else if (x !== null && x.BS_PRIVATE_NESTED_SOME_NONE !== void 0) {
    return {
      BS_PRIVATE_NESTED_SOME_NONE: x.BS_PRIVATE_NESTED_SOME_NONE + 1 | 0
    };
  } else {
    return x;
  }
}
function valFromOption(x) {
  if (x === null || x.BS_PRIVATE_NESTED_SOME_NONE === void 0) {
    return x;
  }
  let depth = x.BS_PRIVATE_NESTED_SOME_NONE;
  if (depth === 0) {
    return;
  } else {
    return {
      BS_PRIVATE_NESTED_SOME_NONE: depth - 1 | 0
    };
  }
}

// node_modules/@rescript/runtime/lib/es6/Stdlib_Array.js
function make(length, x) {
  if (length <= 0) {
    return [];
  }
  let arr = new Array(length);
  arr.fill(x);
  return arr;
}
function fromInitializer(length, f) {
  if (length <= 0) {
    return [];
  }
  let arr = new Array(length);
  for (let i = 0; i < length; ++i) {
    arr[i] = f(i);
  }
  return arr;
}
function reduce(arr, init, f) {
  return arr.reduce(f, init);
}
function reduceWithIndex(arr, init, f) {
  return arr.reduce(f, init);
}
function filterMap(a, f) {
  let l = a.length;
  let r = new Array(l);
  let j = 0;
  for (let i = 0; i < l; ++i) {
    let v = a[i];
    let v$1 = f(v);
    if (v$1 !== void 0) {
      r[j] = valFromOption(v$1);
      j = j + 1 | 0;
    }
  }
  r.length = j;
  return r;
}

// node_modules/@rescript/runtime/lib/es6/Primitive_int.js
function compare(x, y) {
  if (x < y) {
    return -1;
  } else if (x === y) {
    return 0;
  } else {
    return 1;
  }
}
function max(x, y) {
  if (x > y) {
    return x;
  } else {
    return y;
  }
}
function div(x, y) {
  if (y === 0) {
    throw {
      RE_EXN_ID: "Division_by_zero",
      Error: new Error()
    };
  }
  return x / y | 0;
}
function mod_(x, y) {
  if (y === 0) {
    throw {
      RE_EXN_ID: "Division_by_zero",
      Error: new Error()
    };
  }
  return x % y;
}

// node_modules/@rescript/runtime/lib/es6/Stdlib_Option.js
function map(opt, f) {
  if (opt !== void 0) {
    return some(f(valFromOption(opt)));
  }
}
function flatMap(opt, f) {
  if (opt !== void 0) {
    return f(valFromOption(opt));
  }
}
function getOr(opt, $$default) {
  if (opt !== void 0) {
    return valFromOption(opt);
  } else {
    return $$default;
  }
}
function isSome(x) {
  return x !== void 0;
}
function isNone(x) {
  return x === void 0;
}

// src/Shape.res.mjs
function len(prim) {
  return prim.length;
}
function at(arr, i) {
  return getOr(arr[i], 0);
}
function numElements(shape) {
  return reduce(shape, 1, (a, b) => a * b | 0);
}
function normalizeAxis(axis, rank) {
  if (axis < 0) {
    return rank + axis | 0;
  } else {
    return axis;
  }
}
function broadcast(s1, s2) {
  let r1 = s1.length;
  let r2 = s2.length;
  let maxRank = max(r1, r2);
  let getDim = (s, r, i) => {
    let offset = maxRank - r | 0;
    if (i < offset) {
      return 1;
    } else {
      return at(s, i - offset | 0);
    }
  };
  let result = fromInitializer(maxRank, (i) => {
    let d1 = getDim(s1, r1, i);
    let d2 = getDim(s2, r2, i);
    if (d1 === d2) {
      return d1;
    } else if (d1 === 1) {
      return d2;
    } else if (d2 === 1) {
      return d1;
    } else {
      return;
    }
  });
  if (result.every(isSome)) {
    return result.map((__x) => getOr(__x, 0));
  }
}
function inferReduce(input2, axes, keepDims) {
  let rank = input2.length;
  let normAxes = axes.map((a) => normalizeAxis(a, rank));
  if (keepDims) {
    return input2.map((d, i) => {
      if (normAxes.includes(i)) {
        return 1;
      } else {
        return d;
      }
    });
  } else {
    return input2.filter((param, i) => !normAxes.includes(i));
  }
}
function inferMatMul(s1, s2) {
  let r1 = s1.length;
  let r2 = s2.length;
  if (r1 === 0) {
    return;
  }
  if (r1 === 1 && r2 === 1) {
    if (at(s1, 0) === at(s2, 0)) {
      return [1];
    } else {
      return;
    }
  }
  if (r2 === 0) {
    return;
  }
  if (r1 !== 1) {
    if (r2 !== 1) {
      let m = at(s1, r1 - 2 | 0);
      let k1 = at(s1, r1 - 1 | 0);
      let k2 = at(s2, r2 - 2 | 0);
      let n = at(s2, r2 - 1 | 0);
      if (k1 !== k2) {
        return;
      }
      let b1 = s1.slice(0, r1 - 2 | 0);
      let b2 = s2.slice(0, r2 - 2 | 0);
      return map(broadcast(b1, b2), (batch) => batch.concat([
        m,
        n
      ]));
    }
    let k = at(s1, r1 - 1 | 0);
    if (k === at(s2, 0)) {
      return s1.slice(0, r1 - 1 | 0);
    } else {
      return;
    }
  }
  let k$1 = at(s1, 0);
  let k2$1 = at(s2, r2 - 2 | 0);
  let n$1 = at(s2, r2 - 1 | 0);
  if (k$1 === k2$1) {
    return s2.slice(0, r2 - 2 | 0).concat([n$1]);
  }
}
function inferConv2D(input2, filters, param, param$1, padding, param$2) {
  let sW = param$1[1];
  let sH = param$1[0];
  if (input2.length !== 4) {
    return;
  }
  let batch = at(input2, 0);
  let inH = at(input2, 1);
  let inW = at(input2, 2);
  let effKH = ((param[0] - 1 | 0) * param$2[0] | 0) + 1 | 0;
  let effKW = ((param[1] - 1 | 0) * param$2[1] | 0) + 1 | 0;
  let match;
  if (typeof padding !== "object") {
    match = padding === "Same" ? [
      div((inH + sH | 0) - 1 | 0, sH),
      div((inW + sW | 0) - 1 | 0, sW)
    ] : [
      div(inH - effKH | 0, sH) + 1 | 0,
      div(inW - effKW | 0, sW) + 1 | 0
    ];
  } else {
    let pads = padding.pads;
    let p = (i) => getOr(pads[i], 0);
    match = [
      div(((inH + p(0) | 0) + p(2) | 0) - effKH | 0, sH) + 1 | 0,
      div(((inW + p(1) | 0) + p(3) | 0) - effKW | 0, sW) + 1 | 0
    ];
  }
  return [
    batch,
    match[0],
    match[1],
    filters
  ];
}
function inferPool2D(input2, param, param$1, padding) {
  let sW = param$1[1];
  let sH = param$1[0];
  let kW = param[1];
  let kH = param[0];
  if (input2.length !== 4) {
    return;
  }
  let batch = at(input2, 0);
  let inH = at(input2, 1);
  let inW = at(input2, 2);
  let channels = at(input2, 3);
  let match;
  if (typeof padding !== "object") {
    match = padding === "Same" ? [
      div((inH + sH | 0) - 1 | 0, sH),
      div((inW + sW | 0) - 1 | 0, sW)
    ] : [
      div(inH - kH | 0, sH) + 1 | 0,
      div(inW - kW | 0, sW) + 1 | 0
    ];
  } else {
    let pads = padding.pads;
    let p = (i) => getOr(pads[i], 0);
    match = [
      div(((inH + p(0) | 0) + p(2) | 0) - kH | 0, sH) + 1 | 0,
      div(((inW + p(1) | 0) + p(3) | 0) - kW | 0, sW) + 1 | 0
    ];
  }
  return [
    batch,
    match[0],
    match[1],
    channels
  ];
}
function inferReshape(input2, newShape) {
  let total = numElements(input2);
  let neg1Count = reduce(newShape, 0, (acc, d) => {
    if (d === -1) {
      return acc + 1 | 0;
    } else {
      return acc;
    }
  });
  if (neg1Count === 0) {
    if (numElements(newShape) === total) {
      return newShape;
    } else {
      return;
    }
  }
  if (neg1Count !== 1) {
    return;
  }
  let known = reduce(newShape, 1, (acc, d) => {
    if (d === -1) {
      return acc;
    } else {
      return acc * d | 0;
    }
  });
  let inferred = div(total, known);
  if ((inferred * known | 0) === total) {
    return newShape.map((d) => {
      if (d === -1) {
        return inferred;
      } else {
        return d;
      }
    });
  }
}
function inferConcat(inputs, axis) {
  if (inputs.length === 0) {
    return;
  }
  let first = getOr(inputs[0], []);
  let rank = first.length;
  let normAxis = normalizeAxis(axis, rank);
  let concatDim = reduce(inputs, 0, (acc, s) => acc + at(s, normAxis) | 0);
  return first.map((d, i) => {
    if (i === normAxis) {
      return concatDim;
    } else {
      return d;
    }
  });
}
function infer(op, inputs) {
  let get4 = (i) => getOr(inputs[i], []);
  let input2 = get4(0);
  let r = input2.length;
  let exit = 0;
  let kD;
  let kH;
  let kW;
  let sD;
  let sH;
  let sW;
  let padding;
  if (typeof op !== "object") {
    switch (op) {
      case "Add":
      case "Sub":
      case "Mul":
      case "Div":
      case "Pow":
      case "Mod":
      case "FloorDiv":
      case "Maximum":
      case "Minimum":
      case "Atan2":
      case "Equal":
      case "NotEqual":
      case "Greater":
      case "GreaterEqual":
      case "Less":
      case "LessEqual":
      case "And":
      case "Or":
      case "Xor":
        return broadcast(input2, get4(1));
      case "Where":
        return flatMap(broadcast(input2, get4(1)), (s) => broadcast(s, get4(2)));
      case "MatMul":
      case "BatchedMatMul":
        return inferMatMul(input2, get4(1));
      case "Shape":
        return [r];
      case "Dot":
      case "Size":
      case "Rank":
        return [1];
      case "GlobalMaxPool":
      case "GlobalAvgPool":
        exit = 9;
        break;
      case "NonZero":
        return;
      default:
        return input2;
    }
  } else {
    switch (op.TAG) {
      case "Reduce":
        return inferReduce(input2, op.axes, op.keepDims);
      case "ArgMax":
      case "ArgMin":
        exit = 1;
        break;
      case "MatMulInt4":
        let s2 = get4(1);
        let m = at(input2, r - 2 | 0);
        let n = at(s2, 0);
        return [
          m,
          n
        ];
      case "Gemm":
        return inferMatMul(input2, get4(1));
      case "Reshape":
        return inferReshape(input2, op.newShape);
      case "Squeeze":
        let axes = op.axes;
        return input2.filter((d, i) => !(axes.includes(i) && d === 1));
      case "Unsqueeze":
        let axes$1 = op.axes;
        let newRank = r + axes$1.length | 0;
        let sortedAxes = axes$1.toSorted(compare);
        let result = [];
        let inputIdx = 0;
        for (let i = 0; i < newRank; ++i) {
          if (sortedAxes.includes(i)) {
            result = result.concat([1]);
          } else {
            result = result.concat([at(input2, inputIdx)]);
            inputIdx = inputIdx + 1 | 0;
          }
        }
        return result;
      case "Flatten":
        let a = normalizeAxis(op.axis, r);
        return [
          numElements(input2.slice(0, a)),
          numElements(input2.slice(a, r))
        ];
      case "Transpose":
        let perm = op.perm;
        if (perm.length === r) {
          return perm.map((p) => at(input2, p));
        } else {
          return;
        }
      case "Broadcast":
        return op.targetShape;
      case "ExpandDims":
        let normAxis = normalizeAxis(op.axis, r + 1 | 0);
        let before = input2.slice(0, normAxis);
        let after = input2.slice(normAxis, r);
        return before.concat([1]).concat(after);
      case "Slice":
        let steps = op.steps;
        let ends = op.ends;
        let starts = op.starts;
        let result$1 = input2.slice();
        op.axes.forEach((ax, i) => {
          let dimSize = at(input2, ax);
          let start = getOr(starts[i], 0);
          let end_ = getOr(ends[i], dimSize);
          let step = getOr(steps[i], 1);
          let s = start < 0 ? dimSize + start | 0 : start;
          let e = end_ < 0 ? dimSize + end_ | 0 : end_;
          let size3 = div(((e - s | 0) + step | 0) - 1 | 0, step);
          result$1[ax] = size3;
        });
        return result$1;
      case "Gather":
      case "GatherElements":
        exit = 2;
        break;
      case "Concat":
        return inferConcat(inputs, op.axis);
      case "Split":
        let normAxis$1 = normalizeAxis(op.axis, r);
        let firstSize = getOr(op.splitSizes[0], at(input2, normAxis$1));
        return input2.map((d, i) => {
          if (i === normAxis$1) {
            return firstSize;
          } else {
            return d;
          }
        });
      case "Stack":
        let numInputs = inputs.length;
        let normAxis$2 = normalizeAxis(op.axis, r + 1 | 0);
        let before$1 = input2.slice(0, normAxis$2);
        let after$1 = input2.slice(normAxis$2, r);
        return before$1.concat([numInputs]).concat(after$1);
      case "Tile":
        let repeats = op.repeats;
        return input2.map((d, i) => d * getOr(repeats[i], 1) | 0);
      case "Pad":
        let pads = op.pads;
        return fromInitializer(r, (i) => {
          let before2 = getOr(pads[i], 0);
          let after2 = getOr(pads[r + i | 0], 0);
          return (at(input2, i) + before2 | 0) + after2 | 0;
        });
      case "Conv1D":
        if (r !== 3) {
          return;
        }
        let padding$1 = op.padding;
        let stride = op.stride;
        let batch = at(input2, 0);
        let inLen = at(input2, 1);
        let effK = ((op.kernel - 1 | 0) * op.dilation | 0) + 1 | 0;
        let outLen;
        if (typeof padding$1 !== "object") {
          outLen = padding$1 === "Same" ? div((inLen + stride | 0) - 1 | 0, stride) : div(inLen - effK | 0, stride) + 1 | 0;
        } else {
          let pads$1 = padding$1.pads;
          outLen = div(((inLen + at(pads$1, 0) | 0) + at(pads$1, 1) | 0) - effK | 0, stride) + 1 | 0;
        }
        return [
          batch,
          outLen,
          op.filters
        ];
      case "Conv2D":
        return inferConv2D(input2, op.filters, op.kernel, op.stride, op.padding, op.dilation);
      case "Conv3D":
        if (r !== 5) {
          return;
        }
        let match = op.dilation;
        let padding$2 = op.padding;
        let match$1 = op.stride;
        let match$2 = op.kernel;
        let batch$1 = at(input2, 0);
        let calc = (inSize, k2, s, d) => {
          let effK2 = ((k2 - 1 | 0) * d | 0) + 1 | 0;
          if (typeof padding$2 !== "object" && padding$2 === "Same") {
            return div((inSize + s | 0) - 1 | 0, s);
          }
          return div(inSize - effK2 | 0, s) + 1 | 0;
        };
        return [
          batch$1,
          calc(at(input2, 1), match$2[0], match$1[0], match[0]),
          calc(at(input2, 2), match$2[1], match$1[1], match[1]),
          calc(at(input2, 3), match$2[2], match$1[2], match[2]),
          op.filters
        ];
      case "ConvTranspose1D":
        if (r !== 3) {
          return;
        }
        let stride$1 = op.stride;
        let batch$2 = at(input2, 0);
        let inLen$1 = at(input2, 1);
        let tmp;
        let exit$1 = 0;
        let tmp$1 = op.padding;
        if (typeof tmp$1 !== "object" && tmp$1 === "Same") {
          tmp = inLen$1 * stride$1 | 0;
        } else {
          exit$1 = 10;
        }
        if (exit$1 === 10) {
          tmp = (inLen$1 * stride$1 | 0) + max(op.kernel - stride$1 | 0, 0) | 0;
        }
        let outLen$1 = tmp + op.outputPadding | 0;
        return [
          batch$2,
          outLen$1,
          op.filters
        ];
        break;
      case "ConvTranspose2D":
        if (r !== 4) {
          return;
        }
        let match$3 = op.outputPadding;
        let opW = match$3[1];
        let opH = match$3[0];
        let match$4 = op.stride;
        let sW$1 = match$4[1];
        let sH$1 = match$4[0];
        let match$5 = op.kernel;
        let batch$3 = at(input2, 0);
        let inH = at(input2, 1);
        let inW = at(input2, 2);
        let match$6;
        let exit$2 = 0;
        let tmp$2 = op.padding;
        if (typeof tmp$2 !== "object" && tmp$2 === "Same") {
          match$6 = [
            (inH * sH$1 | 0) + opH | 0,
            (inW * sW$1 | 0) + opW | 0
          ];
        } else {
          exit$2 = 10;
        }
        if (exit$2 === 10) {
          match$6 = [
            ((inH * sH$1 | 0) + max(match$5[0] - sH$1 | 0, 0) | 0) + opH | 0,
            ((inW * sW$1 | 0) + max(match$5[1] - sW$1 | 0, 0) | 0) + opW | 0
          ];
        }
        return [
          batch$3,
          match$6[0],
          match$6[1],
          op.filters
        ];
        break;
      case "ConvTranspose3D":
        if (r !== 5) {
          return;
        }
        let match$7 = op.outputPadding;
        let padding$3 = op.padding;
        let match$8 = op.stride;
        let match$9 = op.kernel;
        let batch$4 = at(input2, 0);
        let calc$1 = (inSize, k2, s, op2) => {
          if (typeof padding$3 !== "object" && padding$3 === "Same") {
            return (inSize * s | 0) + op2 | 0;
          }
          return ((inSize * s | 0) + max(k2 - s | 0, 0) | 0) + op2 | 0;
        };
        return [
          batch$4,
          calc$1(at(input2, 1), match$9[0], match$8[0], match$7[0]),
          calc$1(at(input2, 2), match$9[1], match$8[1], match$7[1]),
          calc$1(at(input2, 3), match$9[2], match$8[2], match$7[2]),
          op.filters
        ];
      case "DepthwiseConv2D":
        if (r !== 4) {
          return;
        }
        let match$10 = op.dilation;
        let match$11 = op.stride;
        let sW$2 = match$11[1];
        let sH$2 = match$11[0];
        let match$12 = op.kernel;
        let batch$5 = at(input2, 0);
        let inH$1 = at(input2, 1);
        let inW$1 = at(input2, 2);
        let channels = at(input2, 3);
        let effKH = ((match$12[0] - 1 | 0) * match$10[0] | 0) + 1 | 0;
        let effKW = ((match$12[1] - 1 | 0) * match$10[1] | 0) + 1 | 0;
        let match$13;
        let exit$3 = 0;
        let tmp$3 = op.padding;
        if (typeof tmp$3 !== "object" && tmp$3 === "Same") {
          match$13 = [
            div((inH$1 + sH$2 | 0) - 1 | 0, sH$2),
            div((inW$1 + sW$2 | 0) - 1 | 0, sW$2)
          ];
        } else {
          exit$3 = 10;
        }
        if (exit$3 === 10) {
          match$13 = [
            div(inH$1 - effKH | 0, sH$2) + 1 | 0,
            div(inW$1 - effKW | 0, sW$2) + 1 | 0
          ];
        }
        return [
          batch$5,
          match$13[0],
          match$13[1],
          channels * op.depthMultiplier | 0
        ];
        break;
      case "MaxPool1D":
        if (r !== 3) {
          return;
        }
        let stride$2 = op.stride;
        let batch$6 = at(input2, 0);
        let inLen$2 = at(input2, 1);
        let channels$1 = at(input2, 2);
        let outLen$2;
        let exit$4 = 0;
        let tmp$4 = op.padding;
        if (typeof tmp$4 !== "object" && tmp$4 === "Same") {
          outLen$2 = div((inLen$2 + stride$2 | 0) - 1 | 0, stride$2);
        } else {
          exit$4 = 10;
        }
        if (exit$4 === 10) {
          outLen$2 = div(inLen$2 - op.kernel | 0, stride$2) + 1 | 0;
        }
        return [
          batch$6,
          outLen$2,
          channels$1
        ];
        break;
      case "MaxPool3D":
        let match$14 = op.stride;
        let match$15 = op.kernel;
        kD = match$15[0];
        kH = match$15[1];
        kW = match$15[2];
        sD = match$14[0];
        sH = match$14[1];
        sW = match$14[2];
        padding = op.padding;
        exit = 8;
        break;
      case "AvgPool1D":
        if (r !== 3) {
          return;
        }
        let stride$3 = op.stride;
        let batch$7 = at(input2, 0);
        let inLen$3 = at(input2, 1);
        let channels$2 = at(input2, 2);
        let outLen$3;
        let exit$5 = 0;
        let tmp$5 = op.padding;
        if (typeof tmp$5 !== "object" && tmp$5 === "Same") {
          outLen$3 = div((inLen$3 + stride$3 | 0) - 1 | 0, stride$3);
        } else {
          exit$5 = 10;
        }
        if (exit$5 === 10) {
          outLen$3 = div(inLen$3 - op.kernel | 0, stride$3) + 1 | 0;
        }
        return [
          batch$7,
          outLen$3,
          channels$2
        ];
        break;
      case "MaxPool2D":
      case "AvgPool2D":
        exit = 3;
        break;
      case "AvgPool3D":
        let match$16 = op.stride;
        let match$17 = op.kernel;
        kD = match$17[0];
        kH = match$17[1];
        kW = match$17[2];
        sD = match$16[0];
        sH = match$16[1];
        sW = match$16[2];
        padding = op.padding;
        exit = 8;
        break;
      case "LpPool":
        let match$18 = op.stride;
        let match$19 = op.kernel;
        return inferPool2D(input2, [
          match$19[0],
          match$19[1]
        ], [
          match$18[0],
          match$18[1]
        ], "Valid");
      case "AdaptiveAvgPool1D":
      case "AdaptiveMaxPool1D":
        exit = 4;
        break;
      case "AdaptiveAvgPool2D":
      case "AdaptiveMaxPool2D":
        exit = 5;
        break;
      case "Dense":
        if (r > 0) {
          return input2.slice(0, r - 1 | 0).concat([op.units]);
        } else {
          return;
        }
      case "Attention":
      case "MultiHeadAttention":
        exit = 6;
        break;
      case "RNN":
      case "LSTM":
      case "GRU":
        exit = 7;
        break;
      case "Resize":
        let sizes = op.sizes;
        if (sizes !== void 0) {
          if (r === 4) {
            return [
              at(input2, 0),
              at(sizes, 0),
              at(sizes, 1),
              at(input2, 3)
            ];
          } else {
            return;
          }
        } else {
          return input2;
        }
      case "SpaceToDepth":
        let blockSize = op.blockSize;
        if (r === 4) {
          return [
            at(input2, 0),
            div(at(input2, 1), blockSize),
            div(at(input2, 2), blockSize),
            (at(input2, 3) * blockSize | 0) * blockSize | 0
          ];
        } else {
          return;
        }
      case "DepthToSpace":
        let blockSize$1 = op.blockSize;
        if (r === 4) {
          return [
            at(input2, 0),
            at(input2, 1) * blockSize$1 | 0,
            at(input2, 2) * blockSize$1 | 0,
            div(at(input2, 3), blockSize$1 * blockSize$1 | 0)
          ];
        } else {
          return;
        }
      case "GridSample":
        let grid = get4(1);
        if (r === 4 && grid.length === 4) {
          return [
            at(input2, 0),
            at(grid, 1),
            at(grid, 2),
            at(input2, 3)
          ];
        } else {
          return;
        }
      case "RoiAlign":
        let rois = get4(1);
        if (r === 4 && rois.length === 2) {
          return [
            at(rois, 0),
            op.outputHeight,
            op.outputWidth,
            at(input2, 1)
          ];
        } else {
          return;
        }
      case "OneHot":
        let normAxis$3 = normalizeAxis(op.axis, r + 1 | 0);
        let before$2 = input2.slice(0, normAxis$3);
        let after$2 = input2.slice(normAxis$3, r);
        return before$2.concat([op.depth]).concat(after$2);
      case "Embedding":
        return input2.concat([op.embeddingDim]);
      case "TopK":
        let k = op.k;
        let normAxis$4 = normalizeAxis(op.axis, r);
        return input2.map((d, i) => {
          if (i === normAxis$4) {
            return k;
          } else {
            return d;
          }
        });
      case "Einsum":
      case "GatherND":
      case "NonMaxSuppression":
      case "Unique":
        return;
      case "Input":
      case "Const":
      case "RandomNormal":
      case "RandomUniform":
        return op.shape;
      case "MSELoss":
      case "CrossEntropyLoss":
      case "BCELoss":
      case "BCEWithLogitsLoss":
      case "NLLLoss":
      case "CTCLoss":
      case "HuberLoss":
      case "SmoothL1Loss":
      case "TripletMarginLoss":
      case "CosineEmbeddingLoss":
        return [1];
      default:
        return input2;
    }
  }
  switch (exit) {
    case 1:
      return inferReduce(input2, [op.axis], op.keepDims);
    case 2:
      let indices = get4(1);
      let normAxis$5 = normalizeAxis(op.axis, r);
      let before$3 = input2.slice(0, normAxis$5);
      let after$3 = input2.slice(normAxis$5 + 1 | 0, r);
      return before$3.concat(indices).concat(after$3);
    case 3:
      return inferPool2D(input2, op.kernel, op.stride, op.padding);
    case 4:
      if (r === 3) {
        return [
          at(input2, 0),
          op.outputSize,
          at(input2, 2)
        ];
      } else {
        return;
      }
    case 5:
      let match$20 = op.outputSize;
      if (r === 4) {
        return [
          at(input2, 0),
          match$20[0],
          match$20[1],
          at(input2, 3)
        ];
      } else {
        return;
      }
    case 6:
      if (r >= 2) {
        return [
          at(input2, 0),
          at(input2, 1),
          op.dim
        ];
      } else {
        return;
      }
    case 7:
      if (r >= 2) {
        return [
          at(input2, 0),
          at(input2, 1),
          op.hiddenSize
        ];
      } else {
        return;
      }
    case 8:
      if (r !== 5) {
        return;
      }
      let batch$8 = at(input2, 0);
      let channels$3 = at(input2, 4);
      let calc$2 = (inSize, k, s) => {
        if (typeof padding !== "object" && padding === "Same") {
          return div((inSize + s | 0) - 1 | 0, s);
        }
        return div(inSize - k | 0, s) + 1 | 0;
      };
      return [
        batch$8,
        calc$2(at(input2, 1), kD, sD),
        calc$2(at(input2, 2), kH, sH),
        calc$2(at(input2, 3), kW, sW),
        channels$3
      ];
    case 9:
      if (r === 4) {
        return [
          at(input2, 0),
          1,
          1,
          at(input2, 3)
        ];
      } else {
        return;
      }
  }
}

// src/Codegen.res.mjs
function binding(idx) {
  return `@group(0) @binding(` + idx.toString() + `)`;
}
function storageBuffer(idx, name, access) {
  let accessMode;
  switch (access) {
    case "Uniform":
    case "ReadOnly":
      accessMode = "read";
      break;
    case "Storage":
    case "ReadWrite":
      accessMode = "read_write";
      break;
  }
  return binding(idx) + ` var<storage, ` + accessMode + `> ` + name + `: array<f32>;`;
}
function shaderHeader(numBuffers) {
  let buffers = fromInitializer(numBuffers, (i) => {
    let name = i === (numBuffers - 1 | 0) ? "output" : "input" + i.toString();
    let access = i === (numBuffers - 1 | 0) ? "ReadWrite" : "ReadOnly";
    return storageBuffer(i, name, access);
  });
  return buffers.join("\n");
}
function uniformsStruct(fields) {
  return `struct Uniforms {
` + fields + `
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;`;
}
var mainSignature = `@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;`;
var mainEnd = "}";
function unaryExpr(op, input2) {
  if (typeof op !== "object") {
    switch (op) {
      case "Identity":
        return input2;
      case "Neg":
        return `-` + input2;
      case "Abs":
        return `abs(` + input2 + `)`;
      case "Sign":
        return `sign(` + input2 + `)`;
      case "Reciprocal":
        return `1.0 / ` + input2;
      case "Floor":
        return `floor(` + input2 + `)`;
      case "Ceil":
        return `ceil(` + input2 + `)`;
      case "Round":
        return `round(` + input2 + `)`;
      case "Sqrt":
        return `sqrt(` + input2 + `)`;
      case "Exp":
        return `exp(` + input2 + `)`;
      case "Log":
        return `log(` + input2 + `)`;
      case "Log2":
        return `log2(` + input2 + `)`;
      case "Log10":
        return `log(` + input2 + `) / 2.302585`;
      case "Sin":
        return `sin(` + input2 + `)`;
      case "Cos":
        return `cos(` + input2 + `)`;
      case "Tan":
        return `tan(` + input2 + `)`;
      case "Asin":
        return `asin(` + input2 + `)`;
      case "Acos":
        return `acos(` + input2 + `)`;
      case "Atan":
        return `atan(` + input2 + `)`;
      case "Sinh":
        return `sinh(` + input2 + `)`;
      case "Cosh":
        return `cosh(` + input2 + `)`;
      case "Tanh":
        return `tanh(` + input2 + `)`;
      case "Asinh":
        return `asinh(` + input2 + `)`;
      case "Acosh":
        return `acosh(` + input2 + `)`;
      case "Atanh":
        return `atanh(` + input2 + `)`;
      case "ReLU":
        return `max(` + input2 + `, 0.0)`;
      case "Sigmoid":
        return `1.0 / (1.0 + exp(-` + input2 + `))`;
      case "Softplus":
        return `log(1.0 + exp(` + input2 + `))`;
      case "Softsign":
        return input2 + ` / (1.0 + abs(` + input2 + `))`;
      case "GeLU":
        return `0.5 * ` + input2 + ` * (1.0 + tanh(0.7978845608 * (` + input2 + ` + 0.044715 * ` + input2 + ` * ` + input2 + ` * ` + input2 + `)))`;
      case "SiLU":
        return input2 + ` / (1.0 + exp(-` + input2 + `))`;
      case "Mish":
        return input2 + ` * tanh(log(1.0 + exp(` + input2 + `)))`;
      case "Not":
        return `f32(` + input2 + ` == 0.0)`;
      default:
        return;
    }
  } else {
    switch (op.TAG) {
      case "LeakyReLU":
        return `select(` + op.alpha.toString() + ` * ` + input2 + `, ` + input2 + `, ` + input2 + ` > 0.0)`;
      case "ELU":
        return `select(` + op.alpha.toString() + ` * (exp(` + input2 + `) - 1.0), ` + input2 + `, ` + input2 + ` > 0.0)`;
      default:
        return;
    }
  }
}
function binaryExpr(op, a, b) {
  if (typeof op === "object") {
    return;
  }
  switch (op) {
    case "Add":
      return a + ` + ` + b;
    case "Sub":
      return a + ` - ` + b;
    case "Mul":
      return a + ` * ` + b;
    case "Div":
      return a + ` / ` + b;
    case "Pow":
      return `pow(` + a + `, ` + b + `)`;
    case "Mod":
      return a + ` % ` + b;
    case "FloorDiv":
      return `floor(` + a + ` / ` + b + `)`;
    case "Maximum":
      return `max(` + a + `, ` + b + `)`;
    case "Minimum":
      return `min(` + a + `, ` + b + `)`;
    case "Atan2":
      return `atan2(` + a + `, ` + b + `)`;
    case "Equal":
      return `f32(` + a + ` == ` + b + `)`;
    case "NotEqual":
      return `f32(` + a + ` != ` + b + `)`;
    case "Greater":
      return `f32(` + a + ` > ` + b + `)`;
    case "GreaterEqual":
      return `f32(` + a + ` >= ` + b + `)`;
    case "Less":
      return `f32(` + a + ` < ` + b + `)`;
    case "LessEqual":
      return `f32(` + a + ` <= ` + b + `)`;
    case "And":
      return `f32(` + a + ` != 0.0 && ` + b + ` != 0.0)`;
    case "Or":
      return `f32(` + a + ` != 0.0 || ` + b + ` != 0.0)`;
    case "Xor":
      return `f32((` + a + ` != 0.0) != (` + b + ` != 0.0))`;
    default:
      return;
  }
}
function reduceIdentity(op) {
  switch (op) {
    case "Max":
      return "-3.402823e+38";
    case "Min":
      return "3.402823e+38";
    case "Prod":
      return "1.0";
    default:
      return "0.0";
  }
}
function reduceOp(op, acc, val) {
  let exit = 0;
  switch (op) {
    case "Sum":
    case "Mean":
      exit = 1;
      break;
    case "Max":
      return `max(` + acc + `, ` + val + `)`;
    case "Min":
      return `min(` + acc + `, ` + val + `)`;
    case "Prod":
      return acc + ` * ` + val;
    case "L1":
      return acc + ` + abs(` + val + `)`;
    case "LogSum":
      return acc + ` + ` + val;
    case "LogSumExp":
      return acc + ` + exp(` + val + `)`;
    case "L2":
    case "SumSquare":
      exit = 2;
      break;
  }
  switch (exit) {
    case 1:
      return acc + ` + ` + val;
    case 2:
      return acc + ` + ` + val + ` * ` + val;
  }
}
function reduceFinalize(op, acc, count) {
  switch (op) {
    case "Mean":
      return acc + ` / f32(` + count + `)`;
    case "L2":
      return `sqrt(` + acc + `)`;
    case "LogSum":
    case "LogSumExp":
      return `log(` + acc + `)`;
    default:
      return acc;
  }
}
function genReduceKernel(reduceType, inputShape, axes, _keepDims) {
  let rank = inputShape.length;
  if (rank < 1) {
    return;
  }
  let normAxes = axes.map((a) => {
    if (a < 0) {
      return rank + a | 0;
    } else {
      return a;
    }
  });
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let inputSize = numElements(inputShape);
  let reduceSize = reduceWithIndex(inputShape, 1, (acc, dim, i) => {
    if (normAxes.includes(i)) {
      return acc * dim | 0;
    } else {
      return acc;
    }
  });
  let outputSize = div(inputSize, reduceSize);
  let shapeStr = inputShape.map((d) => d.toString()).join(", ");
  let stridesStr = inputStrides.map((d) => d.toString()).join(", ");
  let axesStr = normAxes.map((d) => d.toString()).join(", ");
  let numAxes = normAxes.length;
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const RANK = ` + rank.toString() + `u;
const INPUT_SIZE = ` + inputSize.toString() + `u;
const OUTPUT_SIZE = ` + outputSize.toString() + `u;
const REDUCE_SIZE = ` + reduceSize.toString() + `u;
const NUM_AXES = ` + numAxes.toString() + `u;
const SHAPE = array<u32, ` + rank.toString() + `>(` + shapeStr + `);
const STRIDES = array<u32, ` + rank.toString() + `>(` + stridesStr + `);
const AXES = array<u32, ` + numAxes.toString() + `>(` + axesStr + `);

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
  if (outIdx >= OUTPUT_SIZE) { return; }
  
  // Convert output index to output coordinates (skipping reduced axes)
  var outCoords: array<u32, ` + rank.toString() + `>;
  var remaining = outIdx;
  var outDimIdx = 0u;
  for (var i = 0u; i < RANK; i = i + 1u) {
    if (!isReduceAxis(i)) {
      var outDimSize = 1u;
      for (var j = i + 1u; j < RANK; j = j + 1u) {
        if (!isReduceAxis(j)) {
          outDimSize = outDimSize * SHAPE[j];
        }
      }
      if (outDimSize > 0u) {
        outCoords[i] = remaining / outDimSize;
        remaining = remaining % outDimSize;
      }
    } else {
      outCoords[i] = 0u;
    }
  }
  
  // Iterate over all reduction indices
  var acc = ` + reduceIdentity(reduceType) + `;
  var count = 0u;
  
  for (var reduceIdx = 0u; reduceIdx < REDUCE_SIZE; reduceIdx = reduceIdx + 1u) {
    // Convert reduceIdx to coordinates for reduced axes
    var coords = outCoords;
    var rem = reduceIdx;
    for (var i = 0u; i < RANK; i = i + 1u) {
      let ri = RANK - 1u - i;
      if (isReduceAxis(ri)) {
        coords[ri] = rem % SHAPE[ri];
        rem = rem / SHAPE[ri];
      }
    }
    
    // Compute input index
    var inIdx = 0u;
    for (var i = 0u; i < RANK; i = i + 1u) {
      inIdx = inIdx + coords[i] * STRIDES[i];
    }
    
    let val = input[inIdx];
    acc = ` + reduceOp(reduceType, "acc", "val") + `;
    count = count + 1u;
  }
  
  output[outIdx] = ` + reduceFinalize(reduceType, "acc", "count") + `;
}`;
  return {
    name: "reduce_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: inputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSmallMMatMulKernel(m, k, n) {
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = M * N;
  if (idx >= total) { return; }
  
  let row = idx / N;
  let col = idx % N;
  
  var sum = 0.0;
  for (var i = 0u; i < K; i = i + 1u) {
    sum = sum + a[row * K + i] * b[i * N + col];
  }
  output[idx] = sum;
}`;
  return {
    name: "matmul_small_" + m.toString() + "x" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (m * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 1,
        size: (k * n | 0) << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 2,
        size: (m * n | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genInt4MatMulKernel(m, k, n, groupSize) {
  let actualGroupSize = 128;
  let numGroups = div((k + actualGroupSize | 0) - 1 | 0, actualGroupSize);
  let packedPerGroup = actualGroupSize / 8 | 0;
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const N = ` + n.toString() + `u;
const NUM_GROUPS = ` + numGroups.toString() + `u;
const PACKED_PER_GROUP = ` + packedPerGroup.toString() + `u;

@compute @workgroup_size(` + 64 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= N) { return; }
  
  var sum = 0.0;
  
  for (var g = 0u; g < NUM_GROUPS; g++) {
    // Load scale once per group (not per weight)
    let s = scales[g * N + col];
    let base = g * PACKED_PER_GROUP;
    var acc = 0.0;
    
    // ========================================
    // First half of group: 64 weights (8 packed u32)
    // ========================================
    // Coalesced loads: adjacent threads read adjacent memory
    let p0 = b_packed[base * N + col];
    let p1 = b_packed[(base + 1u) * N + col];
    let p2 = b_packed[(base + 2u) * N + col];
    let p3 = b_packed[(base + 3u) * N + col];
    let p4 = b_packed[(base + 4u) * N + col];
    let p5 = b_packed[(base + 5u) * N + col];
    let p6 = b_packed[(base + 6u) * N + col];
    let p7 = b_packed[(base + 7u) * N + col];
    
    // Load activations as vec4 for efficient dot products
    let k0 = base * 8u;
    let a0 = a[k0/4u]; let a1 = a[k0/4u+1u]; let a2 = a[k0/4u+2u]; let a3 = a[k0/4u+3u];
    let a4 = a[k0/4u+4u]; let a5 = a[k0/4u+5u]; let a6 = a[k0/4u+6u]; let a7 = a[k0/4u+7u];
    let a8 = a[k0/4u+8u]; let a9 = a[k0/4u+9u]; let a10 = a[k0/4u+10u]; let a11 = a[k0/4u+11u];
    let a12 = a[k0/4u+12u]; let a13 = a[k0/4u+13u]; let a14 = a[k0/4u+14u]; let a15 = a[k0/4u+15u];
    
    // Unrolled dot products - dequantize inline (subtract 8 for signed int4)
    acc += dot(a0, vec4<f32>(f32((p0>>0u)&0xFu)-8.0, f32((p0>>4u)&0xFu)-8.0, f32((p0>>8u)&0xFu)-8.0, f32((p0>>12u)&0xFu)-8.0));
    acc += dot(a1, vec4<f32>(f32((p0>>16u)&0xFu)-8.0, f32((p0>>20u)&0xFu)-8.0, f32((p0>>24u)&0xFu)-8.0, f32((p0>>28u)&0xFu)-8.0));
    acc += dot(a2, vec4<f32>(f32((p1>>0u)&0xFu)-8.0, f32((p1>>4u)&0xFu)-8.0, f32((p1>>8u)&0xFu)-8.0, f32((p1>>12u)&0xFu)-8.0));
    acc += dot(a3, vec4<f32>(f32((p1>>16u)&0xFu)-8.0, f32((p1>>20u)&0xFu)-8.0, f32((p1>>24u)&0xFu)-8.0, f32((p1>>28u)&0xFu)-8.0));
    acc += dot(a4, vec4<f32>(f32((p2>>0u)&0xFu)-8.0, f32((p2>>4u)&0xFu)-8.0, f32((p2>>8u)&0xFu)-8.0, f32((p2>>12u)&0xFu)-8.0));
    acc += dot(a5, vec4<f32>(f32((p2>>16u)&0xFu)-8.0, f32((p2>>20u)&0xFu)-8.0, f32((p2>>24u)&0xFu)-8.0, f32((p2>>28u)&0xFu)-8.0));
    acc += dot(a6, vec4<f32>(f32((p3>>0u)&0xFu)-8.0, f32((p3>>4u)&0xFu)-8.0, f32((p3>>8u)&0xFu)-8.0, f32((p3>>12u)&0xFu)-8.0));
    acc += dot(a7, vec4<f32>(f32((p3>>16u)&0xFu)-8.0, f32((p3>>20u)&0xFu)-8.0, f32((p3>>24u)&0xFu)-8.0, f32((p3>>28u)&0xFu)-8.0));
    acc += dot(a8, vec4<f32>(f32((p4>>0u)&0xFu)-8.0, f32((p4>>4u)&0xFu)-8.0, f32((p4>>8u)&0xFu)-8.0, f32((p4>>12u)&0xFu)-8.0));
    acc += dot(a9, vec4<f32>(f32((p4>>16u)&0xFu)-8.0, f32((p4>>20u)&0xFu)-8.0, f32((p4>>24u)&0xFu)-8.0, f32((p4>>28u)&0xFu)-8.0));
    acc += dot(a10, vec4<f32>(f32((p5>>0u)&0xFu)-8.0, f32((p5>>4u)&0xFu)-8.0, f32((p5>>8u)&0xFu)-8.0, f32((p5>>12u)&0xFu)-8.0));
    acc += dot(a11, vec4<f32>(f32((p5>>16u)&0xFu)-8.0, f32((p5>>20u)&0xFu)-8.0, f32((p5>>24u)&0xFu)-8.0, f32((p5>>28u)&0xFu)-8.0));
    acc += dot(a12, vec4<f32>(f32((p6>>0u)&0xFu)-8.0, f32((p6>>4u)&0xFu)-8.0, f32((p6>>8u)&0xFu)-8.0, f32((p6>>12u)&0xFu)-8.0));
    acc += dot(a13, vec4<f32>(f32((p6>>16u)&0xFu)-8.0, f32((p6>>20u)&0xFu)-8.0, f32((p6>>24u)&0xFu)-8.0, f32((p6>>28u)&0xFu)-8.0));
    acc += dot(a14, vec4<f32>(f32((p7>>0u)&0xFu)-8.0, f32((p7>>4u)&0xFu)-8.0, f32((p7>>8u)&0xFu)-8.0, f32((p7>>12u)&0xFu)-8.0));
    acc += dot(a15, vec4<f32>(f32((p7>>16u)&0xFu)-8.0, f32((p7>>20u)&0xFu)-8.0, f32((p7>>24u)&0xFu)-8.0, f32((p7>>28u)&0xFu)-8.0));
    
    // ========================================
    // Second half of group: 64 weights (8 packed u32)
    // ========================================
    let q0 = b_packed[(base + 8u) * N + col];
    let q1 = b_packed[(base + 9u) * N + col];
    let q2 = b_packed[(base + 10u) * N + col];
    let q3 = b_packed[(base + 11u) * N + col];
    let q4 = b_packed[(base + 12u) * N + col];
    let q5 = b_packed[(base + 13u) * N + col];
    let q6 = b_packed[(base + 14u) * N + col];
    let q7 = b_packed[(base + 15u) * N + col];
    
    let k1 = (base + 8u) * 8u;
    let b0 = a[k1/4u]; let b1 = a[k1/4u+1u]; let b2 = a[k1/4u+2u]; let b3 = a[k1/4u+3u];
    let b4 = a[k1/4u+4u]; let b5 = a[k1/4u+5u]; let b6 = a[k1/4u+6u]; let b7 = a[k1/4u+7u];
    let b8 = a[k1/4u+8u]; let b9 = a[k1/4u+9u]; let b10 = a[k1/4u+10u]; let b11 = a[k1/4u+11u];
    let b12 = a[k1/4u+12u]; let b13 = a[k1/4u+13u]; let b14 = a[k1/4u+14u]; let b15 = a[k1/4u+15u];
    
    acc += dot(b0, vec4<f32>(f32((q0>>0u)&0xFu)-8.0, f32((q0>>4u)&0xFu)-8.0, f32((q0>>8u)&0xFu)-8.0, f32((q0>>12u)&0xFu)-8.0));
    acc += dot(b1, vec4<f32>(f32((q0>>16u)&0xFu)-8.0, f32((q0>>20u)&0xFu)-8.0, f32((q0>>24u)&0xFu)-8.0, f32((q0>>28u)&0xFu)-8.0));
    acc += dot(b2, vec4<f32>(f32((q1>>0u)&0xFu)-8.0, f32((q1>>4u)&0xFu)-8.0, f32((q1>>8u)&0xFu)-8.0, f32((q1>>12u)&0xFu)-8.0));
    acc += dot(b3, vec4<f32>(f32((q1>>16u)&0xFu)-8.0, f32((q1>>20u)&0xFu)-8.0, f32((q1>>24u)&0xFu)-8.0, f32((q1>>28u)&0xFu)-8.0));
    acc += dot(b4, vec4<f32>(f32((q2>>0u)&0xFu)-8.0, f32((q2>>4u)&0xFu)-8.0, f32((q2>>8u)&0xFu)-8.0, f32((q2>>12u)&0xFu)-8.0));
    acc += dot(b5, vec4<f32>(f32((q2>>16u)&0xFu)-8.0, f32((q2>>20u)&0xFu)-8.0, f32((q2>>24u)&0xFu)-8.0, f32((q2>>28u)&0xFu)-8.0));
    acc += dot(b6, vec4<f32>(f32((q3>>0u)&0xFu)-8.0, f32((q3>>4u)&0xFu)-8.0, f32((q3>>8u)&0xFu)-8.0, f32((q3>>12u)&0xFu)-8.0));
    acc += dot(b7, vec4<f32>(f32((q3>>16u)&0xFu)-8.0, f32((q3>>20u)&0xFu)-8.0, f32((q3>>24u)&0xFu)-8.0, f32((q3>>28u)&0xFu)-8.0));
    acc += dot(b8, vec4<f32>(f32((q4>>0u)&0xFu)-8.0, f32((q4>>4u)&0xFu)-8.0, f32((q4>>8u)&0xFu)-8.0, f32((q4>>12u)&0xFu)-8.0));
    acc += dot(b9, vec4<f32>(f32((q4>>16u)&0xFu)-8.0, f32((q4>>20u)&0xFu)-8.0, f32((q4>>24u)&0xFu)-8.0, f32((q4>>28u)&0xFu)-8.0));
    acc += dot(b10, vec4<f32>(f32((q5>>0u)&0xFu)-8.0, f32((q5>>4u)&0xFu)-8.0, f32((q5>>8u)&0xFu)-8.0, f32((q5>>12u)&0xFu)-8.0));
    acc += dot(b11, vec4<f32>(f32((q5>>16u)&0xFu)-8.0, f32((q5>>20u)&0xFu)-8.0, f32((q5>>24u)&0xFu)-8.0, f32((q5>>28u)&0xFu)-8.0));
    acc += dot(b12, vec4<f32>(f32((q6>>0u)&0xFu)-8.0, f32((q6>>4u)&0xFu)-8.0, f32((q6>>8u)&0xFu)-8.0, f32((q6>>12u)&0xFu)-8.0));
    acc += dot(b13, vec4<f32>(f32((q6>>16u)&0xFu)-8.0, f32((q6>>20u)&0xFu)-8.0, f32((q6>>24u)&0xFu)-8.0, f32((q6>>28u)&0xFu)-8.0));
    acc += dot(b14, vec4<f32>(f32((q7>>0u)&0xFu)-8.0, f32((q7>>4u)&0xFu)-8.0, f32((q7>>8u)&0xFu)-8.0, f32((q7>>12u)&0xFu)-8.0));
    acc += dot(b15, vec4<f32>(f32((q7>>16u)&0xFu)-8.0, f32((q7>>20u)&0xFu)-8.0, f32((q7>>24u)&0xFu)-8.0, f32((q7>>28u)&0xFu)-8.0));
    
    // Apply scale once per group (not per weight)
    sum += acc * s;
  }
  
  output[col] = sum;
}`;
  return {
    name: "matmul_int4_opt_" + m.toString() + "x" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: k << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 1,
        size: ((numGroups * packedPerGroup | 0) * n | 0) << 2,
        usage: "ReadOnly",
        name: "b_packed"
      },
      {
        binding: 2,
        size: (numGroups * n | 0) << 2,
        usage: "ReadOnly",
        name: "scales"
      },
      {
        binding: 3,
        size: n << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function computeDispatchInt4Opt(n, wgSize, kernelName, pipelineIndex) {
  let workgroupCount = div((n + wgSize | 0) - 1 | 0, wgSize);
  return {
    workgroupSize: [
      wgSize,
      1,
      1
    ],
    workgroupCount: [
      workgroupCount,
      1,
      1
    ],
    kernelName,
    pipelineIndex
  };
}
function genInt4MatMulTiledKernel(m, k, n, groupSize) {
  let numGroups = div((k + groupSize | 0) - 1 | 0, groupSize);
  let packedK = (k + 7 | 0) / 8 | 0;
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
const TILE = ` + 16 .toString() + `u;
const GROUP_SIZE = ` + groupSize.toString() + `u;
const NUM_GROUPS = ` + numGroups.toString() + `u;
const PACKED_K = ` + packedK.toString() + `u;

var<workgroup> tileA: array<f32, ` + 256 .toString() + `>;
var<workgroup> tileB: array<f32, ` + 256 .toString() + `>;

fn dequantize(col: u32, k_idx: u32) -> f32 {
  let packed_idx = k_idx / 8u;
  let sub_idx = k_idx % 8u;
  let packed = b_packed[col * PACKED_K + packed_idx];
  let shift = sub_idx * 4u;
  let int4_val = (packed >> shift) & 0xFu;
  let group_idx = k_idx / GROUP_SIZE;
  let scale = scales[col * NUM_GROUPS + group_idx];
  return (f32(int4_val) - 8.0) * scale;
}

@compute @workgroup_size(` + 16 .toString() + `, ` + 16 .toString() + `)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  
  var sum = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;
  
  for (var t = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A (activations - FP32)
    let aRow = wid.y * TILE + localRow;
    let aCol = t * TILE + localCol;
    if (aRow < M && aCol < K) {
      tileA[localRow * TILE + localCol] = a[aRow * K + aCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    
    // Load tile of B (dequantize INT4 on the fly)
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = dequantize(bCol, bRow);
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    
    workgroupBarrier();
  }
  
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}`;
  return {
    name: "matmul_int4_tiled_" + m.toString() + "x" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (m * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 1,
        size: (n * packedK | 0) << 2,
        usage: "ReadOnly",
        name: "b_packed"
      },
      {
        binding: 2,
        size: (n * numGroups | 0) << 2,
        usage: "ReadOnly",
        name: "scales"
      },
      {
        binding: 3,
        size: (m * n | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genMatMulKernel(m, k, n) {
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
const TILE = ` + 16 .toString() + `u;

var<workgroup> tileA: array<f32, ` + 256 .toString() + `>;
var<workgroup> tileB: array<f32, ` + 256 .toString() + `>;

@compute @workgroup_size(` + 16 .toString() + `, ` + 16 .toString() + `)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  
  var sum = 0.0;
  
  let numTiles = (K + TILE - 1u) / TILE;
  
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let tiledRow = wid.y * TILE + localRow;
    let tiledCol = t * TILE + localCol;
    
    if (tiledRow < M && tiledCol < K) {
      tileA[localRow * TILE + localCol] = a[tiledRow * K + tiledCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = b[bRow * N + bCol];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    
    workgroupBarrier();
  }
  
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}`;
  return {
    name: "matmul_" + m.toString() + "x" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (m * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 1,
        size: (k * n | 0) << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 2,
        size: (m * n | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genBatchedMatMulKernel(batchShapeA, batchShapeB, batchShapeOut, m, k, n) {
  let batchSizeA = reduce(batchShapeA, 1, (a, b) => a * b | 0);
  let batchSizeB = reduce(batchShapeB, 1, (a, b) => a * b | 0);
  let batchSizeOut = reduce(batchShapeOut, 1, (a, b) => a * b | 0);
  let batchRank = batchShapeOut.length;
  let outBatchStrides = fromInitializer(batchRank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < batchRank; ++j) {
      stride = stride * getOr(batchShapeOut[j], 1) | 0;
    }
    return stride;
  });
  let rankA = batchShapeA.length;
  let aBatchStrides = fromInitializer(batchRank, (i) => {
    let aIdx = i - (batchRank - rankA | 0) | 0;
    if (aIdx < 0) {
      return 0;
    }
    let aDim = getOr(batchShapeA[aIdx], 1);
    let outDim = getOr(batchShapeOut[i], 1);
    if (aDim === 1 && outDim > 1) {
      return 0;
    }
    let stride = 1;
    for (let j = aIdx + 1 | 0; j < rankA; ++j) {
      stride = stride * getOr(batchShapeA[j], 1) | 0;
    }
    return stride;
  });
  let rankB = batchShapeB.length;
  let bBatchStrides = fromInitializer(batchRank, (i) => {
    let bIdx = i - (batchRank - rankB | 0) | 0;
    if (bIdx < 0) {
      return 0;
    }
    let bDim = getOr(batchShapeB[bIdx], 1);
    let outDim = getOr(batchShapeOut[i], 1);
    if (bDim === 1 && outDim > 1) {
      return 0;
    }
    let stride = 1;
    for (let j = bIdx + 1 | 0; j < rankB; ++j) {
      stride = stride * getOr(batchShapeB[j], 1) | 0;
    }
    return stride;
  });
  let outBatchStridesStr = outBatchStrides.map((d) => d.toString()).join(", ");
  let aBatchStridesStr = aBatchStrides.map((d) => d.toString()).join(", ");
  let bBatchStridesStr = bBatchStrides.map((d) => d.toString()).join(", ");
  let totalOutput = (batchSizeOut * m | 0) * n | 0;
  let wgsl = batchSizeOut === 1 ? `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
const TILE = ` + 16 .toString() + `u;
var<workgroup> tileA: array<f32, ` + 256 .toString() + `>;
var<workgroup> tileB: array<f32, ` + 256 .toString() + `>;
@compute @workgroup_size(` + 16 .toString() + `, ` + 16 .toString() + `)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  var sum = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let tiledRow = wid.y * TILE + localRow;
    let tiledCol = t * TILE + localCol;
    if (tiledRow < M && tiledCol < K) {
      tileA[localRow * TILE + localCol] = a[tiledRow * K + tiledCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = b[bRow * N + bCol];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    workgroupBarrier();
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    workgroupBarrier();
  }
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}` : `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
const BATCH_OUT = ` + batchSizeOut.toString() + `u;
const BATCH_RANK = ` + batchRank.toString() + `u;
const OUT_BATCH_STRIDES = array<u32, ` + max(batchRank, 1).toString() + `>(` + (batchRank > 0 ? outBatchStridesStr : "0") + `);
const A_BATCH_STRIDES = array<u32, ` + max(batchRank, 1).toString() + `>(` + (batchRank > 0 ? aBatchStridesStr : "0") + `);
const B_BATCH_STRIDES = array<u32, ` + max(batchRank, 1).toString() + `>(` + (batchRank > 0 ? bBatchStridesStr : "0") + `);
@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ` + totalOutput.toString() + `u) { return; }
  
  // Decompose idx into batch_idx, row, col
  let matrix_size = M * N;
  let batch_idx = idx / matrix_size;
  let matrix_idx = idx % matrix_size;
  let row = matrix_idx / N;
  let col = matrix_idx % N;
  
  // Compute batch indices for A and B using broadcasting strides
  var a_batch_idx = 0u;
  var b_batch_idx = 0u;
  var remaining = batch_idx;
  
  for (var d = 0u; d < BATCH_RANK; d = d + 1u) {
    let coord = remaining / OUT_BATCH_STRIDES[d];
    remaining = remaining % OUT_BATCH_STRIDES[d];
    a_batch_idx = a_batch_idx + coord * A_BATCH_STRIDES[d];
    b_batch_idx = b_batch_idx + coord * B_BATCH_STRIDES[d];
  }
  
  // Compute matmul for this element
  var sum = 0.0;
  let a_base = a_batch_idx * M * K;
  let b_base = b_batch_idx * K * N;
  
  for (var i = 0u; i < K; i = i + 1u) {
    sum = sum + a[a_base + row * K + i] * b[b_base + i * N + col];
  }
  
  output[idx] = sum;
}`;
  return {
    name: "batched_matmul_" + batchSizeOut.toString() + "_" + m.toString() + "x" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: ((batchSizeA * m | 0) * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 1,
        size: ((batchSizeB * k | 0) * n | 0) << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 2,
        size: ((batchSizeOut * m | 0) * n | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genConv2DKernel(batch, inH, inW, inC, outH, outW, outC, kH, kW, strideH, strideW, padH, padW) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const IN_H = ` + inH.toString() + `u;
const IN_W = ` + inW.toString() + `u;
const IN_C = ` + inC.toString() + `u;
const OUT_H = ` + outH.toString() + `u;
const OUT_W = ` + outW.toString() + `u;
const OUT_C = ` + outC.toString() + `u;
const K_H = ` + kH.toString() + `u;
const K_W = ` + kW.toString() + `u;
const STRIDE_H = ` + strideH.toString() + `u;
const STRIDE_W = ` + strideW.toString() + `u;
const PAD_H = ` + padH.toString() + `u;
const PAD_W = ` + padW.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_H * OUT_W * OUT_C;
  if (idx >= totalOut) { return; }
  
  let oc = idx % OUT_C;
  let tmp1 = idx / OUT_C;
  let ow = tmp1 % OUT_W;
  let tmp2 = tmp1 / OUT_W;
  let oh = tmp2 % OUT_H;
  let b = tmp2 / OUT_H;
  
  var sum = bias[oc];
  
  for (var ic = 0u; ic < IN_C; ic = ic + 1u) {
    for (var kh = 0u; kh < K_H; kh = kh + 1u) {
      for (var kw = 0u; kw < K_W; kw = kw + 1u) {
        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw = ow * STRIDE_W + kw - PAD_W;
        
        if (ih < IN_H && iw < IN_W) {
          let inIdx = b * IN_H * IN_W * IN_C + ih * IN_W * IN_C + iw * IN_C + ic;
          let wIdx = oc * K_H * K_W * IN_C + kh * K_W * IN_C + kw * IN_C + ic;
          sum = sum + input[inIdx] * weight[wIdx];
        }
      }
    }
  }
  
  output[idx] = sum;
}`;
  return {
    name: "conv2d_" + outH.toString() + "x" + outW.toString() + "x" + outC.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (((batch * inH | 0) * inW | 0) * inC | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (((outC * kH | 0) * kW | 0) * inC | 0) << 2,
        usage: "ReadOnly",
        name: "weight"
      },
      {
        binding: 2,
        size: outC << 2,
        usage: "ReadOnly",
        name: "bias"
      },
      {
        binding: 3,
        size: (((batch * outH | 0) * outW | 0) * outC | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genDenseKernel(batchSize, inFeatures, outFeatures) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ` + batchSize.toString() + `u;
const IN_F = ` + inFeatures.toString() + `u;
const OUT_F = ` + outFeatures.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * OUT_F) { return; }
  
  let b = idx / OUT_F;
  let o = idx % OUT_F;
  
  var sum = bias[o];
  for (var i = 0u; i < IN_F; i = i + 1u) {
    sum = sum + input[b * IN_F + i] * weight[o * IN_F + i];
  }
  
  output[idx] = sum;
}`;
  return {
    name: "dense_" + batchSize.toString() + "x" + inFeatures.toString() + "x" + outFeatures.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (batchSize * inFeatures | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (outFeatures * inFeatures | 0) << 2,
        usage: "ReadOnly",
        name: "weight"
      },
      {
        binding: 2,
        size: outFeatures << 2,
        usage: "ReadOnly",
        name: "bias"
      },
      {
        binding: 3,
        size: (batchSize * outFeatures | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSoftmaxKernel(outerSize, axisSize) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const OUTER = ` + outerSize.toString() + `u;
const AXIS = ` + axisSize.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outerIdx = gid.x;
  if (outerIdx >= OUTER) { return; }
  
  let baseIdx = outerIdx * AXIS;
  
  // Find max for numerical stability
  var maxVal = input[baseIdx];
  for (var i = 1u; i < AXIS; i = i + 1u) {
    maxVal = max(maxVal, input[baseIdx + i]);
  }
  
  // Compute exp and sum
  var sum = 0.0;
  for (var i = 0u; i < AXIS; i = i + 1u) {
    let e = exp(input[baseIdx + i] - maxVal);
    output[baseIdx + i] = e;
    sum = sum + e;
  }
  
  // Normalize
  for (var i = 0u; i < AXIS; i = i + 1u) {
    output[baseIdx + i] = output[baseIdx + i] / sum;
  }
}`;
  return {
    name: "softmax_" + outerSize.toString() + "x" + axisSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (outerSize * axisSize | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (outerSize * axisSize | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genTransposeKernel(inputShape, perm) {
  let rank = inputShape.length;
  if (rank !== perm.length) {
    return;
  }
  let totalSize = numElements(inputShape);
  let outputShape = perm.map((p) => getOr(inputShape[p], 1));
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let outputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(outputShape[j], 1) | 0;
    }
    return stride;
  });
  let shapeStr = inputShape.map((d) => d.toString()).join(", ");
  let permStr = perm.map((d) => d.toString()).join(", ");
  let inStrideStr = inputStrides.map((d) => d.toString()).join(", ");
  let outStrideStr = outputStrides.map((d) => d.toString()).join(", ");
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const RANK = ` + rank.toString() + `u;
const TOTAL = ` + totalSize.toString() + `u;
const SHAPE = array<u32, ` + rank.toString() + `>(` + shapeStr + `);
const PERM = array<u32, ` + rank.toString() + `>(` + permStr + `);
const IN_STRIDES = array<u32, ` + rank.toString() + `>(` + inStrideStr + `);
const OUT_STRIDES = array<u32, ` + rank.toString() + `>(` + outStrideStr + `);

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
  if (outIdx >= TOTAL) { return; }
  
  // Convert output index to coordinates
  var coords: array<u32, ` + rank.toString() + `>;
  var remaining = outIdx;
  for (var i = 0u; i < RANK; i = i + 1u) {
    coords[i] = remaining / OUT_STRIDES[i];
    remaining = remaining % OUT_STRIDES[i];
  }
  
  // Apply inverse permutation to get input coords
  var inIdx = 0u;
  for (var i = 0u; i < RANK; i = i + 1u) {
    inIdx = inIdx + coords[i] * IN_STRIDES[PERM[i]];
  }
  
  output[outIdx] = input[inIdx];
}`;
  return {
    name: "transpose_" + totalSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: totalSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: totalSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genReshapeKernel(size3) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ` + size3.toString() + `u) { return; }
  output[idx] = input[idx];
}`;
  return {
    name: "reshape_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genUnaryKernel(op, size3) {
  return map(unaryExpr(op, "input0[idx]"), (expr) => {
    let wgsl = shaderHeader(2) + `

` + mainSignature + `
  if (idx >= ` + size3.toString() + `u) { return; }
  output[idx] = ` + expr + `;
` + mainEnd;
    return {
      name: "unary_" + size3.toString(),
      wgsl,
      bindings: [
        {
          binding: 0,
          size: size3 << 2,
          usage: "ReadOnly",
          name: "input0"
        },
        {
          binding: 1,
          size: size3 << 2,
          usage: "ReadWrite",
          name: "output"
        }
      ]
    };
  });
}
function genDropoutKernel(size3, rate) {
  let scale2 = 1 / (1 - rate);
  let threshold = 1 - rate;
  let wgsl = storageBuffer(0, "input0", "ReadOnly") + `
` + storageBuffer(1, "seed", "ReadOnly") + `
` + storageBuffer(2, "output", "ReadWrite") + `

// PCG-style hash for deterministic pseudo-random per element
fn pcg_hash(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_f32(global_idx: u32, seed_val: u32) -> f32 {
  let h = pcg_hash(global_idx ^ seed_val);
  return f32(h) / 4294967295.0;
}

` + mainSignature + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let seed_val = bitcast<u32>(seed[0]);
  let r = rand_f32(idx, seed_val);
  if (r < ` + threshold.toString() + `) {
    output[idx] = input0[idx] * ` + scale2.toString() + `;
  } else {
    output[idx] = 0.0;
  }
` + mainEnd;
  return {
    name: "dropout_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input0"
      },
      {
        binding: 1,
        size: 4,
        usage: "ReadOnly",
        name: "seed"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genBinaryBroadcastKernel(op, inputShape0, inputShape1, outputShape) {
  let outSize = numElements(outputShape);
  let inSize0 = numElements(inputShape0);
  let inSize1 = numElements(inputShape1);
  let outRank = outputShape.length;
  let inRank0 = inputShape0.length;
  let inRank1 = inputShape1.length;
  let outStrides = fromInitializer(outRank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < outRank; ++j) {
      stride = stride * getOr(outputShape[j], 1) | 0;
    }
    return stride;
  });
  let inStrides0 = fromInitializer(outRank, (i) => {
    let inIdx = i - (outRank - inRank0 | 0) | 0;
    if (inIdx < 0) {
      return 0;
    }
    let inDim = getOr(inputShape0[inIdx], 1);
    let outDim = getOr(outputShape[i], 1);
    if (inDim === 1 && outDim > 1) {
      return 0;
    }
    let stride = 1;
    for (let j = inIdx + 1 | 0; j < inRank0; ++j) {
      stride = stride * getOr(inputShape0[j], 1) | 0;
    }
    return stride;
  });
  let inStrides1 = fromInitializer(outRank, (i) => {
    let inIdx = i - (outRank - inRank1 | 0) | 0;
    if (inIdx < 0) {
      return 0;
    }
    let inDim = getOr(inputShape1[inIdx], 1);
    let outDim = getOr(outputShape[i], 1);
    if (inDim === 1 && outDim > 1) {
      return 0;
    }
    let stride = 1;
    for (let j = inIdx + 1 | 0; j < inRank1; ++j) {
      stride = stride * getOr(inputShape1[j], 1) | 0;
    }
    return stride;
  });
  let outShapeStr = outputShape.map((d) => d.toString()).join(", ");
  let outStridesStr = outStrides.map((d) => d.toString()).join(", ");
  let inStrides0Str = inStrides0.map((d) => d.toString()).join(", ");
  let inStrides1Str = inStrides1.map((d) => d.toString()).join(", ");
  return map(binaryExpr(op, "val0", "val1"), (expr) => {
    let wgsl = inSize0 === outSize && inSize1 === outSize ? storageBuffer(0, "input0", "ReadOnly") + `
` + storageBuffer(1, "input1", "ReadOnly") + `
` + storageBuffer(2, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outSize.toString() + `u) { return; }
  let val0 = input0[idx];
  let val1 = input1[idx];
  output[idx] = ` + expr + `;
` + mainEnd : storageBuffer(0, "input0", "ReadOnly") + `
` + storageBuffer(1, "input1", "ReadOnly") + `
` + storageBuffer(2, "output", "ReadWrite") + `
const OUT_RANK = ` + outRank.toString() + `u;
const OUT_SHAPE = array<u32, ` + outRank.toString() + `>(` + outShapeStr + `);
const OUT_STRIDES = array<u32, ` + outRank.toString() + `>(` + outStridesStr + `);
const IN_STRIDES0 = array<u32, ` + outRank.toString() + `>(` + inStrides0Str + `);
const IN_STRIDES1 = array<u32, ` + outRank.toString() + `>(` + inStrides1Str + `);
` + mainSignature + `
  if (idx >= ` + outSize.toString() + `u) { return; }
  
  // Convert flat index to coordinates
  var remaining = idx;
  var in_idx0 = 0u;
  var in_idx1 = 0u;
  
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    let coord = remaining / OUT_STRIDES[d];
    remaining = remaining % OUT_STRIDES[d];
    in_idx0 = in_idx0 + coord * IN_STRIDES0[d];
    in_idx1 = in_idx1 + coord * IN_STRIDES1[d];
  }
  
  let val0 = input0[in_idx0];
  let val1 = input1[in_idx1];
  output[idx] = ` + expr + `;
` + mainEnd;
    return {
      name: "binary_broadcast_" + outSize.toString(),
      wgsl,
      bindings: [
        {
          binding: 0,
          size: inSize0 << 2,
          usage: "ReadOnly",
          name: "input0"
        },
        {
          binding: 1,
          size: inSize1 << 2,
          usage: "ReadOnly",
          name: "input1"
        },
        {
          binding: 2,
          size: outSize << 2,
          usage: "ReadWrite",
          name: "output"
        }
      ]
    };
  });
}
function computeDispatch(totalElements, kernelName, pipelineIndex) {
  let workgroupCount = ((totalElements + 256 | 0) - 1 | 0) / 256 | 0;
  return {
    workgroupSize: [
      256,
      1,
      1
    ],
    workgroupCount: [
      workgroupCount,
      1,
      1
    ],
    kernelName,
    pipelineIndex
  };
}
function computeDispatch2D(x, y, tileSize, kernelName, pipelineIndex) {
  let countX = div((x + tileSize | 0) - 1 | 0, tileSize);
  let countY = div((y + tileSize | 0) - 1 | 0, tileSize);
  return {
    workgroupSize: [
      tileSize,
      tileSize,
      1
    ],
    workgroupCount: [
      countX,
      countY,
      1
    ],
    kernelName,
    pipelineIndex
  };
}
function genPool2DKernel(poolType, batch, inH, inW, channels, outH, outW, kH, kW, strideH, strideW, padH, padW) {
  let identity = poolType === "max" ? "-3.402823e+38" : "0.0";
  let accumulate = poolType === "max" ? "acc = max(acc, val);" : "acc = acc + val;";
  let finalize = poolType === "max" ? "acc" : "acc / f32(K_H * K_W)";
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const IN_H = ` + inH.toString() + `u;
const IN_W = ` + inW.toString() + `u;
const CHANNELS = ` + channels.toString() + `u;
const OUT_H = ` + outH.toString() + `u;
const OUT_W = ` + outW.toString() + `u;
const K_H = ` + kH.toString() + `u;
const K_W = ` + kW.toString() + `u;
const STRIDE_H = ` + strideH.toString() + `u;
const STRIDE_W = ` + strideW.toString() + `u;
const PAD_H = ` + padH.toString() + `u;
const PAD_W = ` + padW.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_H * OUT_W * CHANNELS;
  if (idx >= totalOut) { return; }

  let c = idx % CHANNELS;
  let tmp1 = idx / CHANNELS;
  let ow = tmp1 % OUT_W;
  let tmp2 = tmp1 / OUT_W;
  let oh = tmp2 % OUT_H;
  let b = tmp2 / OUT_H;

  var acc = ` + identity + `;

  for (var kh = 0u; kh < K_H; kh = kh + 1u) {
    for (var kw = 0u; kw < K_W; kw = kw + 1u) {
      let ih = oh * STRIDE_H + kh - PAD_H;
      let iw = ow * STRIDE_W + kw - PAD_W;

      if (ih < IN_H && iw < IN_W) {
        let inIdx = b * IN_H * IN_W * CHANNELS + ih * IN_W * CHANNELS + iw * CHANNELS + c;
        let val = input[inIdx];
        ` + accumulate + `
      }
    }
  }

  output[idx] = ` + finalize + `;
}`;
  let outSize = ((batch * outH | 0) * outW | 0) * channels | 0;
  return {
    name: poolType + "pool2d_" + outH.toString() + "x" + outW.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (((batch * inH | 0) * inW | 0) * channels | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genBatchNormKernel(batch, height, width, channels) {
  let size3 = ((batch * height | 0) * width | 0) * channels | 0;
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const SIZE = ` + size3.toString() + `u;
const CHANNELS = ` + channels.toString() + `u;
const EPSILON = 1e-5;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }

  let c = idx % CHANNELS;
  let x = input[idx];
  let normalized = (x - mean[c]) / sqrt(variance[c] + EPSILON);
  output[idx] = gamma[c] * normalized + beta[c];
}`;
  return {
    name: "batchnorm_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: channels << 2,
        usage: "ReadOnly",
        name: "gamma"
      },
      {
        binding: 2,
        size: channels << 2,
        usage: "ReadOnly",
        name: "beta"
      },
      {
        binding: 3,
        size: channels << 2,
        usage: "ReadOnly",
        name: "mean"
      },
      {
        binding: 4,
        size: channels << 2,
        usage: "ReadOnly",
        name: "variance"
      },
      {
        binding: 5,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genConv1DKernel(batch, inLen, inC, outLen, outC, kernel, stride, pad2) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const IN_LEN = ` + inLen.toString() + `u;
const IN_C = ` + inC.toString() + `u;
const OUT_LEN = ` + outLen.toString() + `u;
const OUT_C = ` + outC.toString() + `u;
const K = ` + kernel.toString() + `u;
const STRIDE = ` + stride.toString() + `u;
const PAD = ` + pad2.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_LEN * OUT_C;
  if (idx >= totalOut) { return; }

  let oc = idx % OUT_C;
  let tmp = idx / OUT_C;
  let ol = tmp % OUT_LEN;
  let b = tmp / OUT_LEN;

  var sum = bias[oc];

  for (var ic = 0u; ic < IN_C; ic = ic + 1u) {
    for (var k = 0u; k < K; k = k + 1u) {
      let il = ol * STRIDE + k - PAD;
      if (il < IN_LEN) {
        let inIdx = b * IN_LEN * IN_C + il * IN_C + ic;
        let wIdx = oc * K * IN_C + k * IN_C + ic;
        sum = sum + input[inIdx] * weight[wIdx];
      }
    }
  }

  output[idx] = sum;
}`;
  return {
    name: "conv1d_" + outLen.toString() + "x" + outC.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: ((batch * inLen | 0) * inC | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: ((outC * kernel | 0) * inC | 0) << 2,
        usage: "ReadOnly",
        name: "weight"
      },
      {
        binding: 2,
        size: outC << 2,
        usage: "ReadOnly",
        name: "bias"
      },
      {
        binding: 3,
        size: ((batch * outLen | 0) * outC | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genGlobalPoolKernel(poolType, batch, height, width, channels) {
  let identity = poolType === "max" ? "-3.402823e+38" : "0.0";
  let accumulate = poolType === "max" ? "acc = max(acc, val);" : "acc = acc + val;";
  let finalize = poolType === "max" ? "acc" : "acc / f32(H * W)";
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const H = ` + height.toString() + `u;
const W = ` + width.toString() + `u;
const C = ` + channels.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * C) { return; }

  let c = idx % C;
  let b = idx / C;

  var acc = ` + identity + `;

  for (var h = 0u; h < H; h = h + 1u) {
    for (var w = 0u; w < W; w = w + 1u) {
      let inIdx = b * H * W * C + h * W * C + w * C + c;
      let val = input[inIdx];
      ` + accumulate + `
    }
  }

  // Output shape is [batch, 1, 1, channels]
  let outIdx = b * C + c;
  output[outIdx] = ` + finalize + `;
}`;
  return {
    name: poolType + "globalpool_" + batch.toString() + "x" + channels.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (((batch * height | 0) * width | 0) * channels | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (batch * channels | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genLayerNormKernel(outerSize, normSize, epsilon) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const OUTER = ` + outerSize.toString() + `u;
const NORM = ` + normSize.toString() + `u;
const EPSILON = ` + epsilon.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outerIdx = gid.x;
  if (outerIdx >= OUTER) { return; }

  let baseIdx = outerIdx * NORM;

  // Compute mean
  var sum = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    sum = sum + input[baseIdx + i];
  }
  let mean = sum / f32(NORM);

  // Compute variance
  var varSum = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let diff = input[baseIdx + i] - mean;
    varSum = varSum + diff * diff;
  }
  let variance = varSum / f32(NORM);
  let invStd = 1.0 / sqrt(variance + EPSILON);

  // Normalize and scale
  for (var i = 0u; i < NORM; i = i + 1u) {
    let normalized = (input[baseIdx + i] - mean) * invStd;
    output[baseIdx + i] = gamma[i] * normalized + beta[i];
  }
}`;
  return {
    name: "layernorm_" + outerSize.toString() + "x" + normSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (outerSize * normSize | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: normSize << 2,
        usage: "ReadOnly",
        name: "gamma"
      },
      {
        binding: 2,
        size: normSize << 2,
        usage: "ReadOnly",
        name: "beta"
      },
      {
        binding: 3,
        size: (outerSize * normSize | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genRMSNormKernel(outerSize, normSize, epsilon) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const OUTER = ` + outerSize.toString() + `u;
const NORM = ` + normSize.toString() + `u;
const EPSILON = ` + epsilon.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= OUTER) { return; }

  let baseIdx = row * NORM;

  // Compute mean of squares
  var sumSq = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let x = input[baseIdx + i];
    sumSq = sumSq + x * x;
  }
  let rms = sqrt(sumSq / f32(NORM) + EPSILON);
  let scale = 1.0 / rms;

  // Normalize and apply weight
  for (var i = 0u; i < NORM; i = i + 1u) {
    output[baseIdx + i] = input[baseIdx + i] * scale * weight[i];
  }
}`;
  return {
    name: "rmsnorm_" + outerSize.toString() + "x" + normSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (outerSize * normSize | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: normSize << 2,
        usage: "ReadOnly",
        name: "weight"
      },
      {
        binding: 2,
        size: (outerSize * normSize | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genRoPEKernel(numQHeads, numKVHeads, headDim, ropeTheta) {
  let totalQDim = numQHeads * headDim | 0;
  let totalKDim = numKVHeads * headDim | 0;
  let halfDim = headDim / 2 | 0;
  Math.max(numQHeads, numKVHeads) * halfDim | 0;
  let wgsl = `@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> position: array<u32>;

const NUM_Q_HEADS = ` + numQHeads.toString() + `u;
const NUM_KV_HEADS = ` + numKVHeads.toString() + `u;
const HEAD_DIM = ` + headDim.toString() + `u;
const HALF_DIM = ` + halfDim.toString() + `u;
const ROPE_THETA = ` + ropeTheta.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pos = f32(position[0]);
  
  // Process Q heads
  if (idx < NUM_Q_HEADS * HALF_DIM) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(2u * pair) / f32(HEAD_DIM));
    let angle = pos * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);
    
    let base_idx = head * HEAD_DIM + pair * 2u;
    let q0 = q[base_idx];
    let q1 = q[base_idx + 1u];
    
    q[base_idx] = q0 * cos_val - q1 * sin_val;
    q[base_idx + 1u] = q0 * sin_val + q1 * cos_val;
  }
  
  // Process K heads
  if (idx < NUM_KV_HEADS * HALF_DIM) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(2u * pair) / f32(HEAD_DIM));
    let angle = pos * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);
    
    let base_idx = head * HEAD_DIM + pair * 2u;
    let k0 = k[base_idx];
    let k1 = k[base_idx + 1u];
    
    k[base_idx] = k0 * cos_val - k1 * sin_val;
    k[base_idx + 1u] = k0 * sin_val + k1 * cos_val;
  }
}`;
  return {
    name: "rope_" + numQHeads.toString() + "x" + numKVHeads.toString() + "x" + headDim.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: totalQDim << 2,
        usage: "ReadWrite",
        name: "q"
      },
      {
        binding: 1,
        size: totalKDim << 2,
        usage: "ReadWrite",
        name: "k"
      },
      {
        binding: 2,
        size: 4,
        usage: "ReadOnly",
        name: "position"
      }
    ]
  };
}
function genGQAAttentionKernel(numQHeads, numKVHeads, headDim, maxSeqLen) {
  let headsPerGroup = div(numQHeads, numKVHeads);
  let scale2 = 1 / Math.sqrt(headDim);
  let wgsl = `@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> seq_len: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const NUM_Q_HEADS = ` + numQHeads.toString() + `u;
const NUM_KV_HEADS = ` + numKVHeads.toString() + `u;
const HEADS_PER_GROUP = ` + headsPerGroup.toString() + `u;
const HEAD_DIM = ` + headDim.toString() + `u;
const MAX_SEQ_LEN = ` + maxSeqLen.toString() + `u;
const SCALE = ` + scale2.toString() + `;
const WG_SIZE = ` + 256 .toString() + `u;

var<workgroup> wg_q: array<f32, ` + headDim.toString() + `>;
var<workgroup> wg_scores: array<f32, ` + maxSeqLen.toString() + `>;
var<workgroup> wg_reduce: array<f32, ` + 256 .toString() + `>;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let q_head = wgid.x;
  let tid = lid.x;
  let kv_head = q_head / HEADS_PER_GROUP;
  let cur_seq_len = seq_len[0];
  
  // Load Q into shared memory
  let q_base = q_head * HEAD_DIM;
  for (var d = tid; d < HEAD_DIM; d = d + WG_SIZE) {
    wg_q[d] = q[q_base + d];
  }
  workgroupBarrier();
  
  // Phase 1: Compute scores Q @ K^T
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    var score = 0.0;
    let k_base = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
    
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
      score = score + wg_q[d] * k_cache[k_base + d];
    }
    wg_scores[pos] = score * SCALE;
  }
  workgroupBarrier();
  
  // Phase 2: Softmax - find max
  var local_max = -1e30;
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    local_max = max(local_max, wg_scores[pos]);
  }
  wg_reduce[tid] = local_max;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_reduce[tid] = max(wg_reduce[tid], wg_reduce[tid + stride]);
    }
    workgroupBarrier();
  }
  let max_val = wg_reduce[0];
  
  // Phase 3: Softmax - exp and sum
  var local_sum = 0.0;
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    let e = exp(wg_scores[pos] - max_val);
    wg_scores[pos] = e;
    local_sum = local_sum + e;
  }
  wg_reduce[tid] = local_sum;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + stride];
    }
    workgroupBarrier();
  }
  let sum_val = wg_reduce[0];
  
  // Phase 4: Normalize probabilities
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    wg_scores[pos] = wg_scores[pos] / sum_val;
  }
  workgroupBarrier();
  
  // Phase 5: Output = probs @ V
  for (var d = tid; d < HEAD_DIM; d = d + WG_SIZE) {
    var out_val = 0.0;
    for (var pos = 0u; pos < cur_seq_len; pos = pos + 1u) {
      let v_idx = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d;
      out_val = out_val + wg_scores[pos] * v_cache[v_idx];
    }
    output[q_head * HEAD_DIM + d] = out_val;
  }
}`;
  return {
    name: "gqa_attention_" + numQHeads.toString() + "x" + numKVHeads.toString() + "x" + headDim.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (numQHeads * headDim | 0) << 2,
        usage: "ReadOnly",
        name: "q"
      },
      {
        binding: 1,
        size: ((maxSeqLen * numKVHeads | 0) * headDim | 0) << 2,
        usage: "ReadOnly",
        name: "k_cache"
      },
      {
        binding: 2,
        size: ((maxSeqLen * numKVHeads | 0) * headDim | 0) << 2,
        usage: "ReadOnly",
        name: "v_cache"
      },
      {
        binding: 3,
        size: 4,
        usage: "ReadOnly",
        name: "seq_len"
      },
      {
        binding: 4,
        size: (numQHeads * headDim | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genAttentionKernel(batch, seqLen, dim) {
  let scale2 = 1 / Math.sqrt(dim);
  let wgsl = `@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const SEQ = ` + seqLen.toString() + `u;
const DIM = ` + dim.toString() + `u;
const SCALE = ` + scale2.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * SEQ * DIM) { return; }

  let d = idx % DIM;
  let tmp = idx / DIM;
  let qPos = tmp % SEQ;
  let b = tmp / SEQ;

  let qBase = b * SEQ * DIM + qPos * DIM;

  // Compute attention scores and find max for stability
  var maxScore = -3.402823e+38;
  for (var kPos = 0u; kPos < SEQ; kPos = kPos + 1u) {
    let kBase = b * SEQ * DIM + kPos * DIM;
    var score = 0.0;
    for (var i = 0u; i < DIM; i = i + 1u) {
      score = score + query[qBase + i] * key[kBase + i];
    }
    score = score * SCALE;
    maxScore = max(maxScore, score);
  }

  // Compute softmax and weighted sum
  var expSum = 0.0;
  var result = 0.0;

  for (var kPos = 0u; kPos < SEQ; kPos = kPos + 1u) {
    let kBase = b * SEQ * DIM + kPos * DIM;
    var score = 0.0;
    for (var i = 0u; i < DIM; i = i + 1u) {
      score = score + query[qBase + i] * key[kBase + i];
    }
    score = score * SCALE;
    let expScore = exp(score - maxScore);
    expSum = expSum + expScore;

    let vBase = b * SEQ * DIM + kPos * DIM;
    result = result + expScore * value[vBase + d];
  }

  output[idx] = result / expSum;
}`;
  return {
    name: "attention_" + batch.toString() + "x" + seqLen.toString() + "x" + dim.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: ((batch * seqLen | 0) * dim | 0) << 2,
        usage: "ReadOnly",
        name: "query"
      },
      {
        binding: 1,
        size: ((batch * seqLen | 0) * dim | 0) << 2,
        usage: "ReadOnly",
        name: "key"
      },
      {
        binding: 2,
        size: ((batch * seqLen | 0) * dim | 0) << 2,
        usage: "ReadOnly",
        name: "value"
      },
      {
        binding: 3,
        size: ((batch * seqLen | 0) * dim | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genEmbeddingKernel(batchSeq, vocabSize, embDim) {
  let wgsl = `@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> embeddings: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const BATCH_SEQ = ` + batchSeq.toString() + `u;
const VOCAB = ` + vocabSize.toString() + `u;
const DIM = ` + embDim.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH_SEQ * DIM) { return; }

  let d = idx % DIM;
  let pos = idx / DIM;

  let tokenId = indices[pos];
  output[idx] = embeddings[tokenId * DIM + d];
}`;
  return {
    name: "embedding_" + batchSeq.toString() + "x" + embDim.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: batchSeq << 2,
        usage: "ReadOnly",
        name: "indices"
      },
      {
        binding: 1,
        size: (vocabSize * embDim | 0) << 2,
        usage: "ReadOnly",
        name: "embeddings"
      },
      {
        binding: 2,
        size: (batchSeq * embDim | 0) << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genConcatKernel(inputShapes, axis) {
  let numInputs = inputShapes.length;
  if (numInputs === 0) {
    return;
  }
  let first = getOr(inputShapes[0], []);
  let rank = first.length;
  let normAxis = axis < 0 ? rank + axis | 0 : axis;
  let concatDim = reduce(inputShapes, 0, (acc, s) => acc + getOr(s[normAxis], 0) | 0);
  let outputShape = first.map((d, i) => {
    if (i === normAxis) {
      return concatDim;
    } else {
      return d;
    }
  });
  let outSize = numElements(outputShape);
  let offsets = fromInitializer(numInputs, (i) => {
    let offset = 0;
    for (let j = 0; j < i; ++j) {
      let s = getOr(inputShapes[j], []);
      offset = offset + getOr(s[normAxis], 0) | 0;
    }
    return offset;
  });
  offsets.map((o) => o.toString()).join(", ");
  inputShapes.map((s) => getOr(s[normAxis], 0).toString()).join(", ");
  if (numInputs !== 2) {
    return;
  }
  let size0 = numElements(getOr(inputShapes[0], []));
  let size1 = numElements(getOr(inputShapes[1], []));
  let axisDim0 = getOr(getOr(inputShapes[0], [])[normAxis], 0);
  let wgsl = `@group(0) @binding(0) var<storage, read> input0: array<f32>;
@group(0) @binding(1) var<storage, read> input1: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const OUT_SIZE = ` + outSize.toString() + `u;
const AXIS = ` + normAxis.toString() + `u;
const AXIS_DIM0 = ` + axisDim0.toString() + `u;
const CONCAT_DIM = ` + concatDim.toString() + `u;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= OUT_SIZE) { return; }
  
  // Simple copy - determine which input based on position
  if (idx < ` + size0.toString() + `u) {
    output[idx] = input0[idx];
  } else {
    output[idx] = input1[idx - ` + size0.toString() + `u];
  }
}`;
  return {
    name: "concat_" + outSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size0 << 2,
        usage: "ReadOnly",
        name: "input0"
      },
      {
        binding: 1,
        size: size1 << 2,
        usage: "ReadOnly",
        name: "input1"
      },
      {
        binding: 2,
        size: outSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genClipKernel(size3, minVal, maxVal) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE = ` + size3.toString() + `u;
const MIN_VAL = ` + minVal.toString() + `;
const MAX_VAL = ` + maxVal.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  output[idx] = clamp(input[idx], MIN_VAL, MAX_VAL);
}`;
  return {
    name: "clip_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genWhereKernel(size3) {
  let wgsl = storageBuffer(0, "condition", "ReadOnly") + `
` + storageBuffer(1, "input_true", "ReadOnly") + `
` + storageBuffer(2, "input_false", "ReadOnly") + `
` + storageBuffer(3, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let cond = condition[idx];
  output[idx] = select(input_false[idx], input_true[idx], cond > 0.0);
` + mainEnd;
  return {
    name: "where_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "condition"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input_true"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input_false"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genGatherKernel(dataShape, indicesSize, axis) {
  let rank = dataShape.length;
  let outerSize = 1;
  let axisSize = getOr(dataShape[axis], 1);
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(dataShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(dataShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = (outer * indicesSize | 0) * inner | 0;
  let wgsl = storageBuffer(0, "data", "ReadOnly") + `
` + storageBuffer(1, "indices", "ReadOnly") + `
` + storageBuffer(2, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  
  let inner_size = ` + inner.toString() + `u;
  let indices_size = ` + indicesSize.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  
  let outer_idx = idx / (indices_size * inner_size);
  let remainder = idx % (indices_size * inner_size);
  let index_idx = remainder / inner_size;
  let inner_idx = remainder % inner_size;
  
  let gather_idx = u32(indices[index_idx]);
  let data_idx = outer_idx * (axis_size * inner_size) + gather_idx * inner_size + inner_idx;
  
  output[idx] = data[data_idx];
` + mainEnd;
  return {
    name: "gather_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(dataShape) << 2,
        usage: "ReadOnly",
        name: "data"
      },
      {
        binding: 1,
        size: indicesSize << 2,
        usage: "ReadOnly",
        name: "indices"
      },
      {
        binding: 2,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSplitKernel(inputShape, axis, splitIndex, splitSize) {
  let rank = inputShape.length;
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = (outer * splitSize | 0) * inner | 0;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  let inner_size = ` + inner.toString() + `u;
  let split_size = ` + splitSize.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  let offset = ` + splitIndex.toString() + `u;
  let outer_idx = idx / (split_size * inner_size);
  let remainder = idx % (split_size * inner_size);
  let axis_idx = remainder / inner_size;
  let inner_idx = remainder % inner_size;
  let input_idx = outer_idx * (axis_size * inner_size) + (offset + axis_idx) * inner_size + inner_idx;
  output[idx] = input[input_idx];
` + mainEnd;
  return {
    name: "split_" + splitIndex.toString() + "_" + splitSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genTopKKernel(inputShape, k, axis) {
  let rank = inputShape.length;
  let outerSize = 1;
  let axisSize = getOr(inputShape[axis], 1);
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let numSlices = outer * inner | 0;
  let outputSize = numSlices * k | 0;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output_values", "ReadWrite") + `
` + storageBuffer(2, "output_indices", "ReadWrite") + `

` + mainSignature + `
  let slice_idx = gid.x;
  if (slice_idx >= ` + numSlices.toString() + `u) { return; }
  
  let k = ` + k.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  let inner_size = ` + inner.toString() + `u;
  
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  
  // Simple selection sort for top-k (works well for small k)
  var top_vals: array<f32, ` + k.toString() + `>;
  var top_idxs: array<i32, ` + k.toString() + `>;
  
  // Initialize with very small values
  for (var i = 0u; i < k; i = i + 1u) {
    top_vals[i] = -3.402823e+38;
    top_idxs[i] = -1;
  }
  
  // Find top k
  for (var i = 0u; i < axis_size; i = i + 1u) {
    let input_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let val = input[input_idx];
    
    // Check if this value is larger than the smallest in top_vals
    if (val > top_vals[k - 1u]) {
      // Insert in sorted position
      var j = k - 1u;
      while (j > 0u && val > top_vals[j - 1u]) {
        top_vals[j] = top_vals[j - 1u];
        top_idxs[j] = top_idxs[j - 1u];
        j = j - 1u;
      }
      top_vals[j] = val;
      top_idxs[j] = i32(i);
    }
  }
  
  // Write output
  for (var i = 0u; i < k; i = i + 1u) {
    let out_idx = slice_idx * k + i;
    output_values[out_idx] = top_vals[i];
    output_indices[out_idx] = f32(top_idxs[i]);
  }
` + mainEnd;
  return {
    name: "topk_" + k.toString() + "_" + numSlices.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output_values"
      },
      {
        binding: 2,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output_indices"
      }
    ]
  };
}
function genArgMaxKernel(inputShape, axis, selectLastIndex) {
  let rank = inputShape.length;
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = outer * inner | 0;
  let cmpOp = selectLastIndex ? ">=" : ">";
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  let inner_size = ` + inner.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  let outer_idx = idx / inner_size;
  let inner_idx = idx % inner_size;
  var max_val = input[outer_idx * (axis_size * inner_size) + inner_idx];
  var max_idx = 0u;
  for (var i = 1u; i < axis_size; i = i + 1u) {
    let val = input[outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx];
    if (val ` + cmpOp + ` max_val) {
      max_val = val;
      max_idx = i;
    }
  }
  output[idx] = f32(max_idx);
` + mainEnd;
  return {
    name: "argmax_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genArgMinKernel(inputShape, axis, selectLastIndex) {
  let rank = inputShape.length;
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = outer * inner | 0;
  let cmpOp = selectLastIndex ? "<=" : "<";
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  let inner_size = ` + inner.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  let outer_idx = idx / inner_size;
  let inner_idx = idx % inner_size;
  var min_val = input[outer_idx * (axis_size * inner_size) + inner_idx];
  var min_idx = 0u;
  for (var i = 1u; i < axis_size; i = i + 1u) {
    let val = input[outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx];
    if (val ` + cmpOp + ` min_val) {
      min_val = val;
      min_idx = i;
    }
  }
  output[idx] = f32(min_idx);
` + mainEnd;
  return {
    name: "argmin_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genPadKernel(inputShape, pads, constantValue) {
  let rank = inputShape.length;
  let outputShape = fromInitializer(rank, (i) => {
    let before = getOr(pads[i], 0);
    let after = getOr(pads[rank + i | 0], 0);
    return (getOr(inputShape[i], 0) + before | 0) + after | 0;
  });
  let outputSize = numElements(outputShape);
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let outputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(outputShape[j], 1) | 0;
    }
    return stride;
  });
  let padsBefore = fromInitializer(rank, (i) => getOr(pads[i], 0));
  let coordCode = outputStrides.map((stride, i) => {
    let padBefore = getOr(padsBefore[i], 0);
    let inDim = getOr(inputShape[i], 1);
    let inStride = getOr(inputStrides[i], 1);
    return `  let coord_` + i.toString() + ` = remaining / ` + stride.toString() + `u;
  remaining = remaining % ` + stride.toString() + `u;
  let in_coord_` + i.toString() + ` = i32(coord_` + i.toString() + `) - ` + padBefore.toString() + `;
  if (in_coord_` + i.toString() + ` < 0 || in_coord_` + i.toString() + ` >= ` + inDim.toString() + `) {
    in_bounds = false;
  } else {
    input_idx = input_idx + u32(in_coord_` + i.toString() + `) * ` + inStride.toString() + `u;
  }`;
  }).join("\n");
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  var remaining = idx;
  var in_bounds = true;
  var input_idx = 0u;
` + coordCode + `
  if (in_bounds) {
    output[idx] = input[input_idx];
  } else {
    output[idx] = ` + constantValue.toString() + `;
  }
` + mainEnd;
  return {
    name: "pad_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genTileKernel(inputShape, repeats) {
  let rank = inputShape.length;
  let outputShape = inputShape.map((d, i) => d * getOr(repeats[i], 1) | 0);
  let outputSize = numElements(outputShape);
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let outputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(outputShape[j], 1) | 0;
    }
    return stride;
  });
  let coordCode = outputStrides.map((stride, i) => {
    let inDim = getOr(inputShape[i], 1);
    let inStride = getOr(inputStrides[i], 1);
    return `  let coord_` + i.toString() + ` = remaining / ` + stride.toString() + `u;
  remaining = remaining % ` + stride.toString() + `u;
  let in_coord_` + i.toString() + ` = coord_` + i.toString() + ` % ` + inDim.toString() + `u;
  input_idx = input_idx + in_coord_` + i.toString() + ` * ` + inStride.toString() + `u;`;
  }).join("\n");
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  var remaining = idx;
  var input_idx = 0u;
` + coordCode + `
  output[idx] = input[input_idx];
` + mainEnd;
  return {
    name: "tile_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSliceKernel(inputShape, starts, ends, axes, steps) {
  let rank = inputShape.length;
  let outputShape = inputShape.slice();
  axes.forEach((ax, i) => {
    let dimSize = getOr(inputShape[ax], 1);
    let start = getOr(starts[i], 0);
    let end_ = getOr(ends[i], dimSize);
    let step = getOr(steps[i], 1);
    let s = start < 0 ? dimSize + start | 0 : start;
    let e = end_ < 0 ? dimSize + end_ | 0 : end_;
    let size3 = div(((e - s | 0) + step | 0) - 1 | 0, step);
    outputShape[ax] = size3;
  });
  let outputSize = numElements(outputShape);
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let outputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(outputShape[j], 1) | 0;
    }
    return stride;
  });
  let allStarts = make(rank, 0);
  let allSteps = make(rank, 1);
  axes.forEach((ax, i) => {
    let dimSize = getOr(inputShape[ax], 1);
    let start = getOr(starts[i], 0);
    let s = start < 0 ? dimSize + start | 0 : start;
    allStarts[ax] = s;
    allSteps[ax] = getOr(steps[i], 1);
  });
  let coordCode = outputStrides.map((stride, i) => {
    let inStride = getOr(inputStrides[i], 1);
    let start = getOr(allStarts[i], 0);
    let step = getOr(allSteps[i], 1);
    return `  let coord_` + i.toString() + ` = remaining / ` + stride.toString() + `u;
  remaining = remaining % ` + stride.toString() + `u;
  let in_coord_` + i.toString() + ` = ` + start.toString() + `u + coord_` + i.toString() + ` * ` + step.toString() + `u;
  input_idx = input_idx + in_coord_` + i.toString() + ` * ` + inStride.toString() + `u;`;
  }).join("\n");
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  var remaining = idx;
  var input_idx = 0u;
` + coordCode + `
  output[idx] = input[input_idx];
` + mainEnd;
  return {
    name: "slice_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: numElements(inputShape) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genCumsumKernel(inputShape, axis, exclusive, reverse2) {
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0, i_finish = inputShape.length; i$1 < i_finish; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = numElements(inputShape);
  let numSlices = outer * inner | 0;
  let loopStart = reverse2 ? "axis_size - 1u" : "0u";
  let loopCond = reverse2 ? "i > 0u || i == 0u" : "i < axis_size";
  let loopIncr = reverse2 ? "i = i - 1u" : "i = i + 1u";
  let exclusiveCode = exclusive ? "output[out_idx] = sum;\n    sum = sum + input[in_idx];" : "sum = sum + input[in_idx];\n    output[out_idx] = sum;";
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  let slice_idx = idx;
  if (slice_idx >= ` + numSlices.toString() + `u) { return; }
  let axis_size = ` + axisSize.toString() + `u;
  let inner_size = ` + inner.toString() + `u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  var sum = 0.0;
  for (var i = ` + loopStart + `; ` + loopCond + `; ` + loopIncr + `) {
    let in_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let out_idx = in_idx;
    ` + exclusiveCode + `
  }
` + mainEnd;
  return {
    name: "cumsum_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genOneHotKernel(inputShape, depth) {
  let inputSize = numElements(inputShape);
  let outputSize = inputSize * depth | 0;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  let depth = ` + depth.toString() + `u;
  let input_idx = idx / depth;
  let depth_idx = idx % depth;
  let class_idx = u32(input[input_idx]);
  output[idx] = select(0.0, 1.0, depth_idx == class_idx);
` + mainEnd;
  return {
    name: "onehot_" + depth.toString() + "_" + inputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: inputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genScatterKernel(dataShape, indicesSize, axis) {
  let outputSize = numElements(dataShape);
  let rank = dataShape.length;
  let axisSize = getOr(dataShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(dataShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(dataShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let updateSize = (outer * indicesSize | 0) * inner | 0;
  let wgsl = storageBuffer(0, "data", "ReadOnly") + `
` + storageBuffer(1, "indices", "ReadOnly") + `
` + storageBuffer(2, "updates", "ReadOnly") + `
` + storageBuffer(3, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  output[idx] = data[idx];
  let inner_size = ` + inner.toString() + `u;
  let indices_size = ` + indicesSize.toString() + `u;
  let axis_size = ` + axisSize.toString() + `u;
  let update_size = ` + updateSize.toString() + `u;
  for (var u = 0u; u < update_size; u = u + 1u) {
    let outer_idx = u / (indices_size * inner_size);
    let remainder = u % (indices_size * inner_size);
    let index_idx = remainder / inner_size;
    let inner_idx = remainder % inner_size;
    let scatter_idx = u32(indices[index_idx]);
    let out_idx = outer_idx * (axis_size * inner_size) + scatter_idx * inner_size + inner_idx;
    if (out_idx == idx) {
      output[idx] = updates[u];
    }
  }
` + mainEnd;
  return {
    name: "scatter_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "data"
      },
      {
        binding: 1,
        size: indicesSize << 2,
        usage: "ReadOnly",
        name: "indices"
      },
      {
        binding: 2,
        size: updateSize << 2,
        usage: "ReadOnly",
        name: "updates"
      },
      {
        binding: 3,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genCastKernel(size3) {
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size3.toString() + `u) { return; }
  output[idx] = input[idx];
` + mainEnd;
  return {
    name: "cast_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSqueezeKernel(size3) {
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size3.toString() + `u) { return; }
  output[idx] = input[idx];
` + mainEnd;
  return {
    name: "squeeze_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genStackKernel(inputShape, numInputs, axis) {
  let rank = inputShape.length;
  let inputSize = numElements(inputShape);
  let outputSize = inputSize * numInputs | 0;
  let normAxis = axis < 0 ? (rank + 1 | 0) + axis | 0 : axis;
  let beforeSize = 1;
  let afterSize = 1;
  for (let i = 0; i < normAxis; ++i) {
    beforeSize = beforeSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = normAxis; i$1 < rank; ++i$1) {
    afterSize = afterSize * getOr(inputShape[i$1], 1) | 0;
  }
  let after = afterSize;
  let wgsl = storageBuffer(0, "input0", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  let after_size = ` + after.toString() + `u;
  let num_inputs = ` + numInputs.toString() + `u;
  let before_idx = idx / (num_inputs * after_size);
  let remainder = idx % (num_inputs * after_size);
  let stack_idx = remainder / after_size;
  let after_idx = remainder % after_size;
  let input_idx = before_idx * after_size + after_idx;
  // For now, only support 2 inputs - would need dynamic binding for more
  output[idx] = input0[input_idx];
` + mainEnd;
  return {
    name: "stack_" + numInputs.toString() + "_" + inputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: inputSize << 2,
        usage: "ReadOnly",
        name: "input0"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genBroadcastKernel(inputShape, targetShape) {
  let inputSize = numElements(inputShape);
  let outputSize = numElements(targetShape);
  let rank = targetShape.length;
  let inputRank = inputShape.length;
  let inputStrides = fromInitializer(inputRank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < inputRank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let outputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(targetShape[j], 1) | 0;
    }
    return stride;
  });
  let rankDiff = rank - inputRank | 0;
  let coordCode = outputStrides.map((stride, i) => {
    let inputIdx = i - rankDiff | 0;
    let inDim = inputIdx >= 0 ? getOr(inputShape[inputIdx], 1) : 1;
    let inStride = inputIdx >= 0 ? getOr(inputStrides[inputIdx], 1) : 0;
    return `  let coord_` + i.toString() + ` = (remaining / ` + stride.toString() + `u) % ` + getOr(targetShape[i], 1).toString() + `u;
  remaining = remaining % ` + stride.toString() + `u;
  let in_coord_` + i.toString() + ` = coord_` + i.toString() + ` % ` + inDim.toString() + `u;
  input_idx = input_idx + in_coord_` + i.toString() + ` * ` + inStride.toString() + `u;`;
  }).join("\n");
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  var remaining = idx;
  var input_idx = 0u;
` + coordCode + `
  output[idx] = input[input_idx];
` + mainEnd;
  return {
    name: "broadcast_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: inputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genLSTMCellKernel(batchSize, inputSize, hiddenSize) {
  let outputSize = batchSize * hiddenSize | 0;
  let gateSize = hiddenSize << 2;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "h_prev", "ReadOnly") + `
` + storageBuffer(2, "c_prev", "ReadOnly") + `
` + storageBuffer(3, "weight_ih", "ReadOnly") + `
` + storageBuffer(4, "weight_hh", "ReadOnly") + `
` + storageBuffer(5, "bias_ih", "ReadOnly") + `
` + storageBuffer(6, "bias_hh", "ReadOnly") + `
` + storageBuffer(7, "h_out", "ReadWrite") + `
` + storageBuffer(8, "c_out", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  
  let batch_size = ` + batchSize.toString() + `u;
  let input_size = ` + inputSize.toString() + `u;
  let hidden_size = ` + hiddenSize.toString() + `u;
  let gate_size = ` + gateSize.toString() + `u;
  
  let batch_idx = idx / hidden_size;
  let h_idx = idx % hidden_size;
  
  // Compute gates: [i, f, g, o] = sigmoid/tanh(W_ih * x + W_hh * h + b)
  var gates: array<f32, 4>;
  for (var g = 0u; g < 4u; g = g + 1u) {
    var sum = 0.0;
    let gate_offset = g * hidden_size + h_idx;
    
    // W_ih * x
    for (var i = 0u; i < input_size; i = i + 1u) {
      let x_val = input[batch_idx * input_size + i];
      let w_val = weight_ih[gate_offset * input_size + i];
      sum = sum + x_val * w_val;
    }
    
    // W_hh * h_prev
    for (var i = 0u; i < hidden_size; i = i + 1u) {
      let h_val = h_prev[batch_idx * hidden_size + i];
      let w_val = weight_hh[gate_offset * hidden_size + i];
      sum = sum + h_val * w_val;
    }
    
    // Add biases
    sum = sum + bias_ih[gate_offset] + bias_hh[gate_offset];
    
    gates[g] = sum;
  }
  
  // Apply activations
  let i_gate = 1.0 / (1.0 + exp(-gates[0]));  // sigmoid
  let f_gate = 1.0 / (1.0 + exp(-gates[1]));  // sigmoid
  let g_gate = tanh(gates[2]);                 // tanh
  let o_gate = 1.0 / (1.0 + exp(-gates[3]));  // sigmoid
  
  // Cell state update: c = f * c_prev + i * g
  let c_prev_val = c_prev[idx];
  let c_new = f_gate * c_prev_val + i_gate * g_gate;
  c_out[idx] = c_new;
  
  // Hidden state: h = o * tanh(c)
  h_out[idx] = o_gate * tanh(c_new);
` + mainEnd;
  return {
    name: "lstm_cell_" + batchSize.toString() + "_" + hiddenSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (batchSize * inputSize | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (batchSize * hiddenSize | 0) << 2,
        usage: "ReadOnly",
        name: "h_prev"
      },
      {
        binding: 2,
        size: (batchSize * hiddenSize | 0) << 2,
        usage: "ReadOnly",
        name: "c_prev"
      },
      {
        binding: 3,
        size: (gateSize * inputSize | 0) << 2,
        usage: "ReadOnly",
        name: "weight_ih"
      },
      {
        binding: 4,
        size: (gateSize * hiddenSize | 0) << 2,
        usage: "ReadOnly",
        name: "weight_hh"
      },
      {
        binding: 5,
        size: gateSize << 2,
        usage: "ReadOnly",
        name: "bias_ih"
      },
      {
        binding: 6,
        size: gateSize << 2,
        usage: "ReadOnly",
        name: "bias_hh"
      },
      {
        binding: 7,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "h_out"
      },
      {
        binding: 8,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "c_out"
      }
    ]
  };
}
function genGRUCellKernel(batchSize, inputSize, hiddenSize) {
  let outputSize = batchSize * hiddenSize | 0;
  let gateSize = 3 * hiddenSize | 0;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "h_prev", "ReadOnly") + `
` + storageBuffer(2, "weight_ih", "ReadOnly") + `
` + storageBuffer(3, "weight_hh", "ReadOnly") + `
` + storageBuffer(4, "bias_ih", "ReadOnly") + `
` + storageBuffer(5, "bias_hh", "ReadOnly") + `
` + storageBuffer(6, "h_out", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  
  let batch_size = ` + batchSize.toString() + `u;
  let input_size = ` + inputSize.toString() + `u;
  let hidden_size = ` + hiddenSize.toString() + `u;
  
  let batch_idx = idx / hidden_size;
  let h_idx = idx % hidden_size;
  
  // Compute r and z gates
  var r_sum = 0.0;
  var z_sum = 0.0;
  
  // W_ih * x for r and z
  for (var i = 0u; i < input_size; i = i + 1u) {
    let x_val = input[batch_idx * input_size + i];
    r_sum = r_sum + x_val * weight_ih[h_idx * input_size + i];
    z_sum = z_sum + x_val * weight_ih[(hidden_size + h_idx) * input_size + i];
  }
  
  // W_hh * h_prev for r and z
  for (var i = 0u; i < hidden_size; i = i + 1u) {
    let h_val = h_prev[batch_idx * hidden_size + i];
    r_sum = r_sum + h_val * weight_hh[h_idx * hidden_size + i];
    z_sum = z_sum + h_val * weight_hh[(hidden_size + h_idx) * hidden_size + i];
  }
  
  r_sum = r_sum + bias_ih[h_idx] + bias_hh[h_idx];
  z_sum = z_sum + bias_ih[hidden_size + h_idx] + bias_hh[hidden_size + h_idx];
  
  let r_gate = 1.0 / (1.0 + exp(-r_sum));  // sigmoid
  let z_gate = 1.0 / (1.0 + exp(-z_sum));  // sigmoid
  
  // Compute n gate with reset
  var n_sum = 0.0;
  for (var i = 0u; i < input_size; i = i + 1u) {
    let x_val = input[batch_idx * input_size + i];
    n_sum = n_sum + x_val * weight_ih[(2u * hidden_size + h_idx) * input_size + i];
  }
  n_sum = n_sum + bias_ih[2u * hidden_size + h_idx];
  
  var h_sum = 0.0;
  for (var i = 0u; i < hidden_size; i = i + 1u) {
    let h_val = h_prev[batch_idx * hidden_size + i];
    h_sum = h_sum + r_gate * h_val * weight_hh[(2u * hidden_size + h_idx) * hidden_size + i];
  }
  h_sum = h_sum + bias_hh[2u * hidden_size + h_idx];
  
  let n_gate = tanh(n_sum + h_sum);
  
  // Output: h = (1 - z) * n + z * h_prev
  let h_prev_val = h_prev[idx];
  h_out[idx] = (1.0 - z_gate) * n_gate + z_gate * h_prev_val;
` + mainEnd;
  return {
    name: "gru_cell_" + batchSize.toString() + "_" + hiddenSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (batchSize * inputSize | 0) << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: (batchSize * hiddenSize | 0) << 2,
        usage: "ReadOnly",
        name: "h_prev"
      },
      {
        binding: 2,
        size: (gateSize * inputSize | 0) << 2,
        usage: "ReadOnly",
        name: "weight_ih"
      },
      {
        binding: 3,
        size: (gateSize * hiddenSize | 0) << 2,
        usage: "ReadOnly",
        name: "weight_hh"
      },
      {
        binding: 4,
        size: gateSize << 2,
        usage: "ReadOnly",
        name: "bias_ih"
      },
      {
        binding: 5,
        size: gateSize << 2,
        usage: "ReadOnly",
        name: "bias_hh"
      },
      {
        binding: 6,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "h_out"
      }
    ]
  };
}
function genCumprodKernel(inputShape, axis, exclusive, reverse2) {
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0, i_finish = inputShape.length; i$1 < i_finish; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = numElements(inputShape);
  let numSlices = outer * inner | 0;
  let loopStart = reverse2 ? "axis_size - 1u" : "0u";
  let loopCond = reverse2 ? "i > 0u || i == 0u" : "i < axis_size";
  let loopIncr = reverse2 ? "i = i - 1u" : "i = i + 1u";
  let exclusiveCode = exclusive ? "output[out_idx] = prod;\n    prod = prod * input[in_idx];" : "prod = prod * input[in_idx];\n    output[out_idx] = prod;";
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  let slice_idx = idx;
  if (slice_idx >= ` + numSlices.toString() + `u) { return; }
  let axis_size = ` + axisSize.toString() + `u;
  let inner_size = ` + inner.toString() + `u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  var prod = 1.0;
  for (var i = ` + loopStart + `; ` + loopCond + `; ` + loopIncr + `) {
    let in_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let out_idx = in_idx;
    ` + exclusiveCode + `
  }
` + mainEnd;
  return {
    name: "cumprod_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genReverseKernel(inputShape, axes) {
  let rank = inputShape.length;
  let outputSize = numElements(inputShape);
  let strides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let coordCode = strides.map((stride, i) => {
    let dim = getOr(inputShape[i], 1);
    let isReversed = axes.includes(i);
    let coordExpr = isReversed ? (dim - 1 | 0).toString() + `u - coord_` + i.toString() : `coord_` + i.toString();
    return `  let coord_` + i.toString() + ` = (remaining / ` + stride.toString() + `u) % ` + dim.toString() + `u;
  remaining = remaining % ` + stride.toString() + `u;
  input_idx = input_idx + (` + coordExpr + `) * ` + stride.toString() + `u;`;
  }).join("\n");
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + outputSize.toString() + `u) { return; }
  var remaining = idx;
  var input_idx = 0u;
` + coordCode + `
  output[idx] = input[input_idx];
` + mainEnd;
  return {
    name: "reverse_" + outputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genLogSoftmaxKernel(outerSize, axisSize) {
  let outputSize = outerSize * axisSize | 0;
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  let outer_idx = idx;
  if (outer_idx >= ` + outerSize.toString() + `u) { return; }
  let axis_size = ` + axisSize.toString() + `u;
  let base = outer_idx * axis_size;
  
  // Find max for numerical stability
  var max_val = input[base];
  for (var i = 1u; i < axis_size; i = i + 1u) {
    max_val = max(max_val, input[base + i]);
  }
  
  // Compute sum of exp(x - max)
  var sum_exp = 0.0;
  for (var i = 0u; i < axis_size; i = i + 1u) {
    sum_exp = sum_exp + exp(input[base + i] - max_val);
  }
  
  // Compute log_softmax = x - max - log(sum_exp)
  let log_sum = log(sum_exp);
  for (var i = 0u; i < axis_size; i = i + 1u) {
    output[base + i] = input[base + i] - max_val - log_sum;
  }
` + mainEnd;
  return {
    name: "logsoftmax_" + outerSize.toString() + "_" + axisSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSortKernel(inputShape, axis, descending) {
  let rank = inputShape.length;
  let axisSize = getOr(inputShape[axis], 1);
  let outerSize = 1;
  let innerSize = 1;
  for (let i = 0; i < axis; ++i) {
    outerSize = outerSize * getOr(inputShape[i], 1) | 0;
  }
  for (let i$1 = axis + 1 | 0; i$1 < rank; ++i$1) {
    innerSize = innerSize * getOr(inputShape[i$1], 1) | 0;
  }
  let outer = outerSize;
  let inner = innerSize;
  let outputSize = numElements(inputShape);
  let numSlices = outer * inner | 0;
  let cmpOp = descending ? "<" : ">";
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  let slice_idx = idx;
  if (slice_idx >= ` + numSlices.toString() + `u) { return; }
  
  let axis_size = ` + axisSize.toString() + `u;
  let inner_size = ` + inner.toString() + `u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  
  // Copy to output first
  for (var i = 0u; i < axis_size; i = i + 1u) {
    let idx_src = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    output[idx_src] = input[idx_src];
  }
  
  // Bubble sort along axis
  for (var i = 0u; i < axis_size - 1u; i = i + 1u) {
    for (var j = 0u; j < axis_size - 1u - i; j = j + 1u) {
      let idx_j = outer_idx * (axis_size * inner_size) + j * inner_size + inner_idx;
      let idx_j1 = outer_idx * (axis_size * inner_size) + (j + 1u) * inner_size + inner_idx;
      let val_j = output[idx_j];
      let val_j1 = output[idx_j1];
      if (val_j ` + cmpOp + ` val_j1) {
        output[idx_j] = val_j1;
        output[idx_j1] = val_j;
      }
    }
  }
` + mainEnd;
  return {
    name: "sort_" + numSlices.toString() + "_" + axisSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: outputSize << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genArangeKernel(size3, start, step) {
  let wgsl = storageBuffer(0, "output", "ReadWrite") + `
@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ` + size3.toString() + `u) { return; }
  output[idx] = ` + start.toString() + ` + f32(idx) * ` + step.toString() + `;
}`;
  return {
    name: "arange_" + size3.toString(),
    wgsl,
    bindings: [{
      binding: 0,
      size: size3 << 2,
      usage: "ReadWrite",
      name: "output"
    }]
  };
}
function at2(arr, i) {
  return getOr(arr[i], 0);
}
function generate(op, inputShapes) {
  let outputShape = infer(op, inputShapes);
  let input2 = getOr(inputShapes[0], []);
  let r = input2.length;
  return flatMap(outputShape, (outShape) => {
    let outSize = numElements(outShape);
    let exit = 0;
    if (typeof op !== "object") {
      switch (op) {
        case "Add":
        case "Sub":
        case "Mul":
        case "Div":
        case "Pow":
        case "Mod":
        case "FloorDiv":
        case "Maximum":
        case "Minimum":
        case "Atan2":
        case "Equal":
        case "NotEqual":
        case "Greater":
        case "GreaterEqual":
        case "Less":
        case "LessEqual":
        case "And":
        case "Or":
        case "Xor":
          exit = 2;
          break;
        case "Identity":
        case "Neg":
        case "Abs":
        case "Sign":
        case "Reciprocal":
        case "Floor":
        case "Ceil":
        case "Round":
        case "Sqrt":
        case "Exp":
        case "Log":
        case "Log2":
        case "Log10":
        case "Sin":
        case "Cos":
        case "Tan":
        case "Asin":
        case "Acos":
        case "Atan":
        case "Sinh":
        case "Cosh":
        case "Tanh":
        case "Asinh":
        case "Acosh":
        case "Atanh":
        case "ReLU":
        case "Sigmoid":
        case "Softplus":
        case "Softsign":
        case "GeLU":
        case "SiLU":
        case "Mish":
        case "Not":
          exit = 1;
          break;
        case "Where":
          let k = genWhereKernel(outSize);
          return [
            k,
            computeDispatch(outSize, k.name, 0)
          ];
        case "MatMul":
          let s2 = getOr(inputShapes[1], []);
          let r1 = input2.length;
          let r2 = s2.length;
          if (!(r1 >= 2 && r2 >= 2)) {
            return;
          }
          let m = at2(input2, r1 - 2 | 0);
          let k$1 = at2(input2, r1 - 1 | 0);
          let n = at2(s2, r2 - 1 | 0);
          let batchA = input2.slice(0, r1 - 2 | 0);
          let batchB = s2.slice(0, r2 - 2 | 0);
          let batchOut = outShape.slice(0, outShape.length - 2 | 0);
          let batchSizeOut = reduce(batchOut, 1, (a, b) => a * b | 0);
          let totalOutput = (batchSizeOut * m | 0) * n | 0;
          let kernel = genBatchedMatMulKernel(batchA, batchB, batchOut, m, k$1, n);
          if (batchSizeOut !== 1) {
            return [
              kernel,
              computeDispatch(totalOutput, kernel.name, 0)
            ];
          }
          if (m > 16) {
            return [
              kernel,
              computeDispatch2D(n, m, 16, kernel.name, 0)
            ];
          }
          let smallKernel = genSmallMMatMulKernel(m, k$1, n);
          return [
            smallKernel,
            computeDispatch(m * n | 0, smallKernel.name, 0)
          ];
        case "GlobalMaxPool":
          if (r !== 4) {
            return;
          }
          let batch = at2(input2, 0);
          let height = at2(input2, 1);
          let width = at2(input2, 2);
          let channels = at2(input2, 3);
          let k$2 = genGlobalPoolKernel("max", batch, height, width, channels);
          return [
            k$2,
            computeDispatch(batch * channels | 0, k$2.name, 0)
          ];
        case "GlobalAvgPool":
          if (r !== 4) {
            return;
          }
          let batch$1 = at2(input2, 0);
          let height$1 = at2(input2, 1);
          let width$1 = at2(input2, 2);
          let channels$1 = at2(input2, 3);
          let k$3 = genGlobalPoolKernel("avg", batch$1, height$1, width$1, channels$1);
          return [
            k$3,
            computeDispatch(batch$1 * channels$1 | 0, k$3.name, 0)
          ];
        default:
          return;
      }
    } else {
      switch (op.TAG) {
        case "LeakyReLU":
        case "ELU":
          exit = 1;
          break;
        case "Clip":
          let minVal = getOr(op.min, -3402823e32);
          let maxVal = getOr(op.max, 3402823e32);
          let k$4 = genClipKernel(outSize, minVal, maxVal);
          return [
            k$4,
            computeDispatch(outSize, k$4.name, 0)
          ];
        case "Reduce":
          return map(genReduceKernel(op.op, input2, op.axes, false), (k) => [
            k,
            computeDispatch(outSize, k.name, 0)
          ]);
        case "ArgMax":
          let kernel$1 = genArgMaxKernel(input2, op.axis, op.selectLastIndex);
          return [
            kernel$1,
            computeDispatch(outSize, kernel$1.name, 0)
          ];
        case "ArgMin":
          let kernel$2 = genArgMinKernel(input2, op.axis, op.selectLastIndex);
          return [
            kernel$2,
            computeDispatch(outSize, kernel$2.name, 0)
          ];
        case "CumSum":
          let axis = op.axis;
          let kernel$3 = genCumsumKernel(input2, axis, op.exclusive, op.reverse);
          return [
            kernel$3,
            computeDispatch(div(numElements(input2), getOr(input2[axis], 1)), kernel$3.name, 0)
          ];
        case "CumProd":
          let axis$1 = op.axis;
          let kernel$4 = genCumprodKernel(input2, axis$1, op.exclusive, op.reverse);
          return [
            kernel$4,
            computeDispatch(div(numElements(input2), getOr(input2[axis$1], 1)), kernel$4.name, 0)
          ];
        case "MatMulInt4":
          let groupSize = op.groupSize;
          let s2$1 = getOr(inputShapes[1], []);
          let r1$1 = input2.length;
          let m$1 = r1$1 === 1 ? 1 : at2(input2, r1$1 - 2 | 0);
          let k$5 = at2(input2, r1$1 - 1 | 0);
          let n$1 = at2(s2$1, 1);
          if (m$1 === 1 && groupSize === 128) {
            let kernel$5 = genInt4MatMulKernel(m$1, k$5, n$1, groupSize);
            return [
              kernel$5,
              computeDispatchInt4Opt(n$1, 64, kernel$5.name, 0)
            ];
          }
          let kernel$6 = genInt4MatMulTiledKernel(m$1, k$5, n$1, groupSize);
          return [
            kernel$6,
            computeDispatch2D(n$1, m$1, 16, kernel$6.name, 0)
          ];
        case "Reshape":
          let kernel$7 = genReshapeKernel(outSize);
          return [
            kernel$7,
            computeDispatch(outSize, kernel$7.name, 0)
          ];
        case "Transpose":
          return map(genTransposeKernel(input2, op.perm), (k) => [
            k,
            computeDispatch(outSize, k.name, 0)
          ]);
        case "Broadcast":
          let kernel$8 = genBroadcastKernel(input2, op.targetShape);
          return [
            kernel$8,
            computeDispatch(outSize, kernel$8.name, 0)
          ];
        case "Squeeze":
        case "Unsqueeze":
        case "ExpandDims":
          exit = 3;
          break;
        case "Slice":
          let kernel$9 = genSliceKernel(input2, op.starts, op.ends, op.axes, op.steps);
          return [
            kernel$9,
            computeDispatch(outSize, kernel$9.name, 0)
          ];
        case "Gather":
          let indicesShape = getOr(inputShapes[1], []);
          let indicesSize = numElements(indicesShape);
          let k$6 = genGatherKernel(input2, indicesSize, op.axis);
          return [
            k$6,
            computeDispatch(outSize, k$6.name, 0)
          ];
        case "Scatter":
          let indicesShape$1 = getOr(inputShapes[1], []);
          let indicesSize$1 = numElements(indicesShape$1);
          let kernel$10 = genScatterKernel(input2, indicesSize$1, op.axis);
          return [
            kernel$10,
            computeDispatch(outSize, kernel$10.name, 0)
          ];
        case "Concat":
          return map(genConcatKernel(inputShapes, op.axis), (k) => [
            k,
            computeDispatch(outSize, k.name, 0)
          ]);
        case "Split":
          let firstSplitSize = getOr(op.splitSizes[0], 1);
          let k$7 = genSplitKernel(input2, op.axis, 0, firstSplitSize);
          return [
            k$7,
            computeDispatch(outSize, k$7.name, 0)
          ];
        case "Stack":
          let numInputs = inputShapes.length;
          let kernel$11 = genStackKernel(input2, numInputs, op.axis);
          return [
            kernel$11,
            computeDispatch(outSize, kernel$11.name, 0)
          ];
        case "Tile":
          let kernel$12 = genTileKernel(input2, op.repeats);
          return [
            kernel$12,
            computeDispatch(outSize, kernel$12.name, 0)
          ];
        case "Pad":
          let kernel$13 = genPadKernel(input2, op.pads, op.constantValue);
          return [
            kernel$13,
            computeDispatch(outSize, kernel$13.name, 0)
          ];
        case "Reverse":
          let kernel$14 = genReverseKernel(input2, op.axes);
          return [
            kernel$14,
            computeDispatch(outSize, kernel$14.name, 0)
          ];
        case "Cast":
          let kernel$15 = genCastKernel(outSize);
          return [
            kernel$15,
            computeDispatch(outSize, kernel$15.name, 0)
          ];
        case "Conv1D":
          if (r !== 3) {
            return;
          }
          let padding = op.padding;
          let kernel$16 = op.kernel;
          let batch$2 = at2(input2, 0);
          let inLen = at2(input2, 1);
          let inC = at2(input2, 2);
          let outLen = at2(outShape, 1);
          let pad2;
          pad2 = typeof padding !== "object" ? padding === "Same" ? (kernel$16 - 1 | 0) / 2 | 0 : 0 : at2(padding.pads, 0);
          let k$8 = genConv1DKernel(batch$2, inLen, inC, outLen, op.filters, kernel$16, op.stride, pad2);
          return [
            k$8,
            computeDispatch(outSize, k$8.name, 0)
          ];
        case "Conv2D":
          if (r !== 4) {
            return;
          }
          let padding$1 = op.padding;
          let match = op.stride;
          let match$1 = op.kernel;
          let kW = match$1[1];
          let kH = match$1[0];
          let batch$3 = at2(input2, 0);
          let inH = at2(input2, 1);
          let inW = at2(input2, 2);
          let inC$1 = at2(input2, 3);
          let outH = at2(outShape, 1);
          let outW = at2(outShape, 2);
          let match$2;
          if (typeof padding$1 !== "object") {
            match$2 = padding$1 === "Same" ? [
              (kH - 1 | 0) / 2 | 0,
              (kW - 1 | 0) / 2 | 0
            ] : [
              0,
              0
            ];
          } else {
            let pads = padding$1.pads;
            match$2 = [
              at2(pads, 0),
              at2(pads, 1)
            ];
          }
          let kernel$17 = genConv2DKernel(batch$3, inH, inW, inC$1, outH, outW, op.filters, kH, kW, match[0], match[1], match$2[0], match$2[1]);
          return [
            kernel$17,
            computeDispatch(outSize, kernel$17.name, 0)
          ];
        case "MaxPool2D":
          if (r !== 4) {
            return;
          }
          let padding$2 = op.padding;
          let match$3 = op.stride;
          let match$4 = op.kernel;
          let kW$1 = match$4[1];
          let kH$1 = match$4[0];
          let batch$4 = at2(input2, 0);
          let inH$1 = at2(input2, 1);
          let inW$1 = at2(input2, 2);
          let channels$2 = at2(input2, 3);
          let outH$1 = at2(outShape, 1);
          let outW$1 = at2(outShape, 2);
          let match$5;
          if (typeof padding$2 !== "object") {
            match$5 = padding$2 === "Same" ? [
              (kH$1 - 1 | 0) / 2 | 0,
              (kW$1 - 1 | 0) / 2 | 0
            ] : [
              0,
              0
            ];
          } else {
            let pads$1 = padding$2.pads;
            match$5 = [
              at2(pads$1, 0),
              at2(pads$1, 1)
            ];
          }
          let kernel$18 = genPool2DKernel("max", batch$4, inH$1, inW$1, channels$2, outH$1, outW$1, kH$1, kW$1, match$3[0], match$3[1], match$5[0], match$5[1]);
          return [
            kernel$18,
            computeDispatch(outSize, kernel$18.name, 0)
          ];
        case "AvgPool2D":
          if (r !== 4) {
            return;
          }
          let padding$3 = op.padding;
          let match$6 = op.stride;
          let match$7 = op.kernel;
          let kW$2 = match$7[1];
          let kH$2 = match$7[0];
          let batch$5 = at2(input2, 0);
          let inH$2 = at2(input2, 1);
          let inW$2 = at2(input2, 2);
          let channels$3 = at2(input2, 3);
          let outH$2 = at2(outShape, 1);
          let outW$2 = at2(outShape, 2);
          let match$8;
          if (typeof padding$3 !== "object") {
            match$8 = padding$3 === "Same" ? [
              (kH$2 - 1 | 0) / 2 | 0,
              (kW$2 - 1 | 0) / 2 | 0
            ] : [
              0,
              0
            ];
          } else {
            let pads$2 = padding$3.pads;
            match$8 = [
              at2(pads$2, 0),
              at2(pads$2, 1)
            ];
          }
          let kernel$19 = genPool2DKernel("avg", batch$5, inH$2, inW$2, channels$3, outH$2, outW$2, kH$2, kW$2, match$6[0], match$6[1], match$8[0], match$8[1]);
          return [
            kernel$19,
            computeDispatch(outSize, kernel$19.name, 0)
          ];
        case "BatchNorm":
          if (r !== 4) {
            return;
          }
          let batch$6 = at2(input2, 0);
          let height$2 = at2(input2, 1);
          let width$2 = at2(input2, 2);
          let channels$4 = at2(input2, 3);
          let kernel$20 = genBatchNormKernel(batch$6, height$2, width$2, channels$4);
          return [
            kernel$20,
            computeDispatch(outSize, kernel$20.name, 0)
          ];
        case "LayerNorm":
          let normAxis = r - 1 | 0;
          let normSize = at2(input2, normAxis);
          let outerSize = div(numElements(input2), normSize);
          let k$9 = genLayerNormKernel(outerSize, normSize, op.epsilon);
          return [
            k$9,
            computeDispatch(outerSize, k$9.name, 0)
          ];
        case "Softmax":
          let axis$2 = op.axis;
          let normAxis$1 = axis$2 < 0 ? r + axis$2 | 0 : axis$2;
          if (normAxis$1 !== (r - 1 | 0)) {
            return;
          }
          let axisSize = at2(input2, normAxis$1);
          let outerSize$1 = div(numElements(input2), axisSize);
          let kernel$21 = genSoftmaxKernel(outerSize$1, axisSize);
          return [
            kernel$21,
            computeDispatch(outerSize$1, kernel$21.name, 0)
          ];
        case "LogSoftmax":
          let axis$3 = op.axis;
          let normAxis$2 = axis$3 < 0 ? r + axis$3 | 0 : axis$3;
          if (normAxis$2 !== (r - 1 | 0)) {
            return;
          }
          let axisSize$1 = getOr(input2[normAxis$2], 1);
          let outerSize$2 = div(numElements(input2), axisSize$1);
          let kernel$22 = genLogSoftmaxKernel(outerSize$2, axisSize$1);
          return [
            kernel$22,
            computeDispatch(outerSize$2, kernel$22.name, 0)
          ];
        case "Dropout":
          let kernel$23 = genDropoutKernel(outSize, op.rate);
          return [
            kernel$23,
            computeDispatch(outSize, kernel$23.name, 0)
          ];
        case "Dense":
          if (r < 1) {
            return;
          }
          let batchSize = numElements(input2.slice(0, r - 1 | 0));
          let inFeatures = at2(input2, r - 1 | 0);
          let kernel$24 = genDenseKernel(max(batchSize, 1), inFeatures, op.units);
          return [
            kernel$24,
            computeDispatch(outSize, kernel$24.name, 0)
          ];
        case "ScaledDotProductAttention":
          if (r !== 3) {
            return;
          }
          let batch$7 = at2(input2, 0);
          let seqLen = at2(input2, 1);
          let dim = at2(input2, 2);
          let k$10 = genAttentionKernel(batch$7, seqLen, dim);
          return [
            k$10,
            computeDispatch(outSize, k$10.name, 0)
          ];
        case "OneHot":
          let kernel$25 = genOneHotKernel(input2, op.depth);
          return [
            kernel$25,
            computeDispatch(outSize, kernel$25.name, 0)
          ];
        case "Embedding":
          let batchSeq = numElements(input2);
          let k$11 = genEmbeddingKernel(batchSeq, op.numEmbeddings, op.embeddingDim);
          return [
            k$11,
            computeDispatch(outSize, k$11.name, 0)
          ];
        case "TopK":
          let k$12 = op.k;
          let kernel$26 = genTopKKernel(input2, k$12, op.axis);
          let numSlices = div(outSize, k$12);
          return [
            kernel$26,
            computeDispatch(numSlices, kernel$26.name, 0)
          ];
        case "Sort":
          let axis$4 = op.axis;
          let kernel$27 = genSortKernel(input2, axis$4, op.descending);
          let numSlices$1 = div(outSize, getOr(input2[axis$4], 1));
          return [
            kernel$27,
            computeDispatch(numSlices$1, kernel$27.name, 0)
          ];
        default:
          return;
      }
    }
    switch (exit) {
      case 1:
        return map(genUnaryKernel(op, outSize), (k) => [
          k,
          computeDispatch(outSize, k.name, 0)
        ]);
      case 2:
        let shape0 = getOr(inputShapes[0], []);
        let shape1 = getOr(inputShapes[1], []);
        return map(genBinaryBroadcastKernel(op, shape0, shape1, outShape), (k) => [
          k,
          computeDispatch(outSize, k.name, 0)
        ]);
      case 3:
        let kernel$28 = genSqueezeKernel(outSize);
        return [
          kernel$28,
          computeDispatch(outSize, kernel$28.name, 0)
        ];
    }
  });
}
var workgroupSize = 256;

// src/Types.res.mjs
var Types_res_exports = {};

// src/Compiler.res.mjs
var Compiler_res_exports = {};
__export(Compiler_res_exports, {
  abs_: () => abs_,
  add: () => add2,
  addNode: () => addNode,
  addNodeInternal: () => addNodeInternal,
  addNodeWithRefs: () => addNodeWithRefs,
  addScalar: () => addScalar,
  allocateBuffers: () => allocateBuffers,
  arange: () => arange,
  argmax: () => argmax,
  argmin: () => argmin,
  avgPool2d: () => avgPool2d,
  batchNormWithParams: () => batchNormWithParams,
  broadcast: () => broadcast2,
  cast: () => cast,
  causalMask: () => causalMask,
  chunk: () => chunk,
  clamp: () => clamp,
  clip: () => clip,
  compile: () => compile,
  compileNode: () => compileNode,
  compileWithOutputs: () => compileWithOutputs,
  concat: () => concat,
  constant: () => constant,
  conv2d: () => conv2d,
  conv2dWithWeights: () => conv2dWithWeights,
  cos_: () => cos_,
  createGraph: () => createGraph,
  cumprod: () => cumprod,
  cumsum: () => cumsum,
  dense: () => dense,
  denseWithWeights: () => denseWithWeights,
  div: () => div2,
  divByScalar: () => divByScalar,
  embedding: () => embedding,
  exp_: () => exp_,
  expandDims: () => expandDims,
  feedForward: () => feedForward,
  findBufferId: () => findBufferId,
  flatten: () => flatten,
  gather: () => gather,
  gelu: () => gelu,
  getOpOutputCount: () => getOpOutputCount,
  globalAvgPool: () => globalAvgPool,
  globalMaxPool: () => globalMaxPool,
  gru: () => gru,
  gruCell: () => gruCell,
  input: () => input,
  layerNorm: () => layerNorm,
  layerNormWithParams: () => layerNormWithParams,
  linear: () => linear,
  logSoftmax: () => logSoftmax,
  log_: () => log_,
  lstm: () => lstm,
  lstmCell: () => lstmCell,
  makeCausalMaskData: () => makeCausalMaskData,
  markOutput: () => markOutput,
  markOutputRef: () => markOutputRef,
  maskedAttention: () => maskedAttention,
  matmul: () => matmul,
  maxPool2d: () => maxPool2d,
  maximum: () => maximum2,
  minimum: () => minimum2,
  mkRef: () => mkRef,
  mul: () => mul,
  multiHeadAttention: () => multiHeadAttention,
  neg: () => neg,
  oneHot: () => oneHot,
  pad: () => pad,
  pow_: () => pow_,
  printCompiled: () => printCompiled,
  printGraph: () => printGraph,
  reduce: () => reduce3,
  reduceMax: () => reduceMax,
  reduceMean: () => reduceMean,
  reduceSum: () => reduceSum,
  refOutput: () => refOutput,
  relu: () => relu,
  reshape: () => reshape,
  reverse: () => reverse,
  scale: () => scale,
  scaledDotProductAttention: () => scaledDotProductAttention,
  scatter: () => scatter,
  sigmoid: () => sigmoid,
  silu: () => silu,
  sin_: () => sin_,
  sinusoidalPositionalEncoding: () => sinusoidalPositionalEncoding,
  slice_: () => slice_,
  softmax: () => softmax,
  sort: () => sort,
  split: () => split,
  sqrt_: () => sqrt_,
  squeeze: () => squeeze,
  stack: () => stack,
  sub: () => sub,
  tanh_: () => tanh_,
  tensorKindToString: () => tensorKindToString,
  tile: () => tile,
  topk: () => topk,
  topkValues: () => topkValues,
  topologicalSort: () => topologicalSort,
  transformerBlock: () => transformerBlock,
  transpose: () => transpose,
  unsqueeze: () => unsqueeze,
  weight: () => weight,
  weightWithData: () => weightWithData,
  where: () => where
});

// node_modules/@rescript/runtime/lib/es6/Belt_internalAVLset.js
function create(l, v, r) {
  let hl = l !== void 0 ? l.h : 0;
  let hr = r !== void 0 ? r.h : 0;
  return {
    v,
    h: (hl >= hr ? hl : hr) + 1 | 0,
    l,
    r
  };
}
function singleton(x) {
  return {
    v: x,
    h: 1,
    l: void 0,
    r: void 0
  };
}
function heightGe(l, r) {
  if (r !== void 0) {
    if (l !== void 0) {
      return l.h >= r.h;
    } else {
      return false;
    }
  } else {
    return true;
  }
}
function bal(l, v, r) {
  let hl = l !== void 0 ? l.h : 0;
  let hr = r !== void 0 ? r.h : 0;
  if (hl > (hr + 2 | 0)) {
    let ll = l.l;
    let lr = l.r;
    if (heightGe(ll, lr)) {
      return create(ll, l.v, create(lr, v, r));
    } else {
      return create(create(ll, l.v, lr.l), lr.v, create(lr.r, v, r));
    }
  }
  if (hr <= (hl + 2 | 0)) {
    return {
      v,
      h: (hl >= hr ? hl : hr) + 1 | 0,
      l,
      r
    };
  }
  let rl = r.l;
  let rr = r.r;
  if (heightGe(rr, rl)) {
    return create(create(l, v, rl), r.v, rr);
  } else {
    return create(create(l, v, rl.l), rl.v, create(rl.r, r.v, rr));
  }
}
function rotateWithLeftChild(k2) {
  let k1 = k2.l;
  k2.l = k1.r;
  k1.r = k2;
  let n = k2.l;
  let hlk2 = n !== void 0 ? n.h : 0;
  let n$1 = k2.r;
  let hrk2 = n$1 !== void 0 ? n$1.h : 0;
  k2.h = max(hlk2, hrk2) + 1 | 0;
  let n$2 = k1.l;
  let hlk1 = n$2 !== void 0 ? n$2.h : 0;
  let hk2 = k2.h;
  k1.h = max(hlk1, hk2) + 1 | 0;
  return k1;
}
function rotateWithRightChild(k1) {
  let k2 = k1.r;
  k1.r = k2.l;
  k2.l = k1;
  let n = k1.l;
  let hlk1 = n !== void 0 ? n.h : 0;
  let n$1 = k1.r;
  let hrk1 = n$1 !== void 0 ? n$1.h : 0;
  k1.h = max(hlk1, hrk1) + 1 | 0;
  let n$2 = k2.r;
  let hrk2 = n$2 !== void 0 ? n$2.h : 0;
  let hk1 = k1.h;
  k2.h = max(hrk2, hk1) + 1 | 0;
  return k2;
}
function doubleWithLeftChild(k3) {
  let k3l = k3.l;
  let v = rotateWithRightChild(k3l);
  k3.l = v;
  return rotateWithLeftChild(k3);
}
function doubleWithRightChild(k2) {
  let k2r = k2.r;
  let v = rotateWithLeftChild(k2r);
  k2.r = v;
  return rotateWithRightChild(k2);
}
function heightUpdateMutate(t) {
  let n = t.l;
  let hlt = n !== void 0 ? n.h : 0;
  let n$1 = t.r;
  let hrt = n$1 !== void 0 ? n$1.h : 0;
  t.h = max(hlt, hrt) + 1 | 0;
  return t;
}
function balMutate(nt) {
  let l = nt.l;
  let r = nt.r;
  let hl = l !== void 0 ? l.h : 0;
  let hr = r !== void 0 ? r.h : 0;
  if (hl > (2 + hr | 0)) {
    let ll = l.l;
    let lr = l.r;
    if (heightGe(ll, lr)) {
      return heightUpdateMutate(rotateWithLeftChild(nt));
    } else {
      return heightUpdateMutate(doubleWithLeftChild(nt));
    }
  }
  if (hr > (2 + hl | 0)) {
    let rl = r.l;
    let rr = r.r;
    if (heightGe(rr, rl)) {
      return heightUpdateMutate(rotateWithRightChild(nt));
    } else {
      return heightUpdateMutate(doubleWithRightChild(nt));
    }
  }
  nt.h = max(hl, hr) + 1 | 0;
  return nt;
}

// node_modules/@rescript/runtime/lib/es6/Belt_internalSetInt.js
function has(_t, x) {
  while (true) {
    let t = _t;
    if (t === void 0) {
      return false;
    }
    let v = t.v;
    if (x === v) {
      return true;
    }
    _t = x < v ? t.l : t.r;
    continue;
  }
  ;
}
function addMutate(t, x) {
  if (t === void 0) {
    return singleton(x);
  }
  let k = t.v;
  if (x === k) {
    return t;
  }
  let l = t.l;
  let r = t.r;
  if (x < k) {
    t.l = addMutate(l, x);
  } else {
    t.r = addMutate(r, x);
  }
  return balMutate(t);
}

// node_modules/@rescript/runtime/lib/es6/Belt_MutableSetInt.js
function add(d, k) {
  let oldRoot = d.data;
  let v = addMutate(oldRoot, k);
  if (v !== oldRoot) {
    d.data = v;
    return;
  }
}
function make2() {
  return {
    data: void 0
  };
}
function has2(d, x) {
  return has(d.data, x);
}

// src/Compiler.res.mjs
function mkRef(nodeId) {
  return {
    nodeId,
    outputIndex: 0
  };
}
function refOutput(nodeId, outputIndex) {
  return {
    nodeId,
    outputIndex
  };
}
function createGraph() {
  return {
    nodes: [],
    nextId: 0,
    outputIds: []
  };
}
function getOpOutputCount(op) {
  if (typeof op !== "object") {
    return 1;
  }
  switch (op.TAG) {
    case "Split":
      return op.splitSizes.length;
    case "TopK":
      return 2;
    default:
      return 1;
  }
}
function addNodeInternal(graph, op, inputs, name, kind, data) {
  let id = graph.nextId;
  let inputShapes = inputs.map((inputRef) => {
    let inputNode = graph.nodes.find((n) => n.id === inputRef.nodeId);
    if (inputNode !== void 0) {
      return getOr(inputNode.outputShapes[inputRef.outputIndex], []);
    } else {
      return [];
    }
  });
  let outputShapes;
  let exit = 0;
  if (typeof op !== "object") {
    exit = 1;
  } else {
    switch (op.TAG) {
      case "Split":
        let axis = op.axis;
        let inputShape = getOr(inputShapes[0], []);
        outputShapes = op.splitSizes.map((splitSize) => inputShape.map((dim, i) => {
          if (i === axis) {
            return splitSize;
          } else {
            return dim;
          }
        }));
        break;
      case "TopK":
        let axis$1 = op.axis;
        let k = op.k;
        let inputShape$1 = getOr(inputShapes[0], []);
        let outShape = inputShape$1.map((dim, i) => {
          if (i === axis$1 || axis$1 < 0 && i === (inputShape$1.length + axis$1 | 0)) {
            return k;
          } else {
            return dim;
          }
        });
        outputShapes = [
          outShape,
          outShape
        ];
        break;
      default:
        exit = 1;
    }
  }
  if (exit === 1) {
    let shape = infer(op, inputShapes);
    outputShapes = shape !== void 0 ? [shape] : [[]];
  }
  let node_dtype = "F32";
  let node = {
    id,
    op,
    inputs,
    outputShapes,
    dtype: node_dtype,
    name,
    kind,
    data
  };
  graph.nodes = graph.nodes.concat([node]);
  graph.nextId = graph.nextId + 1 | 0;
  return id;
}
function addNode(graph, op, inputs, name) {
  let inputRefs = inputs.map((id) => ({
    nodeId: id,
    outputIndex: 0
  }));
  return addNodeInternal(graph, op, inputRefs, name, "Intermediate", void 0);
}
function addNodeWithRefs(graph, op, inputs, name) {
  return addNodeInternal(graph, op, inputs, name, "Intermediate", void 0);
}
function input(graph, shape, name) {
  let id = graph.nextId;
  let node_op = {
    TAG: "Input",
    shape,
    dtype: "F32"
  };
  let node_inputs = [];
  let node_outputShapes = [shape];
  let node_dtype = "F32";
  let node_name = name;
  let node = {
    id,
    op: node_op,
    inputs: node_inputs,
    outputShapes: node_outputShapes,
    dtype: node_dtype,
    name: node_name,
    kind: "Input",
    data: void 0
  };
  graph.nodes = graph.nodes.concat([node]);
  graph.nextId = graph.nextId + 1 | 0;
  return id;
}
function weight(graph, shape, name) {
  let size3 = numElements(shape);
  let zeros = make(size3, 0);
  let id = graph.nextId;
  let node_op = {
    TAG: "Input",
    shape,
    dtype: "F32"
  };
  let node_inputs = [];
  let node_outputShapes = [shape];
  let node_dtype = "F32";
  let node_name = name;
  let node_data = zeros;
  let node = {
    id,
    op: node_op,
    inputs: node_inputs,
    outputShapes: node_outputShapes,
    dtype: node_dtype,
    name: node_name,
    kind: "Weight",
    data: node_data
  };
  graph.nodes = graph.nodes.concat([node]);
  graph.nextId = graph.nextId + 1 | 0;
  return id;
}
function weightWithData(graph, shape, name, data) {
  let id = graph.nextId;
  let node_op = {
    TAG: "Input",
    shape,
    dtype: "F32"
  };
  let node_inputs = [];
  let node_outputShapes = [shape];
  let node_dtype = "F32";
  let node_name = name;
  let node_data = data;
  let node = {
    id,
    op: node_op,
    inputs: node_inputs,
    outputShapes: node_outputShapes,
    dtype: node_dtype,
    name: node_name,
    kind: "Weight",
    data: node_data
  };
  graph.nodes = graph.nodes.concat([node]);
  graph.nextId = graph.nextId + 1 | 0;
  return id;
}
function constant(graph, shape, name, data) {
  let id = graph.nextId;
  let node_op = {
    TAG: "Input",
    shape,
    dtype: "F32"
  };
  let node_inputs = [];
  let node_outputShapes = [shape];
  let node_dtype = "F32";
  let node_name = name;
  let node_data = data;
  let node = {
    id,
    op: node_op,
    inputs: node_inputs,
    outputShapes: node_outputShapes,
    dtype: node_dtype,
    name: node_name,
    kind: "Constant",
    data: node_data
  };
  graph.nodes = graph.nodes.concat([node]);
  graph.nextId = graph.nextId + 1 | 0;
  return id;
}
function markOutputRef(graph, nodeRef) {
  graph.outputIds = graph.outputIds.concat([nodeRef]);
}
function markOutput(graph, nodeId) {
  markOutputRef(graph, {
    nodeId,
    outputIndex: 0
  });
}
function relu(graph, x) {
  return addNode(graph, "ReLU", [x], void 0);
}
function sigmoid(graph, x) {
  return addNode(graph, "Sigmoid", [x], void 0);
}
function tanh_(graph, x) {
  return addNode(graph, "Tanh", [x], void 0);
}
function gelu(graph, x) {
  return addNode(graph, "GeLU", [x], void 0);
}
function silu(graph, x) {
  return addNode(graph, "SiLU", [x], void 0);
}
function neg(graph, x) {
  return addNode(graph, "Neg", [x], void 0);
}
function exp_(graph, x) {
  return addNode(graph, "Exp", [x], void 0);
}
function log_(graph, x) {
  return addNode(graph, "Log", [x], void 0);
}
function sqrt_(graph, x) {
  return addNode(graph, "Sqrt", [x], void 0);
}
function abs_(graph, x) {
  return addNode(graph, "Abs", [x], void 0);
}
function sin_(graph, x) {
  return addNode(graph, "Sin", [x], void 0);
}
function cos_(graph, x) {
  return addNode(graph, "Cos", [x], void 0);
}
function add2(graph, a, b) {
  return addNode(graph, "Add", [
    a,
    b
  ], void 0);
}
function sub(graph, a, b) {
  return addNode(graph, "Sub", [
    a,
    b
  ], void 0);
}
function mul(graph, a, b) {
  return addNode(graph, "Mul", [
    a,
    b
  ], void 0);
}
function div2(graph, a, b) {
  return addNode(graph, "Div", [
    a,
    b
  ], void 0);
}
function pow_(graph, a, b) {
  return addNode(graph, "Pow", [
    a,
    b
  ], void 0);
}
function maximum2(graph, a, b) {
  return addNode(graph, "Maximum", [
    a,
    b
  ], void 0);
}
function minimum2(graph, a, b) {
  return addNode(graph, "Minimum", [
    a,
    b
  ], void 0);
}
function matmul(graph, a, b) {
  return addNode(graph, "MatMul", [
    a,
    b
  ], void 0);
}
function denseWithWeights(graph, x, weights, bias) {
  let out = matmul(graph, x, weights);
  if (bias !== void 0) {
    return add2(graph, out, bias);
  } else {
    return out;
  }
}
function dense(graph, x, units, name) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let inputDim = getOr(inputShape[inputShape.length - 1 | 0], 1);
  let w = weight(graph, [
    inputDim,
    units
  ], name + "_weight");
  let b = weight(graph, [units], name + "_bias");
  return denseWithWeights(graph, x, w, b);
}
function softmax(graph, x, axis) {
  return addNode(graph, {
    TAG: "Softmax",
    axis
  }, [x], void 0);
}
function reduce3(graph, x, op, axes, keepDims) {
  return addNode(graph, {
    TAG: "Reduce",
    op,
    axes,
    keepDims
  }, [x], void 0);
}
function reduceSum(graph, x, axes, keepDims) {
  return reduce3(graph, x, "Sum", axes, keepDims);
}
function reduceMean(graph, x, axes, keepDims) {
  return reduce3(graph, x, "Mean", axes, keepDims);
}
function reduceMax(graph, x, axes, keepDims) {
  return reduce3(graph, x, "Max", axes, keepDims);
}
function conv2dWithWeights(graph, x, weights, bias, stride, padding) {
  let weightsNode = graph.nodes.find((n) => n.id === weights);
  let weightsShape = weightsNode !== void 0 ? getOr(weightsNode.outputShapes[0], []) : [];
  let filters = getOr(weightsShape[3], 1);
  let kH = getOr(weightsShape[0], 1);
  let kW = getOr(weightsShape[1], 1);
  let conv = addNode(graph, {
    TAG: "Conv2D",
    filters,
    kernel: [
      kH,
      kW
    ],
    stride,
    padding,
    dilation: [
      1,
      1
    ],
    groups: 1
  }, [
    x,
    weights
  ], void 0);
  if (bias !== void 0) {
    return add2(graph, conv, bias);
  } else {
    return conv;
  }
}
function conv2d(graph, x, filters, kernelSize, stride, padding, name) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let inChannels = getOr(inputShape[3], 1);
  let w = weight(graph, [
    kernelSize,
    kernelSize,
    inChannels,
    filters
  ], name + "_weight");
  let b = weight(graph, [filters], name + "_bias");
  return conv2dWithWeights(graph, x, w, b, [
    stride,
    stride
  ], padding);
}
function maxPool2d(graph, x, size3, stride) {
  return addNode(graph, {
    TAG: "MaxPool2D",
    kernel: [
      size3,
      size3
    ],
    stride: [
      stride,
      stride
    ],
    padding: "Valid"
  }, [x], void 0);
}
function avgPool2d(graph, x, size3, stride) {
  return addNode(graph, {
    TAG: "AvgPool2D",
    kernel: [
      size3,
      size3
    ],
    stride: [
      stride,
      stride
    ],
    padding: "Valid",
    countIncludePad: true
  }, [x], void 0);
}
function globalAvgPool(graph, x) {
  return addNode(graph, "GlobalAvgPool", [x], void 0);
}
function globalMaxPool(graph, x) {
  return addNode(graph, "GlobalMaxPool", [x], void 0);
}
function batchNormWithParams(graph, x, gamma, beta, mean, variance, epsilon) {
  return addNode(graph, {
    TAG: "BatchNorm",
    epsilon,
    momentum: 0.1
  }, [
    x,
    gamma,
    beta,
    mean,
    variance
  ], void 0);
}
function layerNorm(graph, x, axes, epsilon) {
  return addNode(graph, {
    TAG: "LayerNorm",
    axes,
    epsilon
  }, [x], void 0);
}
function reshape(graph, x, targetShape) {
  return addNode(graph, {
    TAG: "Reshape",
    newShape: targetShape
  }, [x], void 0);
}
function flatten(graph, x) {
  return addNode(graph, {
    TAG: "Flatten",
    axis: 1
  }, [x], void 0);
}
function transpose(graph, x, permutation) {
  return addNode(graph, {
    TAG: "Transpose",
    perm: permutation
  }, [x], void 0);
}
function topk(graph, x, k, axis) {
  let nodeId = addNode(graph, {
    TAG: "TopK",
    k,
    axis,
    largest: true,
    sorted: true
  }, [x], void 0);
  return {
    nodeId,
    values: {
      nodeId,
      outputIndex: 0
    },
    indices: {
      nodeId,
      outputIndex: 1
    }
  };
}
function topkValues(graph, x, k, axis) {
  return topk(graph, x, k, axis).nodeId;
}
function split(graph, x, axis, splitSizes) {
  let nodeId = addNode(graph, {
    TAG: "Split",
    axis,
    splitSizes
  }, [x], void 0);
  let outputs = splitSizes.map((param, i) => ({
    nodeId,
    outputIndex: i
  }));
  return {
    nodeId,
    outputs
  };
}
function chunk(graph, x, axis, numChunks) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let axisSize = getOr(inputShape[axis], numChunks);
  let chunkSize = div(axisSize, numChunks);
  let splitSizes = make(numChunks, chunkSize);
  return split(graph, x, axis, splitSizes);
}
function where(graph, condition, xTrue, xFalse) {
  return addNode(graph, "Where", [
    condition,
    xTrue,
    xFalse
  ], void 0);
}
function gather(graph, data, indices, axis) {
  return addNode(graph, {
    TAG: "Gather",
    axis
  }, [
    data,
    indices
  ], void 0);
}
function clip(graph, x, minVal, maxVal) {
  return addNode(graph, {
    TAG: "Clip",
    min: minVal,
    max: maxVal
  }, [x], void 0);
}
function clamp(graph, x, minVal, maxVal) {
  return clip(graph, x, minVal, maxVal);
}
function topologicalSort(nodes) {
  let visitedIds = {};
  let resultArr = [];
  let visit = (nodeId) => {
    let key = nodeId.toString();
    if (!isNone(visitedIds[key])) {
      return;
    }
    visitedIds[key] = true;
    let nodeOpt = nodes.find((n) => n.id === nodeId);
    if (nodeOpt !== void 0) {
      nodeOpt.inputs.forEach((inputRef) => visit(inputRef.nodeId));
      resultArr.push(nodeOpt);
      return;
    }
  };
  nodes.forEach((node) => visit(node.id));
  return resultArr;
}
function allocateBuffers(sortedNodes, outputRefs) {
  let buffers = [];
  let bufferId = {
    contents: 0
  };
  sortedNodes.forEach((node) => {
    node.outputShapes.forEach((shape, outputIndex) => {
      let size3 = numElements(shape) << 2;
      let isGraphOutput = outputRefs.some((outRef) => {
        if (outRef.nodeId === node.id) {
          return outRef.outputIndex === outputIndex;
        } else {
          return false;
        }
      });
      let kind = isGraphOutput ? "Output" : node.kind;
      let buffer_id = bufferId.contents;
      let buffer_nodeId = node.id;
      let buffer_data = outputIndex === 0 ? node.data : void 0;
      let buffer = {
        id: buffer_id,
        size: size3,
        nodeId: buffer_nodeId,
        outputIndex,
        kind,
        data: buffer_data
      };
      buffers.push(buffer);
      bufferId.contents = bufferId.contents + 1 | 0;
    });
  });
  return buffers;
}
function findBufferId(buffers, nodeRef) {
  let buf = buffers.find((b) => {
    if (b.nodeId === nodeRef.nodeId) {
      return b.outputIndex === nodeRef.outputIndex;
    } else {
      return false;
    }
  });
  return getOr(map(buf, (b) => b.id), -1);
}
function compileNode(node, allNodes, buffers) {
  let inputShapes = node.inputs.map((inputRef) => {
    let inputNode = allNodes.find((n) => n.id === inputRef.nodeId);
    if (inputNode !== void 0) {
      return getOr(inputNode.outputShapes[inputRef.outputIndex], []);
    } else {
      return [];
    }
  });
  let match = node.kind;
  switch (match) {
    case "Intermediate":
    case "Output":
      break;
    default:
      return;
  }
  let result = generate(node.op, inputShapes);
  return map(result, (param) => {
    let inputBufferIds = node.inputs.map((inputRef) => findBufferId(buffers, inputRef));
    let outputBufferIds = filterMap(node.outputShapes.map((param2, i) => i), (outputIndex) => {
      let buf = buffers.find((b) => {
        if (b.nodeId === node.id) {
          return b.outputIndex === outputIndex;
        } else {
          return false;
        }
      });
      return map(buf, (b) => b.id);
    });
    return {
      nodeId: node.id,
      kernel: param[0],
      dispatch: param[1],
      inputBufferIds,
      outputBufferIds
    };
  });
}
function compile(graph) {
  let outputRefs;
  if (graph.outputIds.length !== 0) {
    outputRefs = graph.outputIds;
  } else {
    let usedAsInput = make2();
    graph.nodes.forEach((node) => {
      node.inputs.forEach((inputRef) => add(usedAsInput, inputRef.nodeId));
    });
    outputRefs = filterMap(graph.nodes, (node) => {
      if (!has2(usedAsInput, node.id) && node.kind !== "Input" && node.kind !== "Weight" && node.kind !== "Constant") {
        return {
          nodeId: node.id,
          outputIndex: 0
        };
      }
    });
  }
  let sorted = topologicalSort(graph.nodes);
  let buffers = allocateBuffers(sorted, outputRefs);
  let compiledOps = filterMap(sorted, (node) => compileNode(node, graph.nodes, buffers));
  let inputBufferIds = filterMap(buffers, (buf) => {
    if (buf.kind === "Input" && buf.outputIndex === 0) {
      return buf.id;
    }
  });
  let weightBufferIds = filterMap(buffers, (buf) => {
    if (buf.kind === "Weight" && buf.outputIndex === 0) {
      return buf.id;
    }
  });
  let constantBufferIds = filterMap(buffers, (buf) => {
    if (buf.kind === "Constant" && buf.outputIndex === 0) {
      return buf.id;
    }
  });
  let outputBufferIds = filterMap(buffers, (buf) => {
    if (buf.kind === "Output") {
      return buf.id;
    }
  });
  let weightNodes = sorted.filter((n) => n.kind === "Weight");
  let weightNames = weightNodes.map((n) => getOr(n.name, "unnamed"));
  let weightShapes = weightNodes.map((n) => getOr(n.outputShapes[0], []));
  let totalBufferSize = reduce(buffers, 0, (acc, buf) => acc + buf.size | 0);
  return {
    buffers,
    ops: compiledOps,
    inputBufferIds,
    weightBufferIds,
    constantBufferIds,
    outputBufferIds,
    totalBufferSize,
    weightNames,
    weightShapes
  };
}
function compileWithOutputs(graph, outputIds) {
  graph.outputIds = outputIds.map((id) => ({
    nodeId: id,
    outputIndex: 0
  }));
  return compile(graph);
}
function scale(graph, x, scalar) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let size3 = numElements(inputShape);
  let scalarData = make(size3, scalar);
  let scalarTensor = constant(graph, inputShape, "scale_const", scalarData);
  return mul(graph, x, scalarTensor);
}
function divByScalar(graph, x, scalar) {
  return scale(graph, x, 1 / scalar);
}
function addScalar(graph, x, scalar) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let size3 = numElements(inputShape);
  let scalarData = make(size3, scalar);
  let scalarTensor = constant(graph, inputShape, "add_const", scalarData);
  return add2(graph, x, scalarTensor);
}
function embedding(graph, indices, weights) {
  let weightsNode = graph.nodes.find((n) => n.id === weights);
  let weightsShape = weightsNode !== void 0 ? getOr(weightsNode.outputShapes[0], []) : [];
  let numEmbeddings = getOr(weightsShape[0], 0);
  let embeddingDim = getOr(weightsShape[1], 0);
  return addNode(graph, {
    TAG: "Embedding",
    numEmbeddings,
    embeddingDim
  }, [
    indices,
    weights
  ], void 0);
}
function layerNormWithParams(graph, x, gamma, beta, epsilon) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let lastAxis = inputShape.length - 1 | 0;
  return addNode(graph, {
    TAG: "LayerNorm",
    axes: [lastAxis],
    epsilon
  }, [
    x,
    gamma,
    beta
  ], void 0);
}
function concat(graph, inputs, axis) {
  return addNode(graph, {
    TAG: "Concat",
    axis
  }, inputs, void 0);
}
function scaledDotProductAttention(graph, query, key, value, scaleFactor) {
  let keyT = transpose(graph, key, [
    0,
    2,
    1
  ]);
  let scores = matmul(graph, query, keyT);
  let scaledScores = scale(graph, scores, scaleFactor);
  let attnWeights = softmax(graph, scaledScores, -1);
  return matmul(graph, attnWeights, value);
}
function multiHeadAttention(graph, query, key, value, _numHeads, headDim) {
  let scaleFactor = 1 / Math.sqrt(headDim);
  return scaledDotProductAttention(graph, query, key, value, scaleFactor);
}
var linear = dense;
function feedForward(graph, x, hiddenDim, outDim, name) {
  let h = linear(graph, x, hiddenDim, name + "_fc1");
  let a = gelu(graph, h);
  return linear(graph, a, outDim, name + "_fc2");
}
function transformerBlock(graph, x, numHeads, headDim, ffnDim, name) {
  let inputNode = graph.nodes.find((n) => n.id === x);
  let inputShape = inputNode !== void 0 ? getOr(inputNode.outputShapes[0], []) : [];
  let modelDim = getOr(inputShape[inputShape.length - 1 | 0], 64);
  let ln1Gamma = weight(graph, [modelDim], name + "_ln1_gamma");
  let ln1Beta = weight(graph, [modelDim], name + "_ln1_beta");
  let normed1 = layerNormWithParams(graph, x, ln1Gamma, ln1Beta, 1e-5);
  let q = linear(graph, normed1, numHeads * headDim | 0, name + "_q_proj");
  let k = linear(graph, normed1, numHeads * headDim | 0, name + "_k_proj");
  let v = linear(graph, normed1, numHeads * headDim | 0, name + "_v_proj");
  let attnOut = multiHeadAttention(graph, q, k, v, numHeads, headDim);
  let attnProj = linear(graph, attnOut, modelDim, name + "_out_proj");
  let res1 = add2(graph, x, attnProj);
  let ln2Gamma = weight(graph, [modelDim], name + "_ln2_gamma");
  let ln2Beta = weight(graph, [modelDim], name + "_ln2_beta");
  let normed2 = layerNormWithParams(graph, res1, ln2Gamma, ln2Beta, 1e-5);
  let ffnOut = feedForward(graph, normed2, ffnDim, modelDim, name + "_ffn");
  return add2(graph, res1, ffnOut);
}
function makeCausalMaskData(seqLen) {
  let size3 = seqLen * seqLen | 0;
  let mask = make(size3, 0);
  for (let i = 0; i < seqLen; ++i) {
    for (let j = 0; j < seqLen; ++j) {
      let idx = (i * seqLen | 0) + j | 0;
      if (j > i) {
        mask[idx] = -1e9;
      }
    }
  }
  return mask;
}
function causalMask(graph, seqLen) {
  let maskData = makeCausalMaskData(seqLen);
  return constant(graph, [
    seqLen,
    seqLen
  ], "causal_mask", maskData);
}
function maskedAttention(graph, query, key, value, mask, scaleFactor) {
  let keyT = transpose(graph, key, [
    0,
    2,
    1
  ]);
  let scores = matmul(graph, query, keyT);
  let scaledScores = scale(graph, scores, scaleFactor);
  let maskedScores = add2(graph, scaledScores, mask);
  let attnWeights = softmax(graph, maskedScores, -1);
  return matmul(graph, attnWeights, value);
}
function sinusoidalPositionalEncoding(graph, seqLen, dim) {
  let posData = fromInitializer(seqLen * dim | 0, (i) => {
    let pos = div(i, dim);
    let idx = mod_(i, dim);
    let divTerm = Math.pow(1e4, ((idx / 2 | 0) << 1) / dim);
    if (idx % 2 === 0) {
      return Math.sin(pos / divTerm);
    } else {
      return Math.cos(pos / divTerm);
    }
  });
  return constant(graph, [
    seqLen,
    dim
  ], "pos_encoding", posData);
}
function argmax(graph, x, axis) {
  return topk(graph, x, 1, axis);
}
function argmin(graph, x, axis) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let rank = xShape.length;
  let normAxis = axis < 0 ? rank + axis | 0 : axis;
  return addNode(graph, {
    TAG: "ArgMin",
    axis: normAxis,
    keepDims: false,
    selectLastIndex: false
  }, [x], void 0);
}
function pad(graph, x, pads, constantValue) {
  return addNode(graph, {
    TAG: "Pad",
    pads,
    mode: "Constant",
    constantValue
  }, [x], void 0);
}
function tile(graph, x, repeats) {
  return addNode(graph, {
    TAG: "Tile",
    repeats
  }, [x], void 0);
}
function slice_(graph, x, starts, ends, axes, steps) {
  return addNode(graph, {
    TAG: "Slice",
    starts,
    ends,
    axes,
    steps
  }, [x], void 0);
}
function oneHot(graph, x, depth) {
  return addNode(graph, {
    TAG: "OneHot",
    depth,
    axis: -1
  }, [x], void 0);
}
function scatter(graph, data, indices, updates, axis) {
  return addNode(graph, {
    TAG: "Scatter",
    axis
  }, [
    data,
    indices,
    updates
  ], void 0);
}
function cast(graph, x) {
  return addNode(graph, {
    TAG: "Cast",
    dtype: "F32"
  }, [x], void 0);
}
function cumsum(graph, x, axis) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let normAxis = axis < 0 ? xShape.length + axis | 0 : axis;
  return addNode(graph, {
    TAG: "CumSum",
    axis: normAxis,
    exclusive: false,
    reverse: false
  }, [x], void 0);
}
function squeeze(graph, x, axes) {
  return addNode(graph, {
    TAG: "Squeeze",
    axes
  }, [x], void 0);
}
function unsqueeze(graph, x, axes) {
  return addNode(graph, {
    TAG: "Unsqueeze",
    axes
  }, [x], void 0);
}
function expandDims(graph, x, axis) {
  return addNode(graph, {
    TAG: "ExpandDims",
    axis
  }, [x], void 0);
}
function broadcast2(graph, x, targetShape) {
  return addNode(graph, {
    TAG: "Broadcast",
    targetShape
  }, [x], void 0);
}
function stack(graph, inputs, axis) {
  return addNode(graph, {
    TAG: "Stack",
    axis
  }, inputs, void 0);
}
function cumprod(graph, x, axis) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let normAxis = axis < 0 ? xShape.length + axis | 0 : axis;
  return addNode(graph, {
    TAG: "CumProd",
    axis: normAxis,
    exclusive: false,
    reverse: false
  }, [x], void 0);
}
function reverse(graph, x, axes) {
  return addNode(graph, {
    TAG: "Reverse",
    axes
  }, [x], void 0);
}
function logSoftmax(graph, x, axis) {
  return addNode(graph, {
    TAG: "LogSoftmax",
    axis
  }, [x], void 0);
}
function sort(graph, x, axis, descendingOpt) {
  let descending = descendingOpt !== void 0 ? descendingOpt : false;
  return addNode(graph, {
    TAG: "Sort",
    axis,
    descending
  }, [x], void 0);
}
function arange(graph, size3, startOpt, stepOpt) {
  let start = startOpt !== void 0 ? startOpt : 0;
  let step = stepOpt !== void 0 ? stepOpt : 1;
  let data = fromInitializer(size3, (i) => start + i * step);
  return constant(graph, [size3], "arange", data);
}
function lstmCell(graph, x, hPrev, cPrev, hiddenSize, name) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let inputSize = getOr(xShape[xShape.length - 1 | 0], 1);
  let gateSize = hiddenSize << 2;
  let wIh = weight(graph, [
    gateSize,
    inputSize
  ], name + "_weight_ih");
  let wHh = weight(graph, [
    gateSize,
    hiddenSize
  ], name + "_weight_hh");
  let bIh = weight(graph, [gateSize], name + "_bias_ih");
  let bHh = weight(graph, [gateSize], name + "_bias_hh");
  let wIhT = transpose(graph, wIh, [
    1,
    0
  ]);
  let wHhT = transpose(graph, wHh, [
    1,
    0
  ]);
  let xW = matmul(graph, x, wIhT);
  let hW = matmul(graph, hPrev, wHhT);
  let gates1 = add2(graph, xW, hW);
  let gates2 = add2(graph, gates1, bIh);
  let gates = add2(graph, gates2, bHh);
  let iGatePre = slice_(graph, gates, [0], [hiddenSize], [1], [1]);
  let fGatePre = slice_(graph, gates, [hiddenSize], [hiddenSize << 1], [1], [1]);
  let gGatePre = slice_(graph, gates, [hiddenSize << 1], [3 * hiddenSize | 0], [1], [1]);
  let oGatePre = slice_(graph, gates, [3 * hiddenSize | 0], [hiddenSize << 2], [1], [1]);
  let iGate = sigmoid(graph, iGatePre);
  let fGate = sigmoid(graph, fGatePre);
  let gGate = tanh_(graph, gGatePre);
  let oGate = sigmoid(graph, oGatePre);
  let fc = mul(graph, fGate, cPrev);
  let ig = mul(graph, iGate, gGate);
  let cNew = add2(graph, fc, ig);
  let cTanh = tanh_(graph, cNew);
  let hNew = mul(graph, oGate, cTanh);
  return {
    h: hNew,
    c: cNew
  };
}
function gruCell(graph, x, hPrev, hiddenSize, name) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let inputSize = getOr(xShape[xShape.length - 1 | 0], 1);
  let gateSize = 3 * hiddenSize | 0;
  let wIh = weight(graph, [
    gateSize,
    inputSize
  ], name + "_weight_ih");
  let wHh = weight(graph, [
    gateSize,
    hiddenSize
  ], name + "_weight_hh");
  let bIh = weight(graph, [gateSize], name + "_bias_ih");
  let bHh = weight(graph, [gateSize], name + "_bias_hh");
  let wIhT = transpose(graph, wIh, [
    1,
    0
  ]);
  let wHhT = transpose(graph, wHh, [
    1,
    0
  ]);
  let xW = matmul(graph, x, wIhT);
  let hW = matmul(graph, hPrev, wHhT);
  let xR = slice_(graph, xW, [0], [hiddenSize], [1], [1]);
  let xZ = slice_(graph, xW, [hiddenSize], [hiddenSize << 1], [1], [1]);
  let xN = slice_(graph, xW, [hiddenSize << 1], [3 * hiddenSize | 0], [1], [1]);
  let hR = slice_(graph, hW, [0], [hiddenSize], [1], [1]);
  let hZ = slice_(graph, hW, [hiddenSize], [hiddenSize << 1], [1], [1]);
  let hN = slice_(graph, hW, [hiddenSize << 1], [3 * hiddenSize | 0], [1], [1]);
  let bIhR = slice_(graph, bIh, [0], [hiddenSize], [0], [1]);
  let bIhZ = slice_(graph, bIh, [hiddenSize], [hiddenSize << 1], [0], [1]);
  let bIhN = slice_(graph, bIh, [hiddenSize << 1], [3 * hiddenSize | 0], [0], [1]);
  let bHhR = slice_(graph, bHh, [0], [hiddenSize], [0], [1]);
  let bHhZ = slice_(graph, bHh, [hiddenSize], [hiddenSize << 1], [0], [1]);
  let bHhN = slice_(graph, bHh, [hiddenSize << 1], [3 * hiddenSize | 0], [0], [1]);
  let rPre1 = add2(graph, xR, hR);
  let rPre2 = add2(graph, rPre1, bIhR);
  let rPre3 = add2(graph, rPre2, bHhR);
  let rGate = sigmoid(graph, rPre3);
  let zPre1 = add2(graph, xZ, hZ);
  let zPre2 = add2(graph, zPre1, bIhZ);
  let zPre3 = add2(graph, zPre2, bHhZ);
  let zGate = sigmoid(graph, zPre3);
  let hNBias = add2(graph, hN, bHhN);
  let rH = mul(graph, rGate, hNBias);
  let nPre1 = add2(graph, xN, bIhN);
  let nPre2 = add2(graph, nPre1, rH);
  let nGate = tanh_(graph, nPre2);
  let ones = constant(graph, [hiddenSize], "ones", fromInitializer(hiddenSize, (param) => 1));
  let oneMinusZ = sub(graph, ones, zGate);
  let term1 = mul(graph, oneMinusZ, nGate);
  let term2 = mul(graph, zGate, hPrev);
  return add2(graph, term1, term2);
}
function lstm(graph, x, hiddenSize, name) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let batchSize = getOr(xShape[0], 1);
  let seqLen = getOr(xShape[1], 1);
  let inputSize = getOr(xShape[2], 1);
  let h0 = constant(graph, [
    batchSize,
    hiddenSize
  ], name + "_h0", fromInitializer(batchSize * hiddenSize | 0, (param) => 0));
  let c0 = constant(graph, [
    batchSize,
    hiddenSize
  ], name + "_c0", fromInitializer(batchSize * hiddenSize | 0, (param) => 0));
  let outputs = [];
  let h = h0;
  let c = c0;
  for (let t = 0; t < seqLen; ++t) {
    let xt = slice_(graph, x, [
      0,
      t,
      0
    ], [
      batchSize,
      t + 1 | 0,
      inputSize
    ], [
      0,
      1,
      2
    ], [
      1,
      1,
      1
    ]);
    let xtFlat = reshape(graph, xt, [
      batchSize,
      inputSize
    ]);
    let result = lstmCell(graph, xtFlat, h, c, hiddenSize, name + "_t" + t.toString());
    h = result.h;
    c = result.c;
    let hExp = reshape(graph, result.h, [
      batchSize,
      1,
      hiddenSize
    ]);
    outputs = outputs.concat([hExp]);
  }
  if (outputs.length === 1) {
    return getOr(outputs[0], h0);
  }
  let result$1 = getOr(outputs[0], h0);
  for (let i = 1, i_finish = outputs.length; i < i_finish; ++i) {
    result$1 = concat(graph, [
      result$1,
      getOr(outputs[i], h0)
    ], 1);
  }
  return result$1;
}
function gru(graph, x, hiddenSize, name) {
  let node = graph.nodes.find((n) => n.id === x);
  let xShape = node !== void 0 ? getOr(node.outputShapes[0], []) : [];
  let batchSize = getOr(xShape[0], 1);
  let seqLen = getOr(xShape[1], 1);
  let inputSize = getOr(xShape[2], 1);
  let h0 = constant(graph, [
    batchSize,
    hiddenSize
  ], name + "_h0", fromInitializer(batchSize * hiddenSize | 0, (param) => 0));
  let outputs = [];
  let h = h0;
  for (let t = 0; t < seqLen; ++t) {
    let xt = slice_(graph, x, [
      0,
      t,
      0
    ], [
      batchSize,
      t + 1 | 0,
      inputSize
    ], [
      0,
      1,
      2
    ], [
      1,
      1,
      1
    ]);
    let xtFlat = reshape(graph, xt, [
      batchSize,
      inputSize
    ]);
    let hNew = gruCell(graph, xtFlat, h, hiddenSize, name + "_t" + t.toString());
    h = hNew;
    let hExp = reshape(graph, hNew, [
      batchSize,
      1,
      hiddenSize
    ]);
    outputs = outputs.concat([hExp]);
  }
  if (outputs.length === 1) {
    return getOr(outputs[0], h0);
  }
  let result = getOr(outputs[0], h0);
  for (let i = 1, i_finish = outputs.length; i < i_finish; ++i) {
    result = concat(graph, [
      result,
      getOr(outputs[i], h0)
    ], 1);
  }
  return result;
}
function tensorKindToString(kind) {
  switch (kind) {
    case "Input":
      return "Input";
    case "Weight":
      return "Weight";
    case "Constant":
      return "Constant";
    case "Intermediate":
      return "Intermediate";
    case "Output":
      return "Output";
  }
}
function printGraph(graph) {
  console.log("=== Computation Graph ===");
  console.log("Nodes: " + graph.nodes.length.toString());
  graph.nodes.forEach((node) => {
    let name = getOr(node.name, "unnamed");
    let kind = tensorKindToString(node.kind);
    let shapesStr = node.outputShapes.map((s) => "[" + s.map((d) => d.toString()).join(", ") + "]").join(", ");
    let inputsStr = "[" + node.inputs.map((r) => r.nodeId.toString() + ":" + r.outputIndex.toString()).join(", ") + "]";
    console.log("  Node " + node.id.toString() + " (" + kind + "): " + name + " outputs=" + shapesStr + " inputs=" + inputsStr);
  });
}
function printCompiled(compiled) {
  console.log("\n=== Compiled Graph ===");
  console.log("Buffers: " + compiled.buffers.length.toString());
  console.log("Ops: " + compiled.ops.length.toString());
  console.log("Total buffer size: " + compiled.totalBufferSize.toString() + " bytes");
  console.log("\nInputs: [" + compiled.inputBufferIds.map((i) => i.toString()).join(", ") + "]");
  console.log("Weights: [" + compiled.weightBufferIds.map((i) => i.toString()).join(", ") + "]");
  console.log("Constants: [" + compiled.constantBufferIds.map((i) => i.toString()).join(", ") + "]");
  console.log("Outputs: [" + compiled.outputBufferIds.map((i) => i.toString()).join(", ") + "]");
  console.log("\nWeight names: [" + compiled.weightNames.join(", ") + "]");
  console.log("\nOps:");
  compiled.ops.forEach((op) => {
    let inputsStr = "[" + op.inputBufferIds.map((i) => i.toString()).join(", ") + "]";
    let outputsStr = "[" + op.outputBufferIds.map((i) => i.toString()).join(", ") + "]";
    console.log("  " + op.kernel.name + ": inputs=" + inputsStr + " outputs=" + outputsStr);
  });
}

// src/Autograd.res.mjs
function storageBufferRO(binding2, name) {
  return `@group(0) @binding(` + binding2.toString() + `) var<storage, read> ` + name + `: array<f32>;`;
}
function storageBufferRW(binding2, name) {
  return `@group(0) @binding(` + binding2.toString() + `) var<storage, read_write> ` + name + `: array<f32>;`;
}
var mainSignature2 = `@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;`;
var mainEnd2 = "}";
function genGradReduceKernel(gradShape, targetShape) {
  let gradSize = numElements(gradShape);
  let targetSize = numElements(targetShape);
  let gradRank = gradShape.length;
  let targetRank = targetShape.length;
  let maxRank = max(gradRank, targetRank);
  let paddedGrad = make(maxRank - gradRank | 0, 1).concat(gradShape);
  let paddedTarget = make(maxRank - targetRank | 0, 1).concat(targetShape);
  paddedTarget.map((dim, i) => [
    dim,
    i
  ]).filter((elem, _idx) => {
    if (elem[0] === 1) {
      return getOr(paddedGrad[elem[1]], 1) > 1;
    } else {
      return false;
    }
  }).map((param) => param[1]);
  let gradStrides = fromInitializer(maxRank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < maxRank; ++j) {
      stride = stride * getOr(paddedGrad[j], 1) | 0;
    }
    return stride;
  });
  let targetStrides = fromInitializer(maxRank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < maxRank; ++j) {
      stride = stride * getOr(paddedTarget[j], 1) | 0;
    }
    return stride;
  });
  let gradShapeStr = paddedGrad.map((d) => d.toString()).join(", ");
  let targetShapeStr = paddedTarget.map((d) => d.toString()).join(", ");
  let gradStridesStr = gradStrides.map((d) => d.toString()).join(", ");
  let targetStridesStr = targetStrides.map((d) => d.toString()).join(", ");
  let wgsl = targetSize === gradSize ? storageBufferRO(0, "grad_in") + `
` + storageBufferRW(1, "grad_out") + `
` + mainSignature2 + `
  if (idx >= ` + targetSize.toString() + `u) { return; }
  grad_out[idx] = grad_in[idx];
` + mainEnd2 : storageBufferRO(0, "grad_in") + `
` + storageBufferRW(1, "grad_out") + `
const RANK = ` + maxRank.toString() + `u;
const GRAD_SIZE = ` + gradSize.toString() + `u;
const TARGET_SIZE = ` + targetSize.toString() + `u;
const GRAD_SHAPE = array<u32, ` + maxRank.toString() + `>(` + gradShapeStr + `);
const TARGET_SHAPE = array<u32, ` + maxRank.toString() + `>(` + targetShapeStr + `);
const GRAD_STRIDES = array<u32, ` + maxRank.toString() + `>(` + gradStridesStr + `);
const TARGET_STRIDES = array<u32, ` + maxRank.toString() + `>(` + targetStridesStr + `);
` + mainSignature2 + `
  if (idx >= TARGET_SIZE) { return; }
  
  // Convert target idx to coordinates
  var target_coords: array<u32, ` + maxRank.toString() + `>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    target_coords[d] = remaining / TARGET_STRIDES[d];
    remaining = remaining % TARGET_STRIDES[d];
  }
  
  // Sum over all grad elements that map to this target element
  var sum = 0.0;
  for (var g = 0u; g < GRAD_SIZE; g = g + 1u) {
    // Convert grad idx to coordinates
    var grad_coords: array<u32, ` + maxRank.toString() + `>;
    var rem = g;
    for (var d = 0u; d < RANK; d = d + 1u) {
      grad_coords[d] = rem / GRAD_STRIDES[d];
      rem = rem % GRAD_STRIDES[d];
    }
    
    // Check if this grad element maps to our target element
    var matches = true;
    for (var d = 0u; d < RANK; d = d + 1u) {
      if (TARGET_SHAPE[d] > 1u && grad_coords[d] != target_coords[d]) {
        matches = false;
        break;
      }
    }
    
    if (matches) {
      sum = sum + grad_in[g];
    }
  }
  
  grad_out[idx] = sum;
` + mainEnd2;
  return {
    name: "grad_reduce_" + gradSize.toString() + "_to_" + targetSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: gradSize << 2,
        usage: "ReadOnly",
        name: "grad_in"
      },
      {
        binding: 1,
        size: targetSize << 2,
        usage: "ReadWrite",
        name: "grad_out"
      }
    ]
  };
}
function genNegBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = -grad_out[idx];
` + mainEnd2;
  return {
    name: "neg_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genAbsBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = grad_out[idx] * sign(x[idx]);
` + mainEnd2;
  return {
    name: "abs_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genSqrtBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "out") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let out_val = out[idx];
  grad_x[idx] = select(0.0, grad_out[idx] * 0.5 / out_val, out_val > 0.0);
` + mainEnd2;
  return {
    name: "sqrt_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "out"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genExpBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "out") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = grad_out[idx] * out[idx];
` + mainEnd2;
  return {
    name: "exp_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "out"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genLogBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = grad_out[idx] / x[idx];
` + mainEnd2;
  return {
    name: "log_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genSinBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = grad_out[idx] * cos(x[idx]);
` + mainEnd2;
  return {
    name: "sin_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genCosBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = -grad_out[idx] * sin(x[idx]);
` + mainEnd2;
  return {
    name: "cos_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genTanhBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "out") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let t = out[idx];
  grad_x[idx] = grad_out[idx] * (1.0 - t * t);
` + mainEnd2;
  return {
    name: "tanh_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "out"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genSigmoidBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "out") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let s = out[idx];
  grad_x[idx] = grad_out[idx] * s * (1.0 - s);
` + mainEnd2;
  return {
    name: "sigmoid_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "out"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genReLUBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = select(0.0, grad_out[idx], x[idx] > 0.0);
` + mainEnd2;
  return {
    name: "relu_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genLeakyReLUBackwardKernel(size3, alpha) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  grad_x[idx] = select(` + alpha.toString() + ` * g, g, x[idx] > 0.0);
` + mainEnd2;
  return {
    name: "leaky_relu_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genGeLUBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRW(2, "grad_x") + `
const SQRT_2 = 1.4142135623730951;
const SQRT_2_PI = 0.7978845608028654;
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let x_val = x[idx];
  let cdf = 0.5 * (1.0 + tanh(SQRT_2 / 2.0 * (x_val + 0.044715 * x_val * x_val * x_val)));
  let pdf = SQRT_2_PI * exp(-0.5 * x_val * x_val);
  grad_x[idx] = grad_out[idx] * (cdf + x_val * pdf);
` + mainEnd2;
  return {
    name: "gelu_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genAddBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_a") + `
` + storageBufferRW(2, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g;
  grad_b[idx] = g;
` + mainEnd2;
  return {
    name: "add_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genSubBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_a") + `
` + storageBufferRW(2, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g;
  grad_b[idx] = -g;
` + mainEnd2;
  return {
    name: "sub_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genMulBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRO(2, "b") + `
` + storageBufferRW(3, "grad_a") + `
` + storageBufferRW(4, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g * b[idx];
  grad_b[idx] = g * a[idx];
` + mainEnd2;
  return {
    name: "mul_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 4,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genDivBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRO(2, "b") + `
` + storageBufferRW(3, "grad_a") + `
` + storageBufferRW(4, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  let b_val = b[idx];
  grad_a[idx] = g / b_val;
  grad_b[idx] = -g * a[idx] / (b_val * b_val);
` + mainEnd2;
  return {
    name: "div_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 4,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genPowBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRO(2, "b") + `
` + storageBufferRO(3, "out") + `
` + storageBufferRW(4, "grad_a") + `
` + storageBufferRW(5, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  let out_val = out[idx];
  // dL/da = g * b * a^(b-1) = g * b * out / a
  grad_a[idx] = select(0.0, g * b_val * out_val / a_val, a_val != 0.0);
  // dL/db = g * a^b * ln(a) = g * out * ln(a)
  grad_b[idx] = select(0.0, g * out_val * log(a_val), a_val > 0.0);
` + mainEnd2;
  return {
    name: "pow_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "out"
      },
      {
        binding: 4,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 5,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genMaximumBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRO(2, "b") + `
` + storageBufferRW(3, "grad_a") + `
` + storageBufferRW(4, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  if (a_val > b_val) {
    grad_a[idx] = g;
    grad_b[idx] = 0.0;
  } else if (b_val > a_val) {
    grad_a[idx] = 0.0;
    grad_b[idx] = g;
  } else {
    // Equal: split gradient
    grad_a[idx] = g * 0.5;
    grad_b[idx] = g * 0.5;
  }
` + mainEnd2;
  return {
    name: "maximum_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 4,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genMinimumBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRO(2, "b") + `
` + storageBufferRW(3, "grad_a") + `
` + storageBufferRW(4, "grad_b") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  if (a_val < b_val) {
    grad_a[idx] = g;
    grad_b[idx] = 0.0;
  } else if (b_val < a_val) {
    grad_a[idx] = 0.0;
    grad_b[idx] = g;
  } else {
    grad_a[idx] = g * 0.5;
    grad_b[idx] = g * 0.5;
  }
` + mainEnd2;
  return {
    name: "minimum_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_a"
      },
      {
        binding: 4,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genMatMulBackwardAKernel(m, k, n) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "b") + `
` + storageBufferRW(2, "grad_a") + `
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= M || col >= K) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < N; i = i + 1u) {
    // grad_out[row, i] * b[col, i] (B transposed: b[k, n] -> b^T[n, k])
    sum = sum + grad_out[row * N + i] * b[col * N + i];
  }
  grad_a[row * K + col] = sum;
}`;
  return {
    name: "matmul_backward_a_" + m.toString() + "x" + k.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (m * n | 0) << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: (k * n | 0) << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 2,
        size: (m * k | 0) << 2,
        usage: "ReadWrite",
        name: "grad_a"
      }
    ]
  };
}
function genMatMulBackwardBKernel(m, k, n) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRW(2, "grad_b") + `
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= K || col >= N) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < M; i = i + 1u) {
    // a[i, row] (A transposed: a[m, k] -> a^T[k, m]) * grad_out[i, col]
    sum = sum + a[i * K + row] * grad_out[i * N + col];
  }
  grad_b[row * N + col] = sum;
}`;
  return {
    name: "matmul_backward_b_" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: (m * n | 0) << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: (m * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: (k * n | 0) << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genBatchedMatMulBackwardAKernel(batch, m, k, n) {
  let totalOutput = (batch * m | 0) * k | 0;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "b") + `
` + storageBufferRW(2, "grad_a") + `
const BATCH = ` + batch.toString() + `u;
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
` + mainSignature2 + `
  if (idx >= ` + totalOutput.toString() + `u) { return; }
  
  let batch_idx = idx / (M * K);
  let matrix_idx = idx % (M * K);
  let row = matrix_idx / K;
  let col = matrix_idx % K;
  
  var sum = 0.0;
  let grad_base = batch_idx * M * N;
  let b_base = batch_idx * K * N;
  
  for (var i = 0u; i < N; i = i + 1u) {
    sum = sum + grad_out[grad_base + row * N + i] * b[b_base + col * N + i];
  }
  grad_a[idx] = sum;
` + mainEnd2;
  return {
    name: "batched_matmul_backward_a_" + batch.toString() + "_" + m.toString() + "x" + k.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: ((batch * m | 0) * n | 0) << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: ((batch * k | 0) * n | 0) << 2,
        usage: "ReadOnly",
        name: "b"
      },
      {
        binding: 2,
        size: ((batch * m | 0) * k | 0) << 2,
        usage: "ReadWrite",
        name: "grad_a"
      }
    ]
  };
}
function genBatchedMatMulBackwardBKernel(batch, m, k, n) {
  let totalOutput = (batch * k | 0) * n | 0;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "a") + `
` + storageBufferRW(2, "grad_b") + `
const BATCH = ` + batch.toString() + `u;
const M = ` + m.toString() + `u;
const K = ` + k.toString() + `u;
const N = ` + n.toString() + `u;
` + mainSignature2 + `
  if (idx >= ` + totalOutput.toString() + `u) { return; }
  
  let batch_idx = idx / (K * N);
  let matrix_idx = idx % (K * N);
  let row = matrix_idx / N;
  let col = matrix_idx % N;
  
  var sum = 0.0;
  let grad_base = batch_idx * M * N;
  let a_base = batch_idx * M * K;
  
  for (var i = 0u; i < M; i = i + 1u) {
    sum = sum + a[a_base + i * K + row] * grad_out[grad_base + i * N + col];
  }
  grad_b[idx] = sum;
` + mainEnd2;
  return {
    name: "batched_matmul_backward_b_" + batch.toString() + "_" + k.toString() + "x" + n.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: ((batch * m | 0) * n | 0) << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: ((batch * m | 0) * k | 0) << 2,
        usage: "ReadOnly",
        name: "a"
      },
      {
        binding: 2,
        size: ((batch * k | 0) * n | 0) << 2,
        usage: "ReadWrite",
        name: "grad_b"
      }
    ]
  };
}
function genSumBackwardKernel(inputShape, outputShape, axes) {
  let inputSize = numElements(inputShape);
  let outputSize = numElements(outputShape);
  let rank = inputShape.length;
  let inputStrides = fromInitializer(rank, (i) => {
    let stride2 = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride2 = stride2 * getOr(inputShape[j], 1) | 0;
    }
    return stride2;
  });
  let normAxes = axes.map((a) => {
    if (a < 0) {
      return rank + a | 0;
    } else {
      return a;
    }
  });
  let strides = [];
  let stride = 1;
  for (let i = rank - 1 | 0; i >= 0; --i) {
    if (!normAxes.includes(i)) {
      strides = [stride].concat(strides);
    }
    stride = stride * getOr(inputShape[i], 1) | 0;
  }
  let inputShapeStr = inputShape.map((d) => d.toString()).join(", ");
  let inputStridesStr = inputStrides.map((d) => d.toString()).join(", ");
  let axesStr = normAxes.map((d) => d.toString()).join(", ");
  let numAxes = normAxes.length;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_x") + `
const RANK = ` + rank.toString() + `u;
const INPUT_SIZE = ` + inputSize.toString() + `u;
const OUTPUT_SIZE = ` + outputSize.toString() + `u;
const NUM_AXES = ` + numAxes.toString() + `u;
const INPUT_SHAPE = array<u32, ` + rank.toString() + `>(` + inputShapeStr + `);
const INPUT_STRIDES = array<u32, ` + rank.toString() + `>(` + inputStridesStr + `);
const AXES = array<u32, ` + numAxes.toString() + `>(` + axesStr + `);

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

` + mainSignature2 + `
  if (idx >= INPUT_SIZE) { return; }
  
  // Convert flat idx to coordinates
  var coords: array<u32, ` + rank.toString() + `>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / INPUT_STRIDES[d];
    remaining = remaining % INPUT_STRIDES[d];
  }
  
  // Compute output index (skip reduced dimensions)
  var out_idx = 0u;
  var out_stride = 1u;
  for (var d = i32(RANK) - 1; d >= 0; d = d - 1) {
    if (!isReduceAxis(u32(d))) {
      out_idx = out_idx + coords[d] * out_stride;
      out_stride = out_stride * INPUT_SHAPE[d];
    }
  }
  
  grad_x[idx] = grad_out[out_idx];
` + mainEnd2;
  return {
    name: "sum_backward_" + inputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: inputSize << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genMeanBackwardKernel(inputShape, outputShape, axes) {
  let inputSize = numElements(inputShape);
  let outputSize = numElements(outputShape);
  let rank = inputShape.length;
  let normAxes = axes.map((a) => {
    if (a < 0) {
      return rank + a | 0;
    } else {
      return a;
    }
  });
  let reduceCount = reduce(normAxes, 1, (acc, axis) => acc * getOr(inputShape[axis], 1) | 0);
  let inputStrides = fromInitializer(rank, (i) => {
    let stride = 1;
    for (let j = i + 1 | 0; j < rank; ++j) {
      stride = stride * getOr(inputShape[j], 1) | 0;
    }
    return stride;
  });
  let inputShapeStr = inputShape.map((d) => d.toString()).join(", ");
  let inputStridesStr = inputStrides.map((d) => d.toString()).join(", ");
  let axesStr = normAxes.map((d) => d.toString()).join(", ");
  let numAxes = normAxes.length;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_x") + `
const RANK = ` + rank.toString() + `u;
const INPUT_SIZE = ` + inputSize.toString() + `u;
const REDUCE_COUNT = ` + reduceCount.toString() + `;
const NUM_AXES = ` + numAxes.toString() + `u;
const INPUT_SHAPE = array<u32, ` + rank.toString() + `>(` + inputShapeStr + `);
const INPUT_STRIDES = array<u32, ` + rank.toString() + `>(` + inputStridesStr + `);
const AXES = array<u32, ` + numAxes.toString() + `>(` + axesStr + `);

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

` + mainSignature2 + `
  if (idx >= INPUT_SIZE) { return; }
  
  var coords: array<u32, ` + rank.toString() + `>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / INPUT_STRIDES[d];
    remaining = remaining % INPUT_STRIDES[d];
  }
  
  var out_idx = 0u;
  var out_stride = 1u;
  for (var d = i32(RANK) - 1; d >= 0; d = d - 1) {
    if (!isReduceAxis(u32(d))) {
      out_idx = out_idx + coords[d] * out_stride;
      out_stride = out_stride * INPUT_SHAPE[d];
    }
  }
  
  grad_x[idx] = grad_out[out_idx] / REDUCE_COUNT;
` + mainEnd2;
  return {
    name: "mean_backward_" + inputSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: outputSize << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: inputSize << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genSoftmaxBackwardKernel(outerSize, axisSize) {
  let totalSize = outerSize * axisSize | 0;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "softmax_out") + `
` + storageBufferRW(2, "grad_x") + `
const OUTER_SIZE = ` + outerSize.toString() + `u;
const AXIS_SIZE = ` + axisSize.toString() + `u;
` + mainSignature2 + `
  let outer_idx = idx;
  if (outer_idx >= OUTER_SIZE) { return; }
  
  let base = outer_idx * AXIS_SIZE;
  
  // Compute dot product: sum_j(grad_out[j] * softmax[j])
  var dot = 0.0;
  for (var j = 0u; j < AXIS_SIZE; j = j + 1u) {
    dot = dot + grad_out[base + j] * softmax_out[base + j];
  }
  
  // grad_x[i] = softmax[i] * (grad_out[i] - dot)
  for (var i = 0u; i < AXIS_SIZE; i = i + 1u) {
    let s = softmax_out[base + i];
    grad_x[base + i] = s * (grad_out[base + i] - dot);
  }
` + mainEnd2;
  return {
    name: "softmax_backward_" + outerSize.toString() + "x" + axisSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: totalSize << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: totalSize << 2,
        usage: "ReadOnly",
        name: "softmax_out"
      },
      {
        binding: 2,
        size: totalSize << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genLayerNormBackwardKernel(outerSize, normSize, epsilon) {
  let totalSize = outerSize * normSize | 0;
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRO(1, "x") + `
` + storageBufferRO(2, "gamma") + `
` + storageBufferRW(3, "grad_x") + `
` + storageBufferRW(4, "grad_gamma") + `
` + storageBufferRW(5, "grad_beta") + `
const OUTER = ` + outerSize.toString() + `u;
const NORM = ` + normSize.toString() + `u;
const EPSILON = ` + epsilon.toString() + `;
` + mainSignature2 + `
  let outer_idx = idx;
  if (outer_idx >= OUTER) { return; }
  
  let base = outer_idx * NORM;
  
  // Recompute mean and variance
  var mean = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    mean = mean + x[base + i];
  }
  mean = mean / f32(NORM);
  
  var variance = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let diff = x[base + i] - mean;
    variance = variance + diff * diff;
  }
  variance = variance / f32(NORM);
  let inv_std = 1.0 / sqrt(variance + EPSILON);
  
  // Compute intermediate sums for gradient
  var sum_grad_out = 0.0;
  var sum_grad_out_x_centered = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let g = grad_out[base + i] * gamma[i];
    sum_grad_out = sum_grad_out + g;
    sum_grad_out_x_centered = sum_grad_out_x_centered + g * (x[base + i] - mean);
  }
  
  // Compute grad_x
  let norm_factor = 1.0 / f32(NORM);
  for (var i = 0u; i < NORM; i = i + 1u) {
    let x_centered = x[base + i] - mean;
    let x_norm = x_centered * inv_std;
    let g = grad_out[base + i] * gamma[i];
    grad_x[base + i] = inv_std * (g - norm_factor * sum_grad_out - norm_factor * x_norm * sum_grad_out_x_centered * inv_std);
  }
  
  // Accumulate grad_gamma and grad_beta (atomic would be better but this works for single outer)
  if (outer_idx == 0u) {
    for (var i = 0u; i < NORM; i = i + 1u) {
      var gg = 0.0;
      var gb = 0.0;
      for (var o = 0u; o < OUTER; o = o + 1u) {
        let b = o * NORM;
        var m = 0.0;
        for (var j = 0u; j < NORM; j = j + 1u) { m = m + x[b + j]; }
        m = m / f32(NORM);
        var v = 0.0;
        for (var j = 0u; j < NORM; j = j + 1u) { let d = x[b + j] - m; v = v + d * d; }
        v = v / f32(NORM);
        let istd = 1.0 / sqrt(v + EPSILON);
        let x_norm = (x[b + i] - m) * istd;
        gg = gg + grad_out[b + i] * x_norm;
        gb = gb + grad_out[b + i];
      }
      grad_gamma[i] = gg;
      grad_beta[i] = gb;
    }
  }
` + mainEnd2;
  return {
    name: "layernorm_backward_" + outerSize.toString() + "x" + normSize.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: totalSize << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: totalSize << 2,
        usage: "ReadOnly",
        name: "x"
      },
      {
        binding: 2,
        size: normSize << 2,
        usage: "ReadOnly",
        name: "gamma"
      },
      {
        binding: 3,
        size: totalSize << 2,
        usage: "ReadWrite",
        name: "grad_x"
      },
      {
        binding: 4,
        size: normSize << 2,
        usage: "ReadWrite",
        name: "grad_gamma"
      },
      {
        binding: 5,
        size: normSize << 2,
        usage: "ReadWrite",
        name: "grad_beta"
      }
    ]
  };
}
function genCopyBackwardKernel(size3) {
  let wgsl = storageBufferRO(0, "grad_out") + `
` + storageBufferRW(1, "grad_x") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_x[idx] = grad_out[idx];
` + mainEnd2;
  return {
    name: "copy_backward_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_out"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_x"
      }
    ]
  };
}
function genGradAccumulateKernel(size3) {
  let wgsl = storageBufferRW(0, "grad_acc") + `
` + storageBufferRO(1, "grad_new") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad_acc[idx] = grad_acc[idx] + grad_new[idx];
` + mainEnd2;
  return {
    name: "grad_accumulate_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "grad_acc"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad_new"
      }
    ]
  };
}
function genGradZeroKernel(size3) {
  let wgsl = storageBufferRW(0, "grad") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  grad[idx] = 0.0;
` + mainEnd2;
  return {
    name: "grad_zero_" + size3.toString(),
    wgsl,
    bindings: [{
      binding: 0,
      size: size3 << 2,
      usage: "ReadWrite",
      name: "grad"
    }]
  };
}
function genSGDKernel(size3, lr) {
  let wgsl = storageBufferRW(0, "param") + `
` + storageBufferRO(1, "grad") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  param[idx] = param[idx] - ` + lr.toString() + ` * grad[idx];
` + mainEnd2;
  return {
    name: "sgd_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "param"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad"
      }
    ]
  };
}
function genSGDMomentumKernel(size3, lr, momentum) {
  let wgsl = storageBufferRW(0, "param") + `
` + storageBufferRO(1, "grad") + `
` + storageBufferRW(2, "velocity") + `
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  let v = ` + momentum.toString() + ` * velocity[idx] + grad[idx];
  velocity[idx] = v;
  param[idx] = param[idx] - ` + lr.toString() + ` * v;
` + mainEnd2;
  return {
    name: "sgd_momentum_" + size3.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "param"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "velocity"
      }
    ]
  };
}
function genAdamKernel(size3, lr, beta1, beta2, epsilon, t) {
  let beta1_t = Math.pow(beta1, t);
  let beta2_t = Math.pow(beta2, t);
  let lr_t = lr * Math.sqrt(1 - beta2_t) / (1 - beta1_t);
  let wgsl = storageBufferRW(0, "param") + `
` + storageBufferRO(1, "grad") + `
` + storageBufferRW(2, "m") + `
` + storageBufferRW(3, "v") + `
const LR_T = ` + lr_t.toString() + `;
const BETA1 = ` + beta1.toString() + `;
const BETA2 = ` + beta2.toString() + `;
const EPSILON = ` + epsilon.toString() + `;
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  
  let g = grad[idx];
  
  // Update biased first moment estimate
  let m_new = BETA1 * m[idx] + (1.0 - BETA1) * g;
  m[idx] = m_new;
  
  // Update biased second raw moment estimate  
  let v_new = BETA2 * v[idx] + (1.0 - BETA2) * g * g;
  v[idx] = v_new;
  
  // Update parameters (bias correction already in LR_T)
  param[idx] = param[idx] - LR_T * m_new / (sqrt(v_new) + EPSILON);
` + mainEnd2;
  return {
    name: "adam_" + size3.toString() + "_t" + t.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "param"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "m"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "v"
      }
    ]
  };
}
function genAdamWKernel(size3, lr, beta1, beta2, epsilon, weightDecay, t) {
  let beta1_t = Math.pow(beta1, t);
  let beta2_t = Math.pow(beta2, t);
  let lr_t = lr * Math.sqrt(1 - beta2_t) / (1 - beta1_t);
  let wgsl = storageBufferRW(0, "param") + `
` + storageBufferRO(1, "grad") + `
` + storageBufferRW(2, "m") + `
` + storageBufferRW(3, "v") + `
const LR = ` + lr.toString() + `;
const LR_T = ` + lr_t.toString() + `;
const BETA1 = ` + beta1.toString() + `;
const BETA2 = ` + beta2.toString() + `;
const EPSILON = ` + epsilon.toString() + `;
const WEIGHT_DECAY = ` + weightDecay.toString() + `;
` + mainSignature2 + `
  if (idx >= ` + size3.toString() + `u) { return; }
  
  let g = grad[idx];
  let p = param[idx];
  
  // Update moments
  let m_new = BETA1 * m[idx] + (1.0 - BETA1) * g;
  m[idx] = m_new;
  let v_new = BETA2 * v[idx] + (1.0 - BETA2) * g * g;
  v[idx] = v_new;
  
  // Update with decoupled weight decay
  param[idx] = p - LR_T * m_new / (sqrt(v_new) + EPSILON) - LR * WEIGHT_DECAY * p;
` + mainEnd2;
  return {
    name: "adamw_" + size3.toString() + "_t" + t.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "param"
      },
      {
        binding: 1,
        size: size3 << 2,
        usage: "ReadOnly",
        name: "grad"
      },
      {
        binding: 2,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "m"
      },
      {
        binding: 3,
        size: size3 << 2,
        usage: "ReadWrite",
        name: "v"
      }
    ]
  };
}

// node_modules/@rescript/runtime/lib/es6/Belt_internalAVLtree.js
function treeHeight(n) {
  if (n !== void 0) {
    return n.h;
  } else {
    return 0;
  }
}
function create2(l, x, d, r) {
  let hl = treeHeight(l);
  let hr = treeHeight(r);
  return {
    k: x,
    v: d,
    h: hl >= hr ? hl + 1 | 0 : hr + 1 | 0,
    l,
    r
  };
}
function singleton2(x, d) {
  return {
    k: x,
    v: d,
    h: 1,
    l: void 0,
    r: void 0
  };
}
function updateValue(n, newValue) {
  if (n.v === newValue) {
    return n;
  } else {
    return {
      k: n.k,
      v: newValue,
      h: n.h,
      l: n.l,
      r: n.r
    };
  }
}
function bal2(l, x, d, r) {
  let hl = l !== void 0 ? l.h : 0;
  let hr = r !== void 0 ? r.h : 0;
  if (hl > (hr + 2 | 0)) {
    let ll = l.l;
    let lr = l.r;
    if (treeHeight(ll) >= treeHeight(lr)) {
      return create2(ll, l.k, l.v, create2(lr, x, d, r));
    } else {
      return create2(create2(ll, l.k, l.v, lr.l), lr.k, lr.v, create2(lr.r, x, d, r));
    }
  }
  if (hr <= (hl + 2 | 0)) {
    return {
      k: x,
      v: d,
      h: hl >= hr ? hl + 1 | 0 : hr + 1 | 0,
      l,
      r
    };
  }
  let rl = r.l;
  let rr = r.r;
  if (treeHeight(rr) >= treeHeight(rl)) {
    return create2(create2(l, x, d, rl), r.k, r.v, rr);
  } else {
    return create2(create2(l, x, d, rl.l), rl.k, rl.v, create2(rl.r, r.k, r.v, rr));
  }
}

// node_modules/@rescript/runtime/lib/es6/Belt_internalMapInt.js
function get2(_n, x) {
  while (true) {
    let n = _n;
    if (n === void 0) {
      return;
    }
    let v = n.k;
    if (x === v) {
      return some(n.v);
    }
    _n = x < v ? n.l : n.r;
    continue;
  }
  ;
}
function has3(_n, x) {
  while (true) {
    let n = _n;
    if (n === void 0) {
      return false;
    }
    let v = n.k;
    if (x === v) {
      return true;
    }
    _n = x < v ? n.l : n.r;
    continue;
  }
  ;
}

// node_modules/@rescript/runtime/lib/es6/Belt_MapInt.js
function set(t, newK, newD) {
  if (t === void 0) {
    return singleton2(newK, newD);
  }
  let k = t.k;
  if (newK === k) {
    return updateValue(t, newD);
  }
  let v = t.v;
  if (newK < k) {
    return bal2(set(t.l, newK, newD), k, v, t.r);
  } else {
    return bal2(t.l, k, v, set(t.r, newK, newD));
  }
}
var has4 = has3;
var get3 = get2;

// node_modules/@rescript/runtime/lib/es6/Belt_SetInt.js
function add3(t, x) {
  if (t === void 0) {
    return singleton(x);
  }
  let v = t.v;
  if (x === v) {
    return t;
  }
  let l = t.l;
  let r = t.r;
  if (x < v) {
    let ll = add3(l, x);
    if (ll === l) {
      return t;
    } else {
      return bal(ll, v, r);
    }
  }
  let rr = add3(r, x);
  if (rr === r) {
    return t;
  } else {
    return bal(l, v, rr);
  }
}
var has5 = has;

// src/GradTape.res.mjs
function create3() {
  return {
    ops: [],
    parameterIds: void 0,
    gradients: void 0
  };
}
function markParameter(tape, nodeId) {
  tape.parameterIds = add3(tape.parameterIds, nodeId);
}
function isParameter(tape, nodeId) {
  return has5(tape.parameterIds, nodeId);
}
function recordOp(tape, nodeId, op, inputIds, inputShapes, outputShape) {
  let record2 = {
    nodeId,
    op,
    inputIds,
    inputShapes,
    outputShape,
    outputId: nodeId
  };
  tape.ops = tape.ops.concat([record2]);
}
function reset(tape) {
  tape.ops = [];
  tape.gradients = void 0;
}
function getBackwardKernels(record2, _gradOutputShape) {
  let outputSize = numElements(record2.outputShape);
  let match = record2.op;
  if (typeof match !== "object") {
    switch (match) {
      case "Identity":
        return [[
          genCopyBackwardKernel(outputSize),
          [
            "grad_out",
            "grad_x"
          ]
        ]];
      case "Neg":
        return [[
          genNegBackwardKernel(outputSize),
          [
            "grad_out",
            "grad_x"
          ]
        ]];
      case "Abs":
        return [[
          genAbsBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Sqrt":
        return [[
          genSqrtBackwardKernel(outputSize),
          [
            "grad_out",
            "out",
            "grad_x"
          ]
        ]];
      case "Exp":
        return [[
          genExpBackwardKernel(outputSize),
          [
            "grad_out",
            "out",
            "grad_x"
          ]
        ]];
      case "Log":
        return [[
          genLogBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Sin":
        return [[
          genSinBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Cos":
        return [[
          genCosBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Tanh":
        return [[
          genTanhBackwardKernel(outputSize),
          [
            "grad_out",
            "out",
            "grad_x"
          ]
        ]];
      case "ReLU":
        return [[
          genReLUBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Sigmoid":
        return [[
          genSigmoidBackwardKernel(outputSize),
          [
            "grad_out",
            "out",
            "grad_x"
          ]
        ]];
      case "GeLU":
        return [[
          genGeLUBackwardKernel(outputSize),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Add":
        let inputShape0 = getOr(record2.inputShapes[0], []);
        let inputShape1 = getOr(record2.inputShapes[1], []);
        let kernels = [[
          genAddBackwardKernel(outputSize),
          [
            "grad_out",
            "grad_a",
            "grad_b"
          ]
        ]];
        let input0Size = numElements(inputShape0);
        let input1Size = numElements(inputShape1);
        if (input0Size < outputSize) {
          let reduceKernel = genGradReduceKernel(record2.outputShape, inputShape0);
          return kernels.concat([[
            reduceKernel,
            [
              "grad_a_full",
              "grad_a_reduced"
            ]
          ]]);
        }
        if (input1Size >= outputSize) {
          return kernels;
        }
        let reduceKernel$1 = genGradReduceKernel(record2.outputShape, inputShape1);
        return kernels.concat([[
          reduceKernel$1,
          [
            "grad_b_full",
            "grad_b_reduced"
          ]
        ]]);
      case "Sub":
        return [[
          genSubBackwardKernel(outputSize),
          [
            "grad_out",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "Mul":
        return [[
          genMulBackwardKernel(outputSize),
          [
            "grad_out",
            "a",
            "b",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "Div":
        return [[
          genDivBackwardKernel(outputSize),
          [
            "grad_out",
            "a",
            "b",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "Pow":
        return [[
          genPowBackwardKernel(outputSize),
          [
            "grad_out",
            "a",
            "b",
            "out",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "Maximum":
        return [[
          genMaximumBackwardKernel(outputSize),
          [
            "grad_out",
            "a",
            "b",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "Minimum":
        return [[
          genMinimumBackwardKernel(outputSize),
          [
            "grad_out",
            "a",
            "b",
            "grad_a",
            "grad_b"
          ]
        ]];
      case "MatMul":
        let shape0 = getOr(record2.inputShapes[0], []);
        let shape1 = getOr(record2.inputShapes[1], []);
        let r0 = shape0.length;
        let r1 = shape1.length;
        if (!(r0 >= 2 && r1 >= 2)) {
          return [];
        }
        let m = getOr(shape0[r0 - 2 | 0], 1);
        let k = getOr(shape0[r0 - 1 | 0], 1);
        let n = getOr(shape1[r1 - 1 | 0], 1);
        let batchDims = record2.outputShape.slice(0, record2.outputShape.length - 2 | 0);
        let batchSize = reduce(batchDims, 1, (a, b) => a * b | 0);
        if (batchSize > 1) {
          return [
            [
              genBatchedMatMulBackwardAKernel(batchSize, m, k, n),
              [
                "grad_out",
                "b",
                "grad_a"
              ]
            ],
            [
              genBatchedMatMulBackwardBKernel(batchSize, m, k, n),
              [
                "grad_out",
                "a",
                "grad_b"
              ]
            ]
          ];
        } else {
          return [
            [
              genMatMulBackwardAKernel(m, k, n),
              [
                "grad_out",
                "b",
                "grad_a"
              ]
            ],
            [
              genMatMulBackwardBKernel(m, k, n),
              [
                "grad_out",
                "a",
                "grad_b"
              ]
            ]
          ];
        }
      default:
        return [];
    }
  } else {
    switch (match.TAG) {
      case "LeakyReLU":
        return [[
          genLeakyReLUBackwardKernel(outputSize, match.alpha),
          [
            "grad_out",
            "x",
            "grad_x"
          ]
        ]];
      case "Reduce":
        switch (match.op) {
          case "Sum":
            let inputShape = getOr(record2.inputShapes[0], []);
            return [[
              genSumBackwardKernel(inputShape, record2.outputShape, match.axes),
              [
                "grad_out",
                "grad_x"
              ]
            ]];
          case "Mean":
            let inputShape$1 = getOr(record2.inputShapes[0], []);
            return [[
              genMeanBackwardKernel(inputShape$1, record2.outputShape, match.axes),
              [
                "grad_out",
                "grad_x"
              ]
            ]];
          default:
            return [];
        }
      case "Transpose":
        return [[
          genCopyBackwardKernel(outputSize),
          [
            "grad_out",
            "grad_x"
          ]
        ]];
      case "Reshape":
      case "Squeeze":
      case "Unsqueeze":
      case "Flatten":
      case "ExpandDims":
        break;
      case "LayerNorm":
        let inputShape$2 = getOr(record2.inputShapes[0], []);
        let rank = inputShape$2.length;
        let normAxes = match.axes.map((a) => {
          if (a < 0) {
            return rank + a | 0;
          } else {
            return a;
          }
        });
        let normSize = reduce(normAxes, 1, (acc, axis2) => acc * getOr(inputShape$2[axis2], 1) | 0);
        let outerSize = div(numElements(inputShape$2), normSize);
        return [[
          genLayerNormBackwardKernel(outerSize, normSize, match.epsilon),
          [
            "grad_out",
            "x",
            "gamma",
            "grad_x",
            "grad_gamma",
            "grad_beta"
          ]
        ]];
      case "Softmax":
        let axis = match.axis;
        let inputShape$3 = getOr(record2.inputShapes[0], []);
        let rank$1 = inputShape$3.length;
        let normAxis = axis < 0 ? rank$1 + axis | 0 : axis;
        let axisSize = getOr(inputShape$3[normAxis], 1);
        let outerSize$1 = div(numElements(inputShape$3), axisSize);
        return [[
          genSoftmaxBackwardKernel(outerSize$1, axisSize),
          [
            "grad_out",
            "softmax_out",
            "grad_x"
          ]
        ]];
      default:
        return [];
    }
  }
  return [[
    genCopyBackwardKernel(outputSize),
    [
      "grad_out",
      "grad_x"
    ]
  ]];
}
function supportsGradient(op) {
  if (typeof op !== "object") {
    switch (op) {
      case "Identity":
      case "Neg":
      case "Abs":
      case "Sqrt":
      case "Exp":
      case "Log":
      case "Sin":
      case "Cos":
      case "Tanh":
      case "ReLU":
      case "Sigmoid":
      case "GeLU":
      case "Add":
      case "Sub":
      case "Mul":
      case "Div":
      case "Pow":
      case "Maximum":
      case "Minimum":
      case "MatMul":
        return true;
      default:
        return false;
    }
  } else {
    switch (op.TAG) {
      case "Reduce":
        switch (op.op) {
          case "Sum":
          case "Mean":
            return true;
          default:
            return false;
        }
      case "LeakyReLU":
      case "Reshape":
      case "Squeeze":
      case "Unsqueeze":
      case "Flatten":
      case "Transpose":
      case "ExpandDims":
      case "LayerNorm":
      case "Softmax":
        return true;
      default:
        return false;
    }
  }
}
function compileBackward(tape, lossNodeId, nodeShapes) {
  let ops = tape.ops.toReversed();
  let kernels = {
    contents: []
  };
  let dispatches = {
    contents: []
  };
  let buffers = {
    contents: []
  };
  let nextBufferId = {
    contents: 0
  };
  let gradientBufferIds = {
    contents: void 0
  };
  let paramGradIds = {
    contents: []
  };
  let lossShape = getOr(get3(nodeShapes, lossNodeId), [1]);
  let lossGradSize = numElements(lossShape);
  let lossGradBufferId = nextBufferId.contents;
  nextBufferId.contents = nextBufferId.contents + 1 | 0;
  buffers.contents = buffers.contents.concat([{
    id: lossGradBufferId,
    size: lossGradSize << 2,
    name: "grad_loss"
  }]);
  gradientBufferIds.contents = set(gradientBufferIds.contents, lossNodeId, lossGradBufferId);
  ops.forEach((record2) => {
    if (!supportsGradient(record2.op)) {
      return;
    }
    let gradOutShape = getOr(get3(nodeShapes, record2.outputId), []);
    let backwardKernels = getBackwardKernels(record2, gradOutShape);
    backwardKernels.forEach((param) => {
      let kernel = param[0];
      record2.inputIds.forEach((inputId, _idx) => {
        if (has4(gradientBufferIds.contents, inputId)) {
          return;
        }
        let inputShape = getOr(record2.inputShapes[_idx], []);
        let gradSize = numElements(inputShape);
        let gradBufferId = nextBufferId.contents;
        nextBufferId.contents = nextBufferId.contents + 1 | 0;
        buffers.contents = buffers.contents.concat([{
          id: gradBufferId,
          size: gradSize << 2,
          name: "grad_" + inputId.toString()
        }]);
        gradientBufferIds.contents = set(gradientBufferIds.contents, inputId, gradBufferId);
        if (has5(tape.parameterIds, inputId)) {
          paramGradIds.contents = paramGradIds.contents.concat([[
            inputId,
            gradBufferId
          ]]);
          return;
        }
      });
      kernels.contents = kernels.contents.concat([kernel]);
      let totalElements = numElements(gradOutShape);
      let workgroupCount = (totalElements + 255 | 0) / 256 | 0;
      dispatches.contents = dispatches.contents.concat([{
        workgroupSize: [
          256,
          1,
          1
        ],
        workgroupCount: [
          workgroupCount,
          1,
          1
        ],
        kernelName: kernel.name,
        pipelineIndex: kernels.contents.length - 1 | 0
      }]);
    });
  });
  return {
    kernels: kernels.contents,
    dispatches: dispatches.contents,
    buffers: buffers.contents,
    gradientBufferIds: gradientBufferIds.contents,
    parameterGradientIds: paramGradIds.contents
  };
}

// src/AutogradEngine.res.mjs
function create4() {
  return {
    tape: [],
    parameters: [],
    requiresGrad: void 0,
    gradients: void 0,
    adamStates: void 0,
    timestep: 0,
    bufferCache: void 0,
    shapeCache: void 0
  };
}
function markRequiresGrad(engine, nodeId, shape, name) {
  engine.requiresGrad = add3(engine.requiresGrad, nodeId);
  engine.parameters = engine.parameters.concat([{
    nodeId,
    bufferId: -1,
    shape,
    name
  }]);
  engine.shapeCache = set(engine.shapeCache, nodeId, shape);
}
function needsGrad(engine, nodeId) {
  return has5(engine.requiresGrad, nodeId);
}
function record(engine, nodeId, op, inputNodeIds, inputBufferIds, inputShapes, outputBufferId, outputShape) {
  let anyRequiresGrad = inputNodeIds.some((id) => has5(engine.requiresGrad, id));
  if (!anyRequiresGrad) {
    return;
  }
  let inputs = inputNodeIds.map((id, idx) => ({
    nodeId: id,
    bufferId: getOr(inputBufferIds[idx], -1),
    shape: getOr(inputShapes[idx], []),
    data: void 0
  }));
  let output = {
    nodeId,
    bufferId: outputBufferId,
    shape: outputShape,
    data: void 0
  };
  engine.tape = engine.tape.concat([{
    nodeId,
    op,
    inputs,
    output
  }]);
  engine.bufferCache = set(engine.bufferCache, nodeId, outputBufferId);
  engine.shapeCache = set(engine.shapeCache, nodeId, outputShape);
  engine.requiresGrad = add3(engine.requiresGrad, nodeId);
}
function clearTape(engine) {
  engine.tape = [];
}
function getOrAllocateGradient(engine, nodeId, _size) {
  let bufferId = get3(engine.gradients, nodeId);
  if (bufferId !== void 0) {
    return bufferId;
  }
  let bufferId$1 = (nodeId * 1e3 | 0) + 500 | 0;
  engine.gradients = set(engine.gradients, nodeId, bufferId$1);
  return bufferId$1;
}
function generateBackwardOps(engine, lossNodeId) {
  let ops = {
    contents: []
  };
  let reversedTape = engine.tape.toReversed();
  let lossShape = getOr(get3(engine.shapeCache, lossNodeId), [1]);
  let lossSize = numElements(lossShape);
  getOrAllocateGradient(engine, lossNodeId, lossSize);
  reversedTape.forEach((record2) => {
    let outputSize = numElements(record2.output.shape);
    let gradOutBufferId = getOrAllocateGradient(engine, record2.nodeId, outputSize);
    let match = record2.op;
    if (typeof match !== "object") {
      switch (match) {
        case "Identity":
          break;
        case "Neg":
          let inputId = getOr(map(record2.inputs[0], (i) => i.nodeId), -1);
          let inputSize = getOr(map(record2.inputs[0], (i) => numElements(i.shape)), 0);
          let gradInBufferId = getOrAllocateGradient(engine, inputId, inputSize);
          ops.contents = ops.contents.concat([{
            kernel: genNegBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "grad_x",
                gradInBufferId
              ]
            ],
            outputSize
          }]);
          return;
        case "Abs":
          let input2 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$1 = numElements(input2.shape);
          let gradInBufferId$1 = getOrAllocateGradient(engine, input2.nodeId, inputSize$1);
          ops.contents = ops.contents.concat([{
            kernel: genAbsBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input2.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$1
              ]
            ],
            outputSize
          }]);
          return;
        case "Sqrt":
          let input$1 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$2 = numElements(input$1.shape);
          let gradInBufferId$2 = getOrAllocateGradient(engine, input$1.nodeId, inputSize$2);
          ops.contents = ops.contents.concat([{
            kernel: genSqrtBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "out",
                record2.output.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$2
              ]
            ],
            outputSize
          }]);
          return;
        case "Exp":
          let input$2 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$3 = numElements(input$2.shape);
          let gradInBufferId$3 = getOrAllocateGradient(engine, input$2.nodeId, inputSize$3);
          ops.contents = ops.contents.concat([{
            kernel: genExpBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "out",
                record2.output.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$3
              ]
            ],
            outputSize
          }]);
          return;
        case "Log":
          let input$3 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$4 = numElements(input$3.shape);
          let gradInBufferId$4 = getOrAllocateGradient(engine, input$3.nodeId, inputSize$4);
          ops.contents = ops.contents.concat([{
            kernel: genLogBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$3.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$4
              ]
            ],
            outputSize
          }]);
          return;
        case "Sin":
          let input$4 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$5 = numElements(input$4.shape);
          let gradInBufferId$5 = getOrAllocateGradient(engine, input$4.nodeId, inputSize$5);
          ops.contents = ops.contents.concat([{
            kernel: genSinBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$4.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$5
              ]
            ],
            outputSize
          }]);
          return;
        case "Cos":
          let input$5 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$6 = numElements(input$5.shape);
          let gradInBufferId$6 = getOrAllocateGradient(engine, input$5.nodeId, inputSize$6);
          ops.contents = ops.contents.concat([{
            kernel: genCosBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$5.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$6
              ]
            ],
            outputSize
          }]);
          return;
        case "Tanh":
          let input$6 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$7 = numElements(input$6.shape);
          let gradInBufferId$7 = getOrAllocateGradient(engine, input$6.nodeId, inputSize$7);
          ops.contents = ops.contents.concat([{
            kernel: genTanhBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "out",
                record2.output.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$7
              ]
            ],
            outputSize
          }]);
          return;
        case "ReLU":
          let input$7 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$8 = numElements(input$7.shape);
          let gradInBufferId$8 = getOrAllocateGradient(engine, input$7.nodeId, inputSize$8);
          ops.contents = ops.contents.concat([{
            kernel: genReLUBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$7.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$8
              ]
            ],
            outputSize
          }]);
          return;
        case "Sigmoid":
          let input$8 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$9 = numElements(input$8.shape);
          let gradInBufferId$9 = getOrAllocateGradient(engine, input$8.nodeId, inputSize$9);
          ops.contents = ops.contents.concat([{
            kernel: genSigmoidBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "out",
                record2.output.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$9
              ]
            ],
            outputSize
          }]);
          return;
        case "GeLU":
          let input$9 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$10 = numElements(input$9.shape);
          let gradInBufferId$10 = getOrAllocateGradient(engine, input$9.nodeId, inputSize$10);
          ops.contents = ops.contents.concat([{
            kernel: genGeLUBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$9.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$10
              ]
            ],
            outputSize
          }]);
          return;
        case "Add":
          let input0 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let input1 = getOr(record2.inputs[1], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let size0 = numElements(input0.shape);
          let size1 = numElements(input1.shape);
          let gradA = getOrAllocateGradient(engine, input0.nodeId, size0);
          let gradB = getOrAllocateGradient(engine, input1.nodeId, size1);
          if (size0 === outputSize && size1 === outputSize) {
            ops.contents = ops.contents.concat([{
              kernel: genAddBackwardKernel(outputSize),
              bindings: [
                [
                  "grad_out",
                  gradOutBufferId
                ],
                [
                  "grad_a",
                  gradA
                ],
                [
                  "grad_b",
                  gradB
                ]
              ],
              outputSize
            }]);
          } else {
            ops.contents = ops.contents.concat([{
              kernel: genAddBackwardKernel(outputSize),
              bindings: [
                [
                  "grad_out",
                  gradOutBufferId
                ],
                [
                  "grad_a",
                  gradA
                ],
                [
                  "grad_b",
                  gradB
                ]
              ],
              outputSize
            }]);
          }
          return;
        case "Sub":
          let input0$1 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let input1$1 = getOr(record2.inputs[1], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let size0$1 = numElements(input0$1.shape);
          let size1$1 = numElements(input1$1.shape);
          let gradA$1 = getOrAllocateGradient(engine, input0$1.nodeId, size0$1);
          let gradB$1 = getOrAllocateGradient(engine, input1$1.nodeId, size1$1);
          ops.contents = ops.contents.concat([{
            kernel: genSubBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "grad_a",
                gradA$1
              ],
              [
                "grad_b",
                gradB$1
              ]
            ],
            outputSize
          }]);
          return;
        case "Mul":
          let input0$2 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let input1$2 = getOr(record2.inputs[1], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let size0$2 = numElements(input0$2.shape);
          let size1$2 = numElements(input1$2.shape);
          let gradA$2 = getOrAllocateGradient(engine, input0$2.nodeId, size0$2);
          let gradB$2 = getOrAllocateGradient(engine, input1$2.nodeId, size1$2);
          ops.contents = ops.contents.concat([{
            kernel: genMulBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "a",
                input0$2.bufferId
              ],
              [
                "b",
                input1$2.bufferId
              ],
              [
                "grad_a",
                gradA$2
              ],
              [
                "grad_b",
                gradB$2
              ]
            ],
            outputSize
          }]);
          return;
        case "Div":
          let input0$3 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let input1$3 = getOr(record2.inputs[1], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let size0$3 = numElements(input0$3.shape);
          let size1$3 = numElements(input1$3.shape);
          let gradA$3 = getOrAllocateGradient(engine, input0$3.nodeId, size0$3);
          let gradB$3 = getOrAllocateGradient(engine, input1$3.nodeId, size1$3);
          ops.contents = ops.contents.concat([{
            kernel: genDivBackwardKernel(outputSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "a",
                input0$3.bufferId
              ],
              [
                "b",
                input1$3.bufferId
              ],
              [
                "grad_a",
                gradA$3
              ],
              [
                "grad_b",
                gradB$3
              ]
            ],
            outputSize
          }]);
          return;
        case "MatMul":
          let input0$4 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let input1$4 = getOr(record2.inputs[1], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let shape0 = input0$4.shape;
          let shape1 = input1$4.shape;
          let r0 = shape0.length;
          let r1 = shape1.length;
          if (!(r0 >= 2 && r1 >= 2)) {
            return;
          }
          let m = getOr(shape0[r0 - 2 | 0], 1);
          let k = getOr(shape0[r0 - 1 | 0], 1);
          let n = getOr(shape1[r1 - 1 | 0], 1);
          let size0$4 = numElements(shape0);
          let size1$4 = numElements(shape1);
          let gradA$4 = getOrAllocateGradient(engine, input0$4.nodeId, size0$4);
          let gradB$4 = getOrAllocateGradient(engine, input1$4.nodeId, size1$4);
          let batchDims = record2.output.shape.slice(0, record2.output.shape.length - 2 | 0);
          let batchSize = reduce(batchDims, 1, (a, b) => a * b | 0);
          if (batchSize > 1) {
            ops.contents = ops.contents.concat([
              {
                kernel: genBatchedMatMulBackwardAKernel(batchSize, m, k, n),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "b",
                    input1$4.bufferId
                  ],
                  [
                    "grad_a",
                    gradA$4
                  ]
                ],
                outputSize: size0$4
              },
              {
                kernel: genBatchedMatMulBackwardBKernel(batchSize, m, k, n),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "a",
                    input0$4.bufferId
                  ],
                  [
                    "grad_b",
                    gradB$4
                  ]
                ],
                outputSize: size1$4
              }
            ]);
          } else {
            ops.contents = ops.contents.concat([
              {
                kernel: genMatMulBackwardAKernel(m, k, n),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "b",
                    input1$4.bufferId
                  ],
                  [
                    "grad_a",
                    gradA$4
                  ]
                ],
                outputSize: size0$4
              },
              {
                kernel: genMatMulBackwardBKernel(m, k, n),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "a",
                    input0$4.bufferId
                  ],
                  [
                    "grad_b",
                    gradB$4
                  ]
                ],
                outputSize: size1$4
              }
            ]);
          }
          return;
        default:
          return;
      }
    } else {
      switch (match.TAG) {
        case "LeakyReLU":
          let input$10 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputSize$11 = numElements(input$10.shape);
          let gradInBufferId$11 = getOrAllocateGradient(engine, input$10.nodeId, inputSize$11);
          ops.contents = ops.contents.concat([{
            kernel: genLeakyReLUBackwardKernel(outputSize, match.alpha),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "x",
                input$10.bufferId
              ],
              [
                "grad_x",
                gradInBufferId$11
              ]
            ],
            outputSize
          }]);
          return;
        case "Reduce":
          switch (match.op) {
            case "Sum":
              let input$11 = getOr(record2.inputs[0], {
                nodeId: -1,
                bufferId: -1,
                shape: [],
                data: void 0
              });
              let inputSize$12 = numElements(input$11.shape);
              let gradIn = getOrAllocateGradient(engine, input$11.nodeId, inputSize$12);
              ops.contents = ops.contents.concat([{
                kernel: genSumBackwardKernel(input$11.shape, record2.output.shape, match.axes),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "grad_x",
                    gradIn
                  ]
                ],
                outputSize: inputSize$12
              }]);
              return;
            case "Mean":
              let input$12 = getOr(record2.inputs[0], {
                nodeId: -1,
                bufferId: -1,
                shape: [],
                data: void 0
              });
              let inputSize$13 = numElements(input$12.shape);
              let gradIn$1 = getOrAllocateGradient(engine, input$12.nodeId, inputSize$13);
              ops.contents = ops.contents.concat([{
                kernel: genMeanBackwardKernel(input$12.shape, record2.output.shape, match.axes),
                bindings: [
                  [
                    "grad_out",
                    gradOutBufferId
                  ],
                  [
                    "grad_x",
                    gradIn$1
                  ]
                ],
                outputSize: inputSize$13
              }]);
              return;
            default:
              return;
          }
        case "Reshape":
        case "Squeeze":
        case "Unsqueeze":
        case "Flatten":
        case "ExpandDims":
          break;
        case "Softmax":
          let axis = match.axis;
          let input$13 = getOr(record2.inputs[0], {
            nodeId: -1,
            bufferId: -1,
            shape: [],
            data: void 0
          });
          let inputShape = input$13.shape;
          let rank = inputShape.length;
          let normAxis = axis < 0 ? rank + axis | 0 : axis;
          let axisSize = getOr(inputShape[normAxis], 1);
          let outerSize = div(outputSize, axisSize);
          let inputSize$14 = numElements(input$13.shape);
          let gradIn$2 = getOrAllocateGradient(engine, input$13.nodeId, inputSize$14);
          ops.contents = ops.contents.concat([{
            kernel: genSoftmaxBackwardKernel(outerSize, axisSize),
            bindings: [
              [
                "grad_out",
                gradOutBufferId
              ],
              [
                "softmax_out",
                record2.output.bufferId
              ],
              [
                "grad_x",
                gradIn$2
              ]
            ],
            outputSize: inputSize$14
          }]);
          return;
        default:
          return;
      }
    }
    let input$14 = getOr(record2.inputs[0], {
      nodeId: -1,
      bufferId: -1,
      shape: [],
      data: void 0
    });
    let inputSize$15 = numElements(input$14.shape);
    let gradIn$3 = getOrAllocateGradient(engine, input$14.nodeId, inputSize$15);
    ops.contents = ops.contents.concat([{
      kernel: genCopyBackwardKernel(outputSize),
      bindings: [
        [
          "grad_out",
          gradOutBufferId
        ],
        [
          "grad_x",
          gradIn$3
        ]
      ],
      outputSize: inputSize$15
    }]);
  });
  return ops.contents;
}
function getParameterGradients(engine) {
  return filterMap(engine.parameters, (param) => {
    let gradBufferId = get3(engine.gradients, param.nodeId);
    if (gradBufferId !== void 0) {
      return [
        param.nodeId,
        gradBufferId,
        param.shape
      ];
    }
  });
}
function generateOptimizerOps(engine, optimizer, lr, beta1, beta2, epsilon, weightDecay) {
  engine.timestep = engine.timestep + 1 | 0;
  let t = engine.timestep;
  return filterMap(engine.parameters, (param) => {
    let gradBufferId = get3(engine.gradients, param.nodeId);
    if (gradBufferId === void 0) {
      return;
    }
    let size3 = numElements(param.shape);
    switch (optimizer) {
      case "adam":
        let state = get3(engine.adamStates, param.nodeId);
        let adamState;
        if (state !== void 0) {
          adamState = state;
        } else {
          let mId = (param.nodeId * 1e3 | 0) + 600 | 0;
          let vId = (param.nodeId * 1e3 | 0) + 700 | 0;
          let state$1 = {
            m: mId,
            v: vId
          };
          engine.adamStates = set(engine.adamStates, param.nodeId, state$1);
          adamState = state$1;
        }
        return {
          kernel: genAdamKernel(size3, lr, beta1, beta2, epsilon, t),
          bindings: [
            [
              "param",
              param.bufferId
            ],
            [
              "grad",
              gradBufferId
            ],
            [
              "m",
              adamState.m
            ],
            [
              "v",
              adamState.v
            ]
          ],
          outputSize: size3
        };
      case "adamw":
        let state$2 = get3(engine.adamStates, param.nodeId);
        let adamState$1;
        if (state$2 !== void 0) {
          adamState$1 = state$2;
        } else {
          let mId$1 = (param.nodeId * 1e3 | 0) + 600 | 0;
          let vId$1 = (param.nodeId * 1e3 | 0) + 700 | 0;
          let state$3 = {
            m: mId$1,
            v: vId$1
          };
          engine.adamStates = set(engine.adamStates, param.nodeId, state$3);
          adamState$1 = state$3;
        }
        return {
          kernel: genAdamWKernel(size3, lr, beta1, beta2, epsilon, weightDecay, t),
          bindings: [
            [
              "param",
              param.bufferId
            ],
            [
              "grad",
              gradBufferId
            ],
            [
              "m",
              adamState$1.m
            ],
            [
              "v",
              adamState$1.v
            ]
          ],
          outputSize: size3
        };
      case "sgd":
        return {
          kernel: genSGDKernel(size3, lr),
          bindings: [
            [
              "param",
              param.bufferId
            ],
            [
              "grad",
              gradBufferId
            ]
          ],
          outputSize: size3
        };
      default:
        return;
    }
  });
}
function generateZeroGradOps(engine) {
  return filterMap(engine.parameters, (param) => {
    let gradBufferId = get3(engine.gradients, param.nodeId);
    if (gradBufferId === void 0) {
      return;
    }
    let size3 = numElements(param.shape);
    return {
      kernel: genGradZeroKernel(size3),
      bindings: [[
        "grad",
        gradBufferId
      ]],
      outputSize: size3
    };
  });
}

// src/bridge.js
var Autograd = {
  // Gradient reduction for broadcasting
  genGradReduceKernel: (gradShape, targetShape) => genGradReduceKernel(gradShape, targetShape),
  // Unary backward kernels
  genNegBackwardKernel,
  genAbsBackwardKernel,
  genSqrtBackwardKernel,
  genExpBackwardKernel,
  genLogBackwardKernel,
  genSinBackwardKernel,
  genCosBackwardKernel,
  genTanhBackwardKernel,
  genSigmoidBackwardKernel,
  genReLUBackwardKernel,
  genLeakyReLUBackwardKernel,
  genGeLUBackwardKernel,
  // Binary backward kernels
  genAddBackwardKernel,
  genSubBackwardKernel,
  genMulBackwardKernel,
  genDivBackwardKernel,
  genPowBackwardKernel,
  genMaximumBackwardKernel,
  genMinimumBackwardKernel,
  // MatMul backward
  genMatMulBackwardAKernel,
  genMatMulBackwardBKernel,
  genBatchedMatMulBackwardAKernel,
  genBatchedMatMulBackwardBKernel,
  // Reduction backward
  genSumBackwardKernel,
  genMeanBackwardKernel,
  // Softmax backward
  genSoftmaxBackwardKernel,
  // LayerNorm backward
  genLayerNormBackwardKernel,
  // Utility
  genCopyBackwardKernel,
  genGradAccumulateKernel,
  genGradZeroKernel,
  // Optimizers
  genSGDKernel,
  genSGDMomentumKernel,
  genAdamKernel,
  genAdamWKernel
};
var GradTape = {
  create: create3,
  markParameter,
  isParameter,
  recordOp,
  reset,
  getBackwardKernels,
  supportsGradient,
  compileBackward
};
var AutogradEngine = {
  create: create4,
  markRequiresGrad,
  needsGrad,
  record,
  clearTape,
  generateBackwardOps,
  getParameterGradients,
  generateOptimizerOps,
  generateZeroGradOps
};
export {
  Autograd,
  AutogradEngine,
  Codegen_res_exports as Codegen,
  Compiler_res_exports as Compiler,
  GradTape,
  Shape_res_exports as Shape,
  Types_res_exports as Types
};
