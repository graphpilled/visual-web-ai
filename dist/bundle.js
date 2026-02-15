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

// src/Shape.res.mjs
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
function inferReduce(input, axes, keepDims) {
  let rank = input.length;
  let normAxes = axes.map((a) => normalizeAxis(a, rank));
  if (keepDims) {
    return input.map((d, i) => {
      if (normAxes.includes(i)) {
        return 1;
      } else {
        return d;
      }
    });
  } else {
    return input.filter((param, i) => !normAxes.includes(i));
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
function inferConv2D(input, filters, param, param$1, padding, param$2) {
  let sW = param$1[1];
  let sH = param$1[0];
  if (input.length !== 4) {
    return;
  }
  let batch = at(input, 0);
  let inH = at(input, 1);
  let inW = at(input, 2);
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
function inferPool2D(input, param, param$1, padding) {
  let sW = param$1[1];
  let sH = param$1[0];
  let kW = param[1];
  let kH = param[0];
  if (input.length !== 4) {
    return;
  }
  let batch = at(input, 0);
  let inH = at(input, 1);
  let inW = at(input, 2);
  let channels = at(input, 3);
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
function inferReshape(input, newShape) {
  let total = numElements(input);
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
  let get = (i) => getOr(inputs[i], []);
  let input = get(0);
  let r = input.length;
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
        return broadcast(input, get(1));
      case "Where":
        return flatMap(broadcast(input, get(1)), (s) => broadcast(s, get(2)));
      case "MatMul":
      case "BatchedMatMul":
        return inferMatMul(input, get(1));
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
        return input;
    }
  } else {
    switch (op.TAG) {
      case "Reduce":
        return inferReduce(input, op.axes, op.keepDims);
      case "ArgMax":
      case "ArgMin":
        exit = 1;
        break;
      case "MatMulInt4":
        let s2 = get(1);
        let m = at(input, r - 2 | 0);
        let n = at(s2, 0);
        return [
          m,
          n
        ];
      case "Gemm":
        return inferMatMul(input, get(1));
      case "Reshape":
        return inferReshape(input, op.newShape);
      case "Squeeze":
        let axes = op.axes;
        return input.filter((d, i) => !(axes.includes(i) && d === 1));
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
            result = result.concat([at(input, inputIdx)]);
            inputIdx = inputIdx + 1 | 0;
          }
        }
        return result;
      case "Flatten":
        let a = normalizeAxis(op.axis, r);
        return [
          numElements(input.slice(0, a)),
          numElements(input.slice(a, r))
        ];
      case "Transpose":
        let perm = op.perm;
        if (perm.length === r) {
          return perm.map((p) => at(input, p));
        } else {
          return;
        }
      case "Broadcast":
        return op.targetShape;
      case "ExpandDims":
        let normAxis = normalizeAxis(op.axis, r + 1 | 0);
        let before = input.slice(0, normAxis);
        let after = input.slice(normAxis, r);
        return before.concat([1]).concat(after);
      case "Slice":
        let steps = op.steps;
        let ends = op.ends;
        let starts = op.starts;
        let result$1 = input.slice();
        op.axes.forEach((ax, i) => {
          let dimSize = at(input, ax);
          let start = getOr(starts[i], 0);
          let end_ = getOr(ends[i], dimSize);
          let step = getOr(steps[i], 1);
          let s = start < 0 ? dimSize + start | 0 : start;
          let e = end_ < 0 ? dimSize + end_ | 0 : end_;
          let size = div(((e - s | 0) + step | 0) - 1 | 0, step);
          result$1[ax] = size;
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
        let firstSize = getOr(op.splitSizes[0], at(input, normAxis$1));
        return input.map((d, i) => {
          if (i === normAxis$1) {
            return firstSize;
          } else {
            return d;
          }
        });
      case "Stack":
        let numInputs = inputs.length;
        let normAxis$2 = normalizeAxis(op.axis, r + 1 | 0);
        let before$1 = input.slice(0, normAxis$2);
        let after$1 = input.slice(normAxis$2, r);
        return before$1.concat([numInputs]).concat(after$1);
      case "Tile":
        let repeats = op.repeats;
        return input.map((d, i) => d * getOr(repeats[i], 1) | 0);
      case "Pad":
        let pads = op.pads;
        return fromInitializer(r, (i) => {
          let before2 = getOr(pads[i], 0);
          let after2 = getOr(pads[r + i | 0], 0);
          return (at(input, i) + before2 | 0) + after2 | 0;
        });
      case "Conv1D":
        if (r !== 3) {
          return;
        }
        let padding$1 = op.padding;
        let stride = op.stride;
        let batch = at(input, 0);
        let inLen = at(input, 1);
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
        return inferConv2D(input, op.filters, op.kernel, op.stride, op.padding, op.dilation);
      case "Conv3D":
        if (r !== 5) {
          return;
        }
        let match = op.dilation;
        let padding$2 = op.padding;
        let match$1 = op.stride;
        let match$2 = op.kernel;
        let batch$1 = at(input, 0);
        let calc = (inSize, k2, s, d) => {
          let effK2 = ((k2 - 1 | 0) * d | 0) + 1 | 0;
          if (typeof padding$2 !== "object" && padding$2 === "Same") {
            return div((inSize + s | 0) - 1 | 0, s);
          }
          return div(inSize - effK2 | 0, s) + 1 | 0;
        };
        return [
          batch$1,
          calc(at(input, 1), match$2[0], match$1[0], match[0]),
          calc(at(input, 2), match$2[1], match$1[1], match[1]),
          calc(at(input, 3), match$2[2], match$1[2], match[2]),
          op.filters
        ];
      case "ConvTranspose1D":
        if (r !== 3) {
          return;
        }
        let stride$1 = op.stride;
        let batch$2 = at(input, 0);
        let inLen$1 = at(input, 1);
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
        let batch$3 = at(input, 0);
        let inH = at(input, 1);
        let inW = at(input, 2);
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
        let batch$4 = at(input, 0);
        let calc$1 = (inSize, k2, s, op2) => {
          if (typeof padding$3 !== "object" && padding$3 === "Same") {
            return (inSize * s | 0) + op2 | 0;
          }
          return ((inSize * s | 0) + max(k2 - s | 0, 0) | 0) + op2 | 0;
        };
        return [
          batch$4,
          calc$1(at(input, 1), match$9[0], match$8[0], match$7[0]),
          calc$1(at(input, 2), match$9[1], match$8[1], match$7[1]),
          calc$1(at(input, 3), match$9[2], match$8[2], match$7[2]),
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
        let batch$5 = at(input, 0);
        let inH$1 = at(input, 1);
        let inW$1 = at(input, 2);
        let channels = at(input, 3);
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
        let batch$6 = at(input, 0);
        let inLen$2 = at(input, 1);
        let channels$1 = at(input, 2);
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
        let batch$7 = at(input, 0);
        let inLen$3 = at(input, 1);
        let channels$2 = at(input, 2);
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
        return inferPool2D(input, [
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
          return input.slice(0, r - 1 | 0).concat([op.units]);
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
              at(input, 0),
              at(sizes, 0),
              at(sizes, 1),
              at(input, 3)
            ];
          } else {
            return;
          }
        } else {
          return input;
        }
      case "SpaceToDepth":
        let blockSize = op.blockSize;
        if (r === 4) {
          return [
            at(input, 0),
            div(at(input, 1), blockSize),
            div(at(input, 2), blockSize),
            (at(input, 3) * blockSize | 0) * blockSize | 0
          ];
        } else {
          return;
        }
      case "DepthToSpace":
        let blockSize$1 = op.blockSize;
        if (r === 4) {
          return [
            at(input, 0),
            at(input, 1) * blockSize$1 | 0,
            at(input, 2) * blockSize$1 | 0,
            div(at(input, 3), blockSize$1 * blockSize$1 | 0)
          ];
        } else {
          return;
        }
      case "GridSample":
        let grid = get(1);
        if (r === 4 && grid.length === 4) {
          return [
            at(input, 0),
            at(grid, 1),
            at(grid, 2),
            at(input, 3)
          ];
        } else {
          return;
        }
      case "RoiAlign":
        let rois = get(1);
        if (r === 4 && rois.length === 2) {
          return [
            at(rois, 0),
            op.outputHeight,
            op.outputWidth,
            at(input, 1)
          ];
        } else {
          return;
        }
      case "OneHot":
        let normAxis$3 = normalizeAxis(op.axis, r + 1 | 0);
        let before$2 = input.slice(0, normAxis$3);
        let after$2 = input.slice(normAxis$3, r);
        return before$2.concat([op.depth]).concat(after$2);
      case "Embedding":
        return input.concat([op.embeddingDim]);
      case "TopK":
        let k = op.k;
        let normAxis$4 = normalizeAxis(op.axis, r);
        return input.map((d, i) => {
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
        return input;
    }
  }
  switch (exit) {
    case 1:
      return inferReduce(input, [op.axis], op.keepDims);
    case 2:
      let indices = get(1);
      let normAxis$5 = normalizeAxis(op.axis, r);
      let before$3 = input.slice(0, normAxis$5);
      let after$3 = input.slice(normAxis$5 + 1 | 0, r);
      return before$3.concat(indices).concat(after$3);
    case 3:
      return inferPool2D(input, op.kernel, op.stride, op.padding);
    case 4:
      if (r === 3) {
        return [
          at(input, 0),
          op.outputSize,
          at(input, 2)
        ];
      } else {
        return;
      }
    case 5:
      let match$20 = op.outputSize;
      if (r === 4) {
        return [
          at(input, 0),
          match$20[0],
          match$20[1],
          at(input, 3)
        ];
      } else {
        return;
      }
    case 6:
      if (r >= 2) {
        return [
          at(input, 0),
          at(input, 1),
          op.dim
        ];
      } else {
        return;
      }
    case 7:
      if (r >= 2) {
        return [
          at(input, 0),
          at(input, 1),
          op.hiddenSize
        ];
      } else {
        return;
      }
    case 8:
      if (r !== 5) {
        return;
      }
      let batch$8 = at(input, 0);
      let channels$3 = at(input, 4);
      let calc$2 = (inSize, k, s) => {
        if (typeof padding !== "object" && padding === "Same") {
          return div((inSize + s | 0) - 1 | 0, s);
        }
        return div(inSize - k | 0, s) + 1 | 0;
      };
      return [
        batch$8,
        calc$2(at(input, 1), kD, sD),
        calc$2(at(input, 2), kH, sH),
        calc$2(at(input, 3), kW, sW),
        channels$3
      ];
    case 9:
      if (r === 4) {
        return [
          at(input, 0),
          1,
          1,
          at(input, 3)
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
var mainSignature = `@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;`;
var mainEnd = "}";
function unaryExpr(op, input) {
  if (typeof op !== "object") {
    switch (op) {
      case "Identity":
        return input;
      case "Neg":
        return `-` + input;
      case "Abs":
        return `abs(` + input + `)`;
      case "Sign":
        return `sign(` + input + `)`;
      case "Reciprocal":
        return `1.0 / ` + input;
      case "Floor":
        return `floor(` + input + `)`;
      case "Ceil":
        return `ceil(` + input + `)`;
      case "Round":
        return `round(` + input + `)`;
      case "Sqrt":
        return `sqrt(` + input + `)`;
      case "Exp":
        return `exp(` + input + `)`;
      case "Log":
        return `log(` + input + `)`;
      case "Log2":
        return `log2(` + input + `)`;
      case "Log10":
        return `log(` + input + `) / 2.302585`;
      case "Sin":
        return `sin(` + input + `)`;
      case "Cos":
        return `cos(` + input + `)`;
      case "Tan":
        return `tan(` + input + `)`;
      case "Asin":
        return `asin(` + input + `)`;
      case "Acos":
        return `acos(` + input + `)`;
      case "Atan":
        return `atan(` + input + `)`;
      case "Sinh":
        return `sinh(` + input + `)`;
      case "Cosh":
        return `cosh(` + input + `)`;
      case "Tanh":
        return `tanh(` + input + `)`;
      case "Asinh":
        return `asinh(` + input + `)`;
      case "Acosh":
        return `acosh(` + input + `)`;
      case "Atanh":
        return `atanh(` + input + `)`;
      case "ReLU":
        return `max(` + input + `, 0.0)`;
      case "Sigmoid":
        return `1.0 / (1.0 + exp(-` + input + `))`;
      case "Softplus":
        return `log(1.0 + exp(` + input + `))`;
      case "Softsign":
        return input + ` / (1.0 + abs(` + input + `))`;
      case "GeLU":
        return `0.5 * ` + input + ` * (1.0 + tanh(0.7978845608 * (` + input + ` + 0.044715 * ` + input + ` * ` + input + ` * ` + input + `)))`;
      case "SiLU":
        return input + ` / (1.0 + exp(-` + input + `))`;
      case "Mish":
        return input + ` * tanh(log(1.0 + exp(` + input + `)))`;
      case "Not":
        return `f32(` + input + ` == 0.0)`;
      default:
        return;
    }
  } else {
    switch (op.TAG) {
      case "LeakyReLU":
        return `select(` + op.alpha.toString() + ` * ` + input + `, ` + input + `, ` + input + ` > 0.0)`;
      case "ELU":
        return `select(` + op.alpha.toString() + ` * (exp(` + input + `) - 1.0), ` + input + `, ` + input + ` > 0.0)`;
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
function genReshapeKernel(size) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ` + size.toString() + `u) { return; }
  output[idx] = input[idx];
}`;
  return {
    name: "reshape_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genUnaryKernel(op, size) {
  return map(unaryExpr(op, "input0[idx]"), (expr) => {
    let wgsl = shaderHeader(2) + `

` + mainSignature + `
  if (idx >= ` + size.toString() + `u) { return; }
  output[idx] = ` + expr + `;
` + mainEnd;
    return {
      name: "unary_" + size.toString(),
      wgsl,
      bindings: [
        {
          binding: 0,
          size: size << 2,
          usage: "ReadOnly",
          name: "input0"
        },
        {
          binding: 1,
          size: size << 2,
          usage: "ReadWrite",
          name: "output"
        }
      ]
    };
  });
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
  let size = ((batch * height | 0) * width | 0) * channels | 0;
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const SIZE = ` + size.toString() + `u;
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
    name: "batchnorm_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
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
        size: size << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genConv1DKernel(batch, inLen, inC, outLen, outC, kernel, stride, pad) {
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
const PAD = ` + pad.toString() + `u;

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
function genAttentionKernel(batch, seqLen, dim) {
  let scale = 1 / Math.sqrt(dim);
  let wgsl = `@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ` + batch.toString() + `u;
const SEQ = ` + seqLen.toString() + `u;
const DIM = ` + dim.toString() + `u;
const SCALE = ` + scale.toString() + `;

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
function genClipKernel(size, minVal, maxVal) {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE = ` + size.toString() + `u;
const MIN_VAL = ` + minVal.toString() + `;
const MAX_VAL = ` + maxVal.toString() + `;

@compute @workgroup_size(` + 256 .toString() + `)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  output[idx] = clamp(input[idx], MIN_VAL, MAX_VAL);
}`;
  return {
    name: "clip_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genWhereKernel(size) {
  let wgsl = storageBuffer(0, "condition", "ReadOnly") + `
` + storageBuffer(1, "input_true", "ReadOnly") + `
` + storageBuffer(2, "input_false", "ReadOnly") + `
` + storageBuffer(3, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size.toString() + `u) { return; }
  let cond = condition[idx];
  output[idx] = select(input_false[idx], input_true[idx], cond > 0.0);
` + mainEnd;
  return {
    name: "where_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
        usage: "ReadOnly",
        name: "condition"
      },
      {
        binding: 1,
        size: size << 2,
        usage: "ReadOnly",
        name: "input_true"
      },
      {
        binding: 2,
        size: size << 2,
        usage: "ReadOnly",
        name: "input_false"
      },
      {
        binding: 3,
        size: size << 2,
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
    let size = div(((e - s | 0) + step | 0) - 1 | 0, step);
    outputShape[ax] = size;
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
function genCumsumKernel(inputShape, axis, exclusive, reverse) {
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
  let loopStart = reverse ? "axis_size - 1u" : "0u";
  let loopCond = reverse ? "i > 0u || i == 0u" : "i < axis_size";
  let loopIncr = reverse ? "i = i - 1u" : "i = i + 1u";
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
function genCastKernel(size) {
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size.toString() + `u) { return; }
  output[idx] = input[idx];
` + mainEnd;
  return {
    name: "cast_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size << 2,
        usage: "ReadWrite",
        name: "output"
      }
    ]
  };
}
function genSqueezeKernel(size) {
  let wgsl = storageBuffer(0, "input", "ReadOnly") + `
` + storageBuffer(1, "output", "ReadWrite") + `
` + mainSignature + `
  if (idx >= ` + size.toString() + `u) { return; }
  output[idx] = input[idx];
` + mainEnd;
  return {
    name: "squeeze_" + size.toString(),
    wgsl,
    bindings: [
      {
        binding: 0,
        size: size << 2,
        usage: "ReadOnly",
        name: "input"
      },
      {
        binding: 1,
        size: size << 2,
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
function genCumprodKernel(inputShape, axis, exclusive, reverse) {
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
  let loopStart = reverse ? "axis_size - 1u" : "0u";
  let loopCond = reverse ? "i > 0u || i == 0u" : "i < axis_size";
  let loopIncr = reverse ? "i = i - 1u" : "i = i + 1u";
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
function at2(arr, i) {
  return getOr(arr[i], 0);
}
function generate(op, inputShapes) {
  let outputShape = infer(op, inputShapes);
  let input = getOr(inputShapes[0], []);
  let r = input.length;
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
          let r1 = input.length;
          let r2 = s2.length;
          if (!(r1 >= 2 && r2 >= 2)) {
            return;
          }
          let m = at2(input, r1 - 2 | 0);
          let k$1 = at2(input, r1 - 1 | 0);
          let n = at2(s2, r2 - 1 | 0);
          let batchA = input.slice(0, r1 - 2 | 0);
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
          let batch = at2(input, 0);
          let height = at2(input, 1);
          let width = at2(input, 2);
          let channels = at2(input, 3);
          let k$2 = genGlobalPoolKernel("max", batch, height, width, channels);
          return [
            k$2,
            computeDispatch(batch * channels | 0, k$2.name, 0)
          ];
        case "GlobalAvgPool":
          if (r !== 4) {
            return;
          }
          let batch$1 = at2(input, 0);
          let height$1 = at2(input, 1);
          let width$1 = at2(input, 2);
          let channels$1 = at2(input, 3);
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
          return map(genReduceKernel(op.op, input, op.axes, false), (k) => [
            k,
            computeDispatch(outSize, k.name, 0)
          ]);
        case "ArgMax":
          let kernel$1 = genArgMaxKernel(input, op.axis, op.selectLastIndex);
          return [
            kernel$1,
            computeDispatch(outSize, kernel$1.name, 0)
          ];
        case "ArgMin":
          let kernel$2 = genArgMinKernel(input, op.axis, op.selectLastIndex);
          return [
            kernel$2,
            computeDispatch(outSize, kernel$2.name, 0)
          ];
        case "CumSum":
          let axis = op.axis;
          let kernel$3 = genCumsumKernel(input, axis, op.exclusive, op.reverse);
          return [
            kernel$3,
            computeDispatch(div(numElements(input), getOr(input[axis], 1)), kernel$3.name, 0)
          ];
        case "CumProd":
          let axis$1 = op.axis;
          let kernel$4 = genCumprodKernel(input, axis$1, op.exclusive, op.reverse);
          return [
            kernel$4,
            computeDispatch(div(numElements(input), getOr(input[axis$1], 1)), kernel$4.name, 0)
          ];
        case "MatMulInt4":
          let groupSize = op.groupSize;
          let s2$1 = getOr(inputShapes[1], []);
          let r1$1 = input.length;
          let m$1 = r1$1 === 1 ? 1 : at2(input, r1$1 - 2 | 0);
          let k$5 = at2(input, r1$1 - 1 | 0);
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
          return map(genTransposeKernel(input, op.perm), (k) => [
            k,
            computeDispatch(outSize, k.name, 0)
          ]);
        case "Broadcast":
          let kernel$8 = genBroadcastKernel(input, op.targetShape);
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
          let kernel$9 = genSliceKernel(input, op.starts, op.ends, op.axes, op.steps);
          return [
            kernel$9,
            computeDispatch(outSize, kernel$9.name, 0)
          ];
        case "Gather":
          let indicesShape = getOr(inputShapes[1], []);
          let indicesSize = numElements(indicesShape);
          let k$6 = genGatherKernel(input, indicesSize, op.axis);
          return [
            k$6,
            computeDispatch(outSize, k$6.name, 0)
          ];
        case "Scatter":
          let indicesShape$1 = getOr(inputShapes[1], []);
          let indicesSize$1 = numElements(indicesShape$1);
          let kernel$10 = genScatterKernel(input, indicesSize$1, op.axis);
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
          let k$7 = genSplitKernel(input, op.axis, 0, firstSplitSize);
          return [
            k$7,
            computeDispatch(outSize, k$7.name, 0)
          ];
        case "Stack":
          let numInputs = inputShapes.length;
          let kernel$11 = genStackKernel(input, numInputs, op.axis);
          return [
            kernel$11,
            computeDispatch(outSize, kernel$11.name, 0)
          ];
        case "Tile":
          let kernel$12 = genTileKernel(input, op.repeats);
          return [
            kernel$12,
            computeDispatch(outSize, kernel$12.name, 0)
          ];
        case "Pad":
          let kernel$13 = genPadKernel(input, op.pads, op.constantValue);
          return [
            kernel$13,
            computeDispatch(outSize, kernel$13.name, 0)
          ];
        case "Reverse":
          let kernel$14 = genReverseKernel(input, op.axes);
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
          let batch$2 = at2(input, 0);
          let inLen = at2(input, 1);
          let inC = at2(input, 2);
          let outLen = at2(outShape, 1);
          let pad;
          pad = typeof padding !== "object" ? padding === "Same" ? (kernel$16 - 1 | 0) / 2 | 0 : 0 : at2(padding.pads, 0);
          let k$8 = genConv1DKernel(batch$2, inLen, inC, outLen, op.filters, kernel$16, op.stride, pad);
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
          let batch$3 = at2(input, 0);
          let inH = at2(input, 1);
          let inW = at2(input, 2);
          let inC$1 = at2(input, 3);
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
          let batch$4 = at2(input, 0);
          let inH$1 = at2(input, 1);
          let inW$1 = at2(input, 2);
          let channels$2 = at2(input, 3);
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
          let batch$5 = at2(input, 0);
          let inH$2 = at2(input, 1);
          let inW$2 = at2(input, 2);
          let channels$3 = at2(input, 3);
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
          let batch$6 = at2(input, 0);
          let height$2 = at2(input, 1);
          let width$2 = at2(input, 2);
          let channels$4 = at2(input, 3);
          let kernel$20 = genBatchNormKernel(batch$6, height$2, width$2, channels$4);
          return [
            kernel$20,
            computeDispatch(outSize, kernel$20.name, 0)
          ];
        case "LayerNorm":
          let normAxis = r - 1 | 0;
          let normSize = at2(input, normAxis);
          let outerSize = div(numElements(input), normSize);
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
          let axisSize = at2(input, normAxis$1);
          let outerSize$1 = div(numElements(input), axisSize);
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
          let axisSize$1 = getOr(input[normAxis$2], 1);
          let outerSize$2 = div(numElements(input), axisSize$1);
          let kernel$22 = genLogSoftmaxKernel(outerSize$2, axisSize$1);
          return [
            kernel$22,
            computeDispatch(outerSize$2, kernel$22.name, 0)
          ];
        case "Dense":
          if (r < 1) {
            return;
          }
          let batchSize = numElements(input.slice(0, r - 1 | 0));
          let inFeatures = at2(input, r - 1 | 0);
          let kernel$23 = genDenseKernel(max(batchSize, 1), inFeatures, op.units);
          return [
            kernel$23,
            computeDispatch(outSize, kernel$23.name, 0)
          ];
        case "ScaledDotProductAttention":
          if (r !== 3) {
            return;
          }
          let batch$7 = at2(input, 0);
          let seqLen = at2(input, 1);
          let dim = at2(input, 2);
          let k$10 = genAttentionKernel(batch$7, seqLen, dim);
          return [
            k$10,
            computeDispatch(outSize, k$10.name, 0)
          ];
        case "OneHot":
          let kernel$24 = genOneHotKernel(input, op.depth);
          return [
            kernel$24,
            computeDispatch(outSize, kernel$24.name, 0)
          ];
        case "Embedding":
          let batchSeq = numElements(input);
          let k$11 = genEmbeddingKernel(batchSeq, op.numEmbeddings, op.embeddingDim);
          return [
            k$11,
            computeDispatch(outSize, k$11.name, 0)
          ];
        case "TopK":
          let k$12 = op.k;
          let kernel$25 = genTopKKernel(input, k$12, op.axis);
          let numSlices = div(outSize, k$12);
          return [
            kernel$25,
            computeDispatch(numSlices, kernel$25.name, 0)
          ];
        case "Sort":
          let axis$4 = op.axis;
          let kernel$26 = genSortKernel(input, axis$4, op.descending);
          let numSlices$1 = div(outSize, getOr(input[axis$4], 1));
          return [
            kernel$26,
            computeDispatch(numSlices$1, kernel$26.name, 0)
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
        let kernel$27 = genSqueezeKernel(outSize);
        return [
          kernel$27,
          computeDispatch(outSize, kernel$27.name, 0)
        ];
    }
  });
}

// src/Main.res.mjs
function log(prim) {
  console.log(prim);
}
function printShape(name, shape) {
  let strs = shape.map((d) => d.toString());
  let joined = strs.join(", ");
  console.log(name + ": [" + joined + "]");
}
function testInfer(name, op, inputs) {
  let s = infer(op, inputs);
  if (s !== void 0) {
    return printShape(name, s);
  } else {
    console.log(name + ": None");
    return;
  }
}
function testCodegen(name, op, inputs) {
  let match = generate(op, inputs);
  if (match !== void 0) {
    let match$1 = match[1].workgroupCount;
    let prim = name + ": " + match[0].name + " (" + match$1[0].toString() + "x" + match$1[1].toString() + ")";
    console.log(prim);
    return;
  }
  console.log(name + ": FAILED");
}
console.log("=== Shape Inference ===");
testInfer("Input", {
  TAG: "Input",
  shape: [
    32,
    224,
    224,
    3
  ],
  dtype: "F32"
}, []);
testInfer("Add broadcast", "Add", [
  [
    32,
    1,
    64
  ],
  [
    1,
    10,
    64
  ]
]);
testInfer("MatMul", "MatMul", [
  [
    32,
    128,
    64
  ],
  [
    32,
    64,
    256
  ]
]);
console.log("\n=== Codegen ===");
testCodegen("ReLU", "ReLU", [[
  4,
  256
]]);
testCodegen("Add", "Add", [
  [
    4,
    256
  ],
  [
    4,
    256
  ]
]);
testCodegen("MatMul", "MatMul", [
  [
    64,
    128
  ],
  [
    128,
    64
  ]
]);
testCodegen("Reduce Sum", {
  TAG: "Reduce",
  op: "Sum",
  axes: [-1],
  keepDims: false
}, [[
  32,
  128
]]);
testCodegen("Reduce Mean", {
  TAG: "Reduce",
  op: "Mean",
  axes: [
    1,
    2
  ],
  keepDims: false
}, [[
  8,
  32,
  32,
  3
]]);
testCodegen("Softmax", {
  TAG: "Softmax",
  axis: -1
}, [[
  4,
  1e3
]]);
testCodegen("Dense", {
  TAG: "Dense",
  units: 256,
  useBias: true
}, [[
  8,
  512
]]);
testCodegen("MaxPool2D", {
  TAG: "MaxPool2D",
  kernel: [
    2,
    2
  ],
  stride: [
    2,
    2
  ],
  padding: "Valid"
}, [[
  1,
  224,
  224,
  64
]]);
testCodegen("AvgPool2D", {
  TAG: "AvgPool2D",
  kernel: [
    2,
    2
  ],
  stride: [
    2,
    2
  ],
  padding: "Valid",
  countIncludePad: false
}, [[
  1,
  112,
  112,
  128
]]);
testCodegen("BatchNorm", {
  TAG: "BatchNorm",
  epsilon: 1e-5,
  momentum: 0.1
}, [[
  1,
  56,
  56,
  256
]]);
testCodegen("Conv1D", {
  TAG: "Conv1D",
  filters: 64,
  kernel: 3,
  stride: 1,
  padding: "Same",
  dilation: 1,
  groups: 1
}, [[
  4,
  128,
  32
]]);
testCodegen("GlobalMaxPool", "GlobalMaxPool", [[
  1,
  7,
  7,
  512
]]);
testCodegen("GlobalAvgPool", "GlobalAvgPool", [[
  1,
  7,
  7,
  512
]]);
testCodegen("LayerNorm", {
  TAG: "LayerNorm",
  axes: [-1],
  epsilon: 1e-5
}, [[
  4,
  128,
  256
]]);
testCodegen("Attention", {
  TAG: "ScaledDotProductAttention",
  dropout: 0,
  causal: false
}, [[
  2,
  64,
  128
]]);
testCodegen("Embedding", {
  TAG: "Embedding",
  numEmbeddings: 5e4,
  embeddingDim: 512
}, [[
  4,
  128
]]);
testCodegen("Clip", {
  TAG: "Clip",
  min: 0,
  max: 6
}, [[
  4,
  256
]]);
console.log("\n=== Done ===");
export {
  log,
  printShape,
  testCodegen,
  testInfer
};
