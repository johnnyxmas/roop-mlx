// Core AI benchmark harness for the roop-mlx spike.
//
// Usage: coreai-bench <model.aimodel> <io-dir> <default|cpu|gpu|ane> [runs]
//
// <io-dir>/manifest.json lists the inputs written by the Python driver:
//   {"inputs": [{"name": "target", "shape": [1,3,256,256], "file": "target.bin"}]}
// Each .bin is raw little-endian float32 in row-major order.  Outputs are
// written back as <io-dir>/out_<name>.bin (float32, row-major) and their
// shapes plus timings go to <io-dir>/result.json.

import CoreAI
import Foundation

struct InputSpec: Decodable { let name: String; let shape: [Int]; let file: String }
struct Manifest: Decodable { let inputs: [InputSpec] }

func ms(_ d: Duration) -> Double {
    Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

/// Copy an NDArray out as contiguous row-major float32, honouring strides
/// (assumed to be in elements) and converting float16 if needed.
func contiguousFloats(_ array: NDArray) -> [Float] {
    let shape = array.shape
    let count = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: count)
    let raw = array.rawView()
    raw.withUnsafeBytes { ptr, _, stridesSpan in
        var strides = [Int](); for i in 0..<stridesSpan.count { strides.append(stridesSpan[i]) }
        var index = [Int](repeating: 0, count: shape.count)
        for flat in 0..<count {
            var rem = flat
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                index[d] = rem % shape[d]; rem /= shape[d]
            }
            var offset = 0
            for d in 0..<shape.count { offset += index[d] * strides[d] }
            switch array.scalarType {
            case .float32: out[flat] = ptr.load(fromByteOffset: offset * 4, as: Float.self)
            case .float16: out[flat] = Float(ptr.load(fromByteOffset: offset * 2, as: Float16.self))
            default: out[flat] = .nan
            }
        }
    }
    return out
}

@main
struct CoreAIBench {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 4 else {
            print("usage: coreai-bench <model.aimodel> <io-dir> <default|cpu|gpu|ane> [runs]")
            exit(2)
        }
        let modelURL = URL(fileURLWithPath: args[1])
        let ioDir = URL(fileURLWithPath: args[2])
        let runs = args.count > 4 ? Int(args[4]) ?? 30 : 30

        let options: SpecializationOptions
        switch args[3] {
        case "cpu": options = SpecializationOptions(preferredComputeUnitKind: .cpu)
        case "gpu": options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        case "ane": options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case "cpuonly": options = .cpuOnly
        default: options = .default
        }

        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: ioDir.appendingPathComponent("manifest.json")))
        var inputs = [String: NDArray]()
        for spec in manifest.inputs {
            let data = try Data(contentsOf: ioDir.appendingPathComponent(spec.file))
            let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            inputs[spec.name] = NDArray(scalars: floats, shape: spec.shape)
        }

        let clock = ContinuousClock()
        let t0 = clock.now
        let model = try await AIModel(contentsOf: modelURL, options: options)
        let tLoad = clock.now
        let fnName = model.functionNames.first ?? "main"
        guard let fn = try model.loadFunction(named: fnName) else {
            print("no function \(fnName)"); exit(1)
        }

        // First run (includes any lazy setup), then warmup, then timed runs.
        let t1 = clock.now
        var outputs = try await fn.run(inputs: inputs)
        let tFirst = clock.now
        for _ in 0..<3 { _ = try await fn.run(inputs: inputs) }
        var times = [Double]()
        for _ in 0..<runs {
            let s = clock.now
            _ = try await fn.run(inputs: inputs)
            times.append(ms(clock.now - s))
        }
        times.sort()

        var shapes = [String: [Int]]()
        for name in Array(outputs.names) {
            guard let value = outputs.remove(name), let nd = value.ndArray else { continue }
            let floats = contiguousFloats(nd)
            shapes[name] = nd.shape
            try floats.withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: ioDir.appendingPathComponent("out_\(name).bin"))
        }

        let result: [String: Any] = [
            "compute": args[3],
            "load_ms": ms(tLoad - t0),
            "first_run_ms": ms(tFirst - t1),
            "median_ms": times[times.count / 2],
            "p10_ms": times[times.count / 10],
            "p90_ms": times[times.count * 9 / 10],
            "output_shapes": shapes,
            "architecture": AIModel.deviceArchitectureName,
        ]
        let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: ioDir.appendingPathComponent("result.json"))
        print(String(data: json, encoding: .utf8)!)
    }
}
