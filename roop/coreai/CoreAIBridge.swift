// C ABI over Core AI so Python can call it through ctypes.
//
//   void* cai_load(const char* path, int unit)        unit: 0 default, 1 cpu, 2 gpu, 3 ane
//   int   cai_run(void* h, int nIn, const char** inNames, const float** inData,
//                 const int64_t* inShapes, const int* inRanks,
//                 int nOut, const char** outNames, float** outData, const int64_t* outCounts)
//   void  cai_free(void* h)
//
// inShapes is all input shapes concatenated; inRanks gives each one's length.
// Output buffers are caller-allocated float32 of outCounts[i] elements.
// cai_run returns 0 on success; errors print to stderr and return nonzero.
// Core AI's API is async; each call blocks its (Python worker) thread on a
// semaphore until the inference task finishes.

import CoreAI
import Foundation

final class Handle: @unchecked Sendable {
    let model: AIModel
    let fn: InferenceFunction
    init(model: AIModel, fn: InferenceFunction) { self.model = model; self.fn = fn }
}

final class Box<T>: @unchecked Sendable { var value: T; init(_ v: T) { value = v } }

func blocking<T>(_ body: @escaping @Sendable () async throws -> T) -> Result<T, any Error> {
    let sem = DispatchSemaphore(value: 0)
    let box = Box<Result<T, any Error>?>(nil)
    Task.detached {
        do { box.value = .success(try await body()) } catch { box.value = .failure(error) }
        sem.signal()
    }
    sem.wait()
    return box.value!
}

@_cdecl("cai_load")
public func cai_load(_ path: UnsafePointer<CChar>, _ unit: Int32) -> UnsafeMutableRawPointer? {
    let url = URL(fileURLWithPath: String(cString: path))
    let options: SpecializationOptions
    switch unit {
    case 1: options = SpecializationOptions(preferredComputeUnitKind: .cpu)
    case 2: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    case 3: options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    default: options = .default
    }
    let r = blocking { () async throws -> Handle in
        let model = try await AIModel(contentsOf: url, options: options)
        let name = model.functionNames.first ?? "main"
        guard let fn = try model.loadFunction(named: name) else {
            throw NSError(domain: "cai", code: 1, userInfo: [NSLocalizedDescriptionKey: "no function \(name)"])
        }
        return Handle(model: model, fn: fn)
    }
    switch r {
    case .success(let h): return Unmanaged.passRetained(h).toOpaque()
    case .failure(let e):
        FileHandle.standardError.write("cai_load failed: \(e)\n".data(using: .utf8)!)
        return nil
    }
}

@_cdecl("cai_free")
public func cai_free(_ h: UnsafeMutableRawPointer) {
    Unmanaged<Handle>.fromOpaque(h).release()
}

/// Copy an output NDArray into a caller buffer as contiguous float32.
func copyOut(_ nd: NDArray, _ dst: UnsafeMutablePointer<Float>, _ count: Int) -> Bool {
    let shape = nd.shape
    guard shape.reduce(1, *) == count else { return false }
    var rowMajor = [Int](repeating: 1, count: shape.count)
    for d in stride(from: shape.count - 2, through: 0, by: -1) { rowMajor[d] = rowMajor[d + 1] * shape[d + 1] }
    let contiguous = nd.strides == rowMajor
    let raw = nd.rawView()
    var ok = true
    raw.withUnsafeBytes { ptr, _, _ in
        switch (nd.scalarType, contiguous) {
        case (.float32, true):
            dst.update(from: ptr.assumingMemoryBound(to: Float.self), count: count)
        case (.float16, true):
            let src = ptr.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { dst[i] = Float(src[i]) }
        case (.float32, false), (.float16, false):
            let strides = nd.strides
            var idx = [Int](repeating: 0, count: shape.count)
            for flat in 0..<count {
                var off = 0
                for d in 0..<shape.count { off += idx[d] * strides[d] }
                dst[flat] = nd.scalarType == .float32
                    ? ptr.load(fromByteOffset: off * 4, as: Float.self)
                    : Float(ptr.load(fromByteOffset: off * 2, as: Float16.self))
                var d = shape.count - 1
                while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
            }
        default:
            ok = false
        }
    }
    return ok
}

@_cdecl("cai_run")
public func cai_run(
    _ h: UnsafeMutableRawPointer,
    _ nIn: Int32, _ inNames: UnsafePointer<UnsafePointer<CChar>>,
    _ inData: UnsafePointer<UnsafePointer<Float>>,
    _ inShapes: UnsafePointer<Int64>, _ inRanks: UnsafePointer<Int32>,
    _ nOut: Int32, _ outNames: UnsafePointer<UnsafePointer<CChar>>,
    _ outData: UnsafePointer<UnsafeMutablePointer<Float>>,
    _ outCounts: UnsafePointer<Int64>
) -> Int32 {
    let handle = Unmanaged<Handle>.fromOpaque(h).takeUnretainedValue()
    var inputs = [String: NDArray]()
    var cursor = 0
    for i in 0..<Int(nIn) {
        let rank = Int(inRanks[i])
        let shape = (0..<rank).map { Int(inShapes[cursor + $0]) }
        cursor += rank
        let count = shape.reduce(1, *)
        inputs[String(cString: inNames[i])] =
            NDArray(scalars: UnsafeBufferPointer(start: inData[i], count: count), shape: shape)
    }
    let names = (0..<Int(nOut)).map { String(cString: outNames[$0]) }
    let dsts = (0..<Int(nOut)).map { (UInt(bitPattern: outData[$0]), Int(outCounts[$0])) }
    let sendableInputs = Box(inputs)

    let r = blocking { () async throws -> Int32 in
        var outputs = try await handle.fn.run(inputs: sendableInputs.value)
        for (i, name) in names.enumerated() {
            guard let value = outputs.remove(name), let nd = value.ndArray else {
                FileHandle.standardError.write("cai_run: missing output \(name)\n".data(using: .utf8)!)
                return 3
            }
            let dst = UnsafeMutablePointer<Float>(bitPattern: dsts[i].0)!
            if !copyOut(nd, dst, dsts[i].1) {
                FileHandle.standardError.write("cai_run: cannot copy output \(name) \(nd.shape) \(nd.scalarType)\n".data(using: .utf8)!)
                return 4
            }
        }
        return 0
    }
    switch r {
    case .success(let code): return code
    case .failure(let e):
        FileHandle.standardError.write("cai_run failed: \(e)\n".data(using: .utf8)!)
        return 2
    }
}
