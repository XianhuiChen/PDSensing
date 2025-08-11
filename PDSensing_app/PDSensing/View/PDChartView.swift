//
//  PDChartView.swift
//  PDSensing
//
//  Created by Betty Chen on 5/5/25.
//

import SwiftUI
import Charts
import Foundation
import CoreML



// MARK: - NPY Loader

func loadNpyFloat32(fileURL: URL) throws -> (data: [Float], shape: [Int]) {
    let fileData = try Data(contentsOf: fileURL)

    // Check magic number
    let magic = String(bytes: fileData.prefix(6), encoding: .ascii)
    guard magic == "\u{93}NUMPY" else {
        throw NSError(domain: "NPYError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Not a valid .npy file"])
    }

    // Read header length (bytes 8–9)
    let headerLen = Int(fileData[8]) + Int(fileData[9]) << 8
    let headerData = fileData.subdata(in: 10..<10+headerLen)

    guard let header = String(data: headerData, encoding: .ascii) else {
        throw NSError(domain: "NPYError", code: -2, userInfo: [NSLocalizedDescriptionKey: "Header not ASCII"])
    }

    // Extract shape from header string
    let shapeMatch = header.range(of: #"shape\s*:\s*\(([^)]*)\)"#, options: .regularExpression)
    guard let match = shapeMatch else {
        throw NSError(domain: "NPYError", code: -3, userInfo: [NSLocalizedDescriptionKey: "Shape not found in header"])
    }

    let shapeString = header[match]
        .replacingOccurrences(of: "shape", with: "")
        .replacingOccurrences(of: ":", with: "")
        .replacingOccurrences(of: "(", with: "")
        .replacingOccurrences(of: ")", with: "")
        .trimmingCharacters(in: .whitespaces)

    let shape = shapeString.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

    // Read actual data
    let offset = 10 + headerLen
    let dataBytes = fileData.dropFirst(offset)
    let count = dataBytes.count / MemoryLayout<Float>.size

    let floatArray = dataBytes.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: count))
    }

    return (data: floatArray, shape: shape)
}

// MARK: - Json Loader
func loadJsonFloat32(fileURL: URL) throws -> (data: [Float], shape: [Int]) {
    let jsonData = try Data(contentsOf: fileURL)
    let decoded = try JSONSerialization.jsonObject(with: jsonData, options: [])

    func extractShape(_ obj: Any) -> [Int] {
        if let arr = obj as? [Any], let first = arr.first {
            return [arr.count] + extractShape(first)
        }
        return []
    }

    func flatten(_ obj: Any) -> [Float] {
        if let num = obj as? NSNumber {
            return [num.floatValue]
        } else if let arr = obj as? [Any] {
            return arr.flatMap { flatten($0) }
        } else {
            return []
        }
    }

    guard let root = decoded as? [Any] else {
        throw NSError(domain: "JSONError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Expected JSON array at root"])
    }

    let shape = extractShape(root)
    let data = flatten(root)

    return (data, shape)
}


// MARK: - MLMultiArray Converter

func createMLMultiArray(from floats: [Float], shape: [Int]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)

    let pointer = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
    for i in 0..<floats.count {
        pointer[i] = floats[i]
    }

    return array
}
// MARK: Get Embeeding vector
func getEmbeddingVector(from array: MLMultiArray, a: Int, b: Int, c: Int) -> [Float] {
    let shape = array.shape.map { $0.intValue }  // [1, A, B, C, 64]
    guard shape.count == 5 else {
        print("❌ Shape is not 5D: \(shape)")
        return []
    }

    let dim1 = shape[1], dim2 = shape[2], dim3 = shape[3], dim4 = shape[4]  // 64 is dim4
    
    // Clamp indices to valid bounds to prevent crashes
    let clampedA = min(max(a, 0), dim1 - 1)
    let clampedB = min(max(b, 0), dim2 - 1)
    let clampedC = min(max(c, 0), dim3 - 1)
    
    if a != clampedA || b != clampedB || c != clampedC {
        print("⚠️ Embedding indices clamped: original(\(a),\(b),\(c)) -> clamped(\(clampedA),\(clampedB),\(clampedC))")
    }

    // Calculate flat index using clamped values
    let flatIndex = (
        0 * dim1 * dim2 * dim3 * dim4 +
        clampedA * dim2 * dim3 * dim4 +
        clampedB * dim3 * dim4 +
        clampedC * dim4
    )

    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
    let vector = Array(UnsafeBufferPointer(start: ptr + flatIndex, count: dim4))  // dim4 = 64

    return vector
}

// MARK: - Data Loading Test Function

/// Test function to verify data files are accessible from bundle
func testDataFilesAccess(for username: String) -> Bool {
    guard let baseURL = Bundle.main.resourceURL else {
        print("❌ Failed to find Bundle resource URL")
        return false
    }

    // Try to load data from user-specific folder first, fall back to Samuel_Hale
    let userDataURL = baseURL.appendingPathComponent("data").appendingPathComponent(username)
    let fallbackDataURL = baseURL.appendingPathComponent("data").appendingPathComponent("Samuel_Hale")
    
    // Check if user-specific data exists, otherwise use Samuel_Hale as template
    let dataSourceURL: URL
    if FileManager.default.fileExists(atPath: userDataURL.path) {
        dataSourceURL = userDataURL
        print("🔍 Testing data files access for user '\(username)' from user-specific folder")
    } else {
        dataSourceURL = fallbackDataURL
        print("🔍 Testing data files access for user '\(username)' from template folder: Samuel_Hale")
    }
    
    let dataFiles = ["walking.json", "tapping.json", "tapping_accel.json", "voice.json"]
    print("📂 Data source path: \(dataSourceURL.path)")
    
    for fileName in dataFiles {
        let fileURL = dataSourceURL.appendingPathComponent(fileName)
        let exists = FileManager.default.fileExists(atPath: fileURL.path)
        print("📄 \(fileName): \(exists ? "✅ Found" : "❌ Missing") at \(fileURL.path)")
        
        if !exists {
            return false
        }
    }
    
    print("🎉 All data files are accessible for user '\(username)'!")
    return true
}

// MARK: - Core Prediction Pipeline

func runPrediction(profile: inout UserProfile) throws -> (Float, MLMultiArray) {
    print("🔄 Starting runPrediction...")
    
    guard let baseURL = Bundle.main.resourceURL else {
        fatalError("❌ Failed to find Bundle resource URL")
    }

    // Try to load data from user-specific folder first, fall back to Samuel_Hale
    let userDataURL = baseURL.appendingPathComponent("data").appendingPathComponent(profile.username)
    let fallbackDataURL = baseURL.appendingPathComponent("data").appendingPathComponent("Samuel_Hale")
    
    // Check if user-specific data exists, otherwise use Samuel_Hale as template
    let dataSourceURL: URL
    if FileManager.default.fileExists(atPath: userDataURL.path) {
        dataSourceURL = userDataURL
        print("🔄 Loading JSON data from user-specific folder: \(profile.username)")
    } else {
        dataSourceURL = fallbackDataURL
        print("🔄 Loading JSON data from template folder: Samuel_Hale (user \(profile.username) folder not found)")
    }
    
    let walkingURL = dataSourceURL.appendingPathComponent("walking.json")
    let tappingURL = dataSourceURL.appendingPathComponent("tapping.json")
    let tappingAccelURL = dataSourceURL.appendingPathComponent("tapping_accel.json")
    let voiceURL = dataSourceURL.appendingPathComponent("voice.json")
  
    let (walkingData, walkingShape) = try loadJsonFloat32(fileURL: walkingURL)
    print("✅ Walking data loaded:", walkingShape)
    let (tappingData, tappingShape) = try loadJsonFloat32(fileURL: tappingURL)
    print("✅ Tapping data loaded:", tappingShape)
    let (tappingAccelData, tappingAccelShape) = try loadJsonFloat32(fileURL: tappingAccelURL)
    print("✅ Tapping accel data loaded:", tappingAccelShape)
    let (voiceData, voiceShape) = try loadJsonFloat32(fileURL: voiceURL)
    print("✅ Voice data loaded:", voiceShape)
    
    print("🔄 Creating MLMultiArrays...")
    let walkingArray = try createMLMultiArray(from: walkingData, shape: walkingShape)
    print("✅ Walking MLMultiArray created")
    let tappingArray = try createMLMultiArray(from: tappingData, shape: tappingShape)
    print("✅ Tapping MLMultiArray created")
    let tappingAccelArray = try createMLMultiArray(from: tappingAccelData, shape: tappingAccelShape)
    print("✅ Tapping accel MLMultiArray created")
    let voiceArray = try createMLMultiArray(from: voiceData, shape: voiceShape)
    print("✅ Voice MLMultiArray created")

    // Configure CoreML to use CPU only to avoid MPS backend issues
    print("🔄 Configuring CoreML model...")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    
    // 2. Run Predictor Model
    print("🔄 Loading predictor model...")
    let model = try predictor(configuration: config)
    print("✅ Predictor model loaded successfully")
    
    print("🔄 Creating model input...")
    let input = predictorInput(
        walking_data: walkingArray,
        tapping_data: tappingArray,
        tapping_accel_data: tappingAccelArray,
        voice_data: voiceArray
    )
    print("✅ Model input created")

    print("🔄 Running model prediction... (this may take a moment)")
    let output = try model.prediction(input: input)
    print("✅ Model prediction completed!")

    let embeddings = output.input_155
    let pd_probs = output.var_1016
    print("✅ Extracted embeddings and probabilities, shapes:", embeddings.shape, pd_probs.shape)
    
    // 3. Calculate index with bounds checking to prevent crashes
    let pdProbsShape = pd_probs.shape.map { $0.intValue }
    guard pdProbsShape.count >= 5 else {
        throw NSError(domain: "PredictionError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid pd_probs array shape"])
    }
    
    // Get the maximum valid indices (subtract 1 because indices are 0-based)
    let maxA = pdProbsShape[1] - 1
    let maxB = pdProbsShape[2] - 1
    let maxC = pdProbsShape[3] - 1
    
    // Clamp the indices to valid bounds - if requested count exceeds available data, use the last valid index
    // Convert 1-based counts to 0-based indices, ensuring non-negative values
    let a = min(max(0, profile.walkingCount - 1), maxA)
    let b = min(max(0, profile.tappingCount - 1), maxB)
    let c = min(max(0, profile.voiceCount - 1), maxC)
    
    // Track if any indices were clamped (meaning we exceeded available data)
    // Only consider clamped if we have tests performed and exceed available data
    let wasWalkingClamped = profile.walkingCount > 0 && (profile.walkingCount - 1) > maxA
    let wasTappingClamped = profile.tappingCount > 0 && (profile.tappingCount - 1) > maxB
    let wasVoiceClamped = profile.voiceCount > 0 && (profile.voiceCount - 1) > maxC
    let anyDataClamped = wasWalkingClamped || wasTappingClamped || wasVoiceClamped
    
    // Log warning if data was clamped
    if anyDataClamped {
        print("⚠️ Data indices clamped - using last available data:")
        if wasWalkingClamped {
            print("   Walking: requested \(profile.walkingCount), using \(a)")
        }
        if wasTappingClamped {
            print("   Tapping: requested \(profile.tappingCount), using \(b)")
        }
        if wasVoiceClamped {
            print("   Voice: requested \(profile.voiceCount), using \(c)")
        }
    }

    let batchIndex = profile.tappingCount  // use tappingCount for batch

    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(pd_probs.dataPointer))
    
    // 5. Extract pd_prob[a][b][c] with bounds-checked indices
    let index = [NSNumber(value: 0), NSNumber(value: a), NSNumber(value: b), NSNumber(value: c), NSNumber(value: 0)]
    let riskScore = pd_probs[index].floatValue
    let embedding = getEmbeddingVector(from: embeddings, a: a, b: b, c: c)
    let embeddingArray = embeddings
    
    // 6. Append new DataRecord with flag indicating if data was clamped
    let record = DataRecord(
        id: profile.dataRecords.count,
        timestamp: Date(),  // current date
        pdRiskScore: Int(riskScore * 100),  // optional: scale or round as needed
        wasDataClamped: anyDataClamped
    )
    profile.dataRecords.append(record)
    profile.pdRiskScore = Int(riskScore * 100)
    print("Updated score: \(profile.pdRiskScore)")
    // 7. Save  updated profile
    saveUserProfile(profile)

    return (riskScore, embeddingArray)
}

// MARK: - Helpers

func extractBatch(_ array: MLMultiArray, batch: Int) throws -> MLMultiArray {
    let shape = array.shape.map { $0.intValue }
    guard shape.count >= 2 else {
        throw NSError(domain: "extractBatch", code: -1, userInfo: [NSLocalizedDescriptionKey: "Array must have at least 2 dimensions"])
    }

    // Calculate the number of elements in one batch
    let batchSize = shape.dropFirst().reduce(1, *)

    let offset = batch * batchSize

    // Create new MLMultiArray for the single batch
    let newShape = Array(shape.dropFirst())
    let newArray = try MLMultiArray(shape: newShape.map { NSNumber(value: $0) }, dataType: .float32)

    let srcPointer = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
    let dstPointer = UnsafeMutablePointer<Float>(OpaquePointer(newArray.dataPointer))

    for i in 0..<batchSize {
        dstPointer[i] = srcPointer[offset + i]
    }

    return newArray
}

struct PDChartView: View {
   
    let profile: UserProfile
    let threshold: Int = 50

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Label("Risk Score", systemImage: "line.diagonal")
                        .foregroundColor(.blue)
                        .font(.caption)
                    Label("Threshold", systemImage: "line.diagonal")
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }

            Chart {
                ForEach(Array(profile.dataRecords.enumerated()), id: \.element.id) { index, record in
                    LineMark(
                        x: .value("Index", index),
                        y: .value("Score", record.pdRiskScore)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(Color.blue)
                    .symbol(Circle())
                }

                RuleMark(y: .value("Threshold", threshold))
                    .foregroundStyle(Color.red)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5]))
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: 1)) { index in
                    if let intIndex = index.as(Int.self),
                       intIndex < profile.dataRecords.count {
                        let date = profile.dataRecords[intIndex].timestamp
                        AxisValueLabel(formattedDate(date), orientation: .vertical)
                    }
                }
            }
            .chartXScale(domain: profile.dataRecords.count > 0 ? 0...(profile.dataRecords.count - 1) : 0...0)
            .frame(height: 200)
            .padding(.vertical, 10)
        }
        .padding(.horizontal)
    }

    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd"
        return formatter.string(from: date)
    }
}

#Preview {
    PDChartView(profile: UserProfile(
        username: "preview_user",
        name: "Sample User",
        age: 50,
        weight: 70.0,
        gender: "Male",
        id: "ABC123",
        pdRiskScore: 75,
        walkingCount: 4,
        tappingCount: 3,
        voiceCount: 2,
        dataRecords: [
            DataRecord(id: 0, timestamp: Date().addingTimeInterval(-432000), pdRiskScore: 45),
            DataRecord(id: 1, timestamp: Date().addingTimeInterval(-345600), pdRiskScore: 52),
            DataRecord(id: 2, timestamp: Date().addingTimeInterval(-259200), pdRiskScore: 68),
            DataRecord(id: 3, timestamp: Date().addingTimeInterval(-172800), pdRiskScore: 71),
            DataRecord(id: 4, timestamp: Date().addingTimeInterval(-86400), pdRiskScore: 75)
        ]
    ))
}
