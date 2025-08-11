//
//  Untitled.swift
//  PDSensing
//
//  Created by Betty Chen on 5/5/25.
//
import SwiftUI
import CoreML

// MARK: - Action logits
func getAction(from embedding: MLMultiArray, a: Int, b: Int, c: Int) throws -> [Float] {
    // Configure CoreML to use CPU only to avoid MPS backend issues
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    
    let model = try actor(configuration: config)
    let input = actorInput(embedding: embedding)
    let output = try model.prediction(input: input)
    let logits = output.var_19 

    let shape = logits.shape.map { $0.intValue }
    let dim1 = shape[1], dim2 = shape[2], dim3 = shape[3], dim4 = shape[4]
    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(logits.dataPointer))

    // Clamp indices to valid bounds to prevent crashes
    let clampedA = min(max(a, 0), dim1 - 1)
    let clampedB = min(max(b, 0), dim2 - 1)
    let clampedC = min(max(c, 0), dim3 - 1)
    
    if a != clampedA || b != clampedB || c != clampedC {
        print("⚠️ Action indices clamped: original(\(a),\(b),\(c)) -> clamped(\(clampedA),\(clampedB),\(clampedC))")
    }

    let flatIndex = clampedA * dim2 * dim3 * dim4 + clampedB * dim3 * dim4 + clampedC * dim4
    return Array(UnsafeBufferPointer(start: ptr + flatIndex, count: dim4))
}

// MARK: - DeltaP logits
func getDeltaP(from embedding: MLMultiArray, a: Int, b: Int, c: Int) throws -> [Float] {
    // Configure CoreML to use CPU only to avoid MPS backend issues
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    
    let model = try deltaP(configuration: config)
    let input = deltaPInput(embedding: embedding)
    let output = try model.prediction(input: input)
    let logits = output.var_19  // shape: [1, 3, 6, 5, 4]

    let shape = logits.shape.map { $0.intValue }
    let dim1 = shape[1], dim2 = shape[2], dim3 = shape[3], dim4 = shape[4]
    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(logits.dataPointer))

    // Clamp indices to valid bounds to prevent crashes
    let clampedA = min(max(a, 0), dim1 - 1)
    let clampedB = min(max(b, 0), dim2 - 1)
    let clampedC = min(max(c, 0), dim3 - 1)
    
    if a != clampedA || b != clampedB || c != clampedC {
        print("⚠️ DeltaP indices clamped: original(\(a),\(b),\(c)) -> clamped(\(clampedA),\(clampedB),\(clampedC))")
    }

    let flatIndex = clampedA * dim2 * dim3 * dim4 + clampedB * dim3 * dim4 + clampedC * dim4
    return Array(UnsafeBufferPointer(start: ptr + flatIndex, count: dim4))
}

// MARK: - TestRecommendationView
struct TestRecommendationView: View {
    let profile: UserProfile
    let embeddings: MLMultiArray

    @EnvironmentObject var userSession: UserSession
    @State private var actions: [Float] = [0, 0, 0, 0]
    @State private var deltas: [Float] = [0, 0, 0, 0]
    @State private var expandedTestID: UUID? = nil
    @State private var tests: [TestData] = []
    @State private var showSimulation = false
    @State private var selectedActivity: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                ForEach(tests.sorted(by: { $0.priority > $1.priority })) { test in
                    GroupBox {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Label(test.name, systemImage: test.icon)
                                    .font(.headline)
                                Spacer()
                                Text("Priority: \(test.priority)%")
                                    .foregroundColor(.blue)
                                    .bold()
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.1))
                                    .cornerRadius(10)
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                withAnimation(.easeInOut) {
                                    if expandedTestID == test.id {
                                        expandedTestID = nil
                                    } else {
                                        expandedTestID = test.id
                                    }
                                }
                            }

                            if expandedTestID == test.id {
                                VStack(alignment: .leading, spacing: 10) {
                                    Text(test.duration > 0 ? "Estimated Time: \(test.duration) seconds" : "Stop the test")

                                    HStack(spacing: 8) {
                                        ForEach(0..<11, id: \.self) { i in
                                            let baseRiskLevel = test.profile.pdRiskScore / 10
                                            let maxRiskLevel = min(10, (test.profile.pdRiskScore + test.delta) / 10)
                                            
                                            Image(systemName: "person.circle.fill")
                                                .foregroundColor(
                                                    i < baseRiskLevel ? .blue :
                                                    i < maxRiskLevel ? .blue.opacity(0.4) :
                                                    .gray.opacity(0.3)
                                                )
                                        }
                                    }

                                    Text("PD Risk Fluctuation with \(test.name.replacingOccurrences(of: " Test", with: "")) Data")
                                        .font(.caption)

                                    VStack(spacing: 8) {
                                        // Estimated Range (Top bubble)
                                        VStack(spacing: 2) {
                                            Text("Estimated Range")
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundColor(.secondary)
                                            Text(test.estimatedRange)
                                                .font(.system(size: 16, weight: .medium))
                                                .foregroundColor(.primary)
                                        }
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 8)
                                        .background(
                                            RoundedRectangle(cornerRadius: 20)
                                                .fill(Color.blue.opacity(0.1))
                                        )
                                        
                                        // Progress bar
                                        GeometryReader { geometry in
                                            ZStack(alignment: .leading) {
                                                // Background track with segments
                                                HStack(spacing: 2) {
                                                    ForEach(0..<10, id: \.self) { segment in
                                                        let minRange = max(0, test.profile.pdRiskScore - test.delta)
                                                        let maxRange = min(100, test.profile.pdRiskScore + test.delta)
                                                        let segmentStart = segment * 10
                                                        let segmentEnd = (segment + 1) * 10
                                                        
                                                        RoundedRectangle(cornerRadius: 3)
                                                            .fill(
                                                                segmentStart < test.profile.pdRiskScore ? Color.blue :
                                                                segmentStart < maxRange && segmentEnd > minRange ? Color.blue.opacity(0.3) :
                                                                Color.gray.opacity(0.2)
                                                            )
                                                            .frame(height: 12)
                                                    }
                                                }
                                                
                                                // Current risk position indicator
                                                HStack {
                                                    Spacer()
                                                    Image(systemName: "person.fill")
                                                        .foregroundColor(.blue)
                                                        .font(.system(size: 14, weight: .bold))
                                                        .background(
                                                            Circle()
                                                                .fill(Color.white)
                                                                .frame(width: 20, height: 20)
                                                        )
                                                    Spacer()
                                                }
                                                .frame(width: geometry.size.width)
                                                .offset(x: (CGFloat(test.profile.pdRiskScore) / 100.0) * geometry.size.width - geometry.size.width/2)
                                            }
                                        }
                                        .frame(height: 20)
                                        
                                        // Current Risk Score (Bottom bubble)
                                        VStack(spacing: 2) {
                                            Text("Risk Score")
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundColor(.secondary)
                                            Text("\(test.profile.pdRiskScore)%")
                                                .font(.system(size: 18, weight: .bold))
                                                .foregroundColor(.blue)
                                        }
                                        .padding(.horizontal, 20)
                                        .padding(.vertical, 10)
                                        .background(
                                            RoundedRectangle(cornerRadius: 20)
                                                .fill(Color.blue.opacity(0.1))
                                        )
                                    }

                                    Text("Δp = ±\(test.delta)%")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                    
                                    // Start Test Button
                                    Button(action: {
                                        if test.name != "Stop" {
                                            startTest(for: test.name)
                                        }
                                    }) {
                                        HStack {
                                            Image(systemName: test.name == "Stop" ? "stop.fill" : "play.fill")
                                            Text(test.name == "Stop" ? "End Testing Session" : "Start \(test.name)")
                                        }
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(test.name == "Stop" ? Color.red.opacity(0.8) : Color.green.opacity(0.8))
                                        .foregroundColor(.white)
                                        .cornerRadius(10)
                                        .font(.system(size: 16, weight: .medium))
                                    }
                                    .disabled(test.name == "Stop")
                                    .opacity(test.name == "Stop" ? 0.6 : 1.0)
                                }
                                .transition(.opacity.combined(with: .scale))
                            }
                        }
                        .padding()
                        .background(test.color)
                        .cornerRadius(10)
                        .scaleEffect(expandedTestID == test.id ? 1.03 : 1.0)
                        .shadow(color: expandedTestID == test.id ? .gray.opacity(0.3) : .clear, radius: 8)
                        .animation(.easeInOut(duration: 0.25), value: expandedTestID)
                    }
                }
            }
            .padding()
        }
        .onAppear {
            // Ensure userSession has the correct username for data collection
            userSession.username = profile.username
            
            do {
                // Convert 1-based counts to 0-based indices, ensuring non-negative values
                let a = max(0, profile.walkingCount - 1)
                let b = max(0, profile.tappingCount - 1)
                let c = max(0, profile.voiceCount - 1)
                actions = try getAction(from: embeddings, a: a, b: b, c: c)
                deltas = try getDeltaP(from: embeddings, a: a, b: b, c: c)

                self.tests = [
                    TestData(name: "Tapping Test", icon: "hand.tap", priority: clamp(actions[1]), duration: 20, delta: clamp(deltas[1]), profile: profile),
                    TestData(name: "Voice Test", icon: "mic.fill", priority: clamp(actions[2]), duration: 10, delta: clamp(deltas[2]), profile: profile),
                    TestData(name: "Walking Test", icon: "figure.walk", priority: clamp(actions[0]), duration: 50, delta: clamp(deltas[0]), profile: profile),
                    TestData(name: "Stop", icon: "stop.fill", priority: clamp(actions[3]), duration: 0, delta: clamp(deltas[3]), profile: profile)
                ]
            } catch {
                print("❌ Failed to load actions or deltas:", error)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text("Activity Recommendation")
                    .font(.headline)
            }
        }
        .navigationDestination(isPresented: $showSimulation) {
            if let activity = selectedActivity {
                DataCollectionSimulationView(activity: activity)
                    .environmentObject(userSession)
            } else {
                Text("No activity selected.")
            }
        }
    }

    private func clamp(_ value: Float) -> Int {
        return min(max(Int(value * 100), 0), 100)
    }
    
    private func startTest(for testName: String) {
        // Extract activity name from test name (remove " Test" suffix)
        let activityName = testName.replacingOccurrences(of: " Test", with: "")
        selectedActivity = activityName
        showSimulation = true
    }
}

// MARK: - TestData
struct TestData: Identifiable {
    let id = UUID()
    let name: String
    let icon: String
    let priority: Int
    let duration: Int
    let delta: Int
    let profile: UserProfile
    var color: Color = Color.primary.opacity(0.05)

    var estimatedRange: String {
        let lower = max(0, profile.pdRiskScore - delta)
        let upper = min(100, profile.pdRiskScore + delta)
        return "\(lower)% ~ \(upper)%"
    }
}

#Preview {
    NavigationView {
        TestRecommendationView(
            profile: UserProfile(
                username: "preview_user",
                name: "John Doe",
                age: 55,
                weight: 80.0,
                gender: "Male",
                id: "XYZ789",
                pdRiskScore: 68,
                walkingCount: 3,
                tappingCount: 2,
                voiceCount: 4
            ),
            embeddings: try! MLMultiArray(shape: [1, 3, 6, 5, 64], dataType: .float32)
        )
        .environmentObject(UserSession())
    }
}

