//
//  UserProfileView.swift
//  PDSensing
//
//  Created by Betty Chen on 5/5/25.
//

import SwiftUI
import CoreML

struct UserProfile: Codable {
    var username: String
    var name: String
    var age: Int
    var weight: Double
    var gender: String
    var id: String
    var pdRiskScore: Int

    var walkingCount: Int = 0
    var tappingCount: Int = 0
    var voiceCount: Int = 0
    var lastEmbedding: MLMultiArray? = nil

    var dataRecords: [DataRecord] = []

    enum CodingKeys: String, CodingKey {
        case username, name, age, weight, gender, id, pdRiskScore,
             walkingCount, tappingCount, voiceCount, dataRecords
    }
}

struct DataRecord: Codable {
    var id: Int
    var timestamp: Date
    var pdRiskScore: Int
    var wasDataClamped: Bool = false
    
    enum CodingKeys: String, CodingKey {
        case id, timestamp, pdRiskScore, wasDataClamped
    }
}

func saveUserProfile(_ profile: UserProfile) {
    let encoder = JSONEncoder()
    do {
        let data = try encoder.encode(profile)
        let documentsURL = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let fileURL = documentsURL.appendingPathComponent("user_\(profile.username)_profile.json")
        try data.write(to: fileURL)
        print("Saved profile for user: \(profile.username)")
    } catch {
        print("Failed to save profile: \(error)")
    }
}

func loadUserProfile(username: String) -> UserProfile? {
    let decoder = JSONDecoder()
    do {
        let documentsURL = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let fileURL = documentsURL.appendingPathComponent("user_\(username)_profile.json")
        let data = try Data(contentsOf: fileURL)
        return try decoder.decode(UserProfile.self, from: data)
    } catch {
        print("Failed to load profile: \(error)")
        return nil
    }
}

struct UserProfileView: View {
    @EnvironmentObject var userSession: UserSession
    @State private var showRecommendationView = false
    @State private var embeddings: MLMultiArray? = nil
    @State private var isLoading = true
    @State private var hasLoaded = false
    @State var user: UserProfile

    var body: some View {
        Group {
            if isLoading {
                ProgressView("Running prediction...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if hasNoTestData {
                // Show message when user hasn't performed any tests yet
                VStack(spacing: 20) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.system(size: 60))
                        .foregroundColor(.gray)
                    
                    Text("No Test Data Available")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Text("Please complete at least one test (Walking, Tapping, or Voice) to see your PD risk analysis.")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                    
                    // Show basic profile info
                    VStack(alignment: .leading, spacing: 16) {
                        Text("User Profile")
                            .font(.title).bold()

                        HStack(alignment: .top, spacing: 16) {
                            VStack(alignment: .leading, spacing: 12) {
                                InfoItem(label: "Name", value: user.name)
                                InfoItem(label: "Age", value: "\(user.age)")
                                InfoItem(label: "Weight (kg)", value: "\(user.weight)")
                            }
                            Spacer()
                            VStack(alignment: .leading, spacing: 12) {
                                InfoItem(label: "Gender", value: user.gender)
                                InfoItem(label: "ID", value: user.id)
                            }
                        }
                    }
                    .padding()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            Text("User Profile")
                                .font(.title).bold()

                            HStack(alignment: .top, spacing: 16) {
                                VStack(alignment: .leading, spacing: 12) {
                                    InfoItem(label: "Name", value: user.name)
                                    InfoItem(label: "Age", value: "\(user.age)")
                                    InfoItem(label: "Weight (kg)", value: "\(user.weight)")
                                }
                                Spacer()
                                VStack(alignment: .leading, spacing: 12) {
                                    InfoItem(label: "PD Risk Score", value: "\(user.pdRiskScore)", color: .orange)
                                    InfoItem(label: "Gender", value: user.gender)
                                    InfoItem(label: "ID", value: user.id)
                                }
                                // Image(systemName: "person.circle.fill")
                                //     .font(.system(size: 30))
                                //     .foregroundColor(.blue)
                            }

                            Text("Test Result")
                                .font(.title2).bold()

                            PDChartView(profile: user)

                            HStack(spacing: 4) {
                                Text("Your PD risk is higher than")
                                Text("\(user.pdRiskScore)%")
                                    .bold()
                                    .foregroundColor(.orange)
                                Text("people")
                            }

                            HStack(spacing: 8) {
                                ForEach(0..<11) { i in
                                    Image(systemName: "person.circle.fill")
                                        .foregroundColor(i < user.pdRiskScore / 10 ? .blue : .gray)
                                }
                            }

                            ProgressView(value: Double(user.pdRiskScore) / 100.0)
                                .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                                .scaleEffect(x: 1, y: 2, anchor: .center)
                                .padding(.vertical, 4)

                            Button {
                                showRecommendationView = true
                            } label: {
                                Label("Next", systemImage: "play.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                        .padding()
                        .navigationDestination(isPresented: $showRecommendationView) {
                            if let unwrappedEmbedding = embeddings {
                                TestRecommendationView(profile: user, embeddings: unwrappedEmbedding)
                            } else {
                                Text("Embeddings not available.")
                            }
                        }
                    }
                }
            }
        }
        .onAppear {
            guard !hasLoaded else { return }
            hasLoaded = true
            
            // Check for data clamping and log warning to terminal
            if hasDataBeenClamped {
                print("⚠️ WARNING: The requested number of data points exceeds available records. The last available record has been used.")
            }
            
            // Only run predictions if user has performed tests
            guard !hasNoTestData else {
                isLoading = false
                return
            }
            
            // Test data files accessibility first
            print("🧪 Testing data files accessibility...")
            let dataAccessible = testDataFilesAccess(for: user.username)
            if !dataAccessible {
                DispatchQueue.main.async {
                    self.isLoading = false
                    print("❌ Data files not accessible - prediction cannot run")
                }
                return
            }

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    print("🔄 Starting prediction on background thread...")
                    
                    // Add timeout mechanism
                    let predictionGroup = DispatchGroup()
                    var predictionResult: (Float, MLMultiArray)?
                    var predictionError: Error?
                    
                    predictionGroup.enter()
                    DispatchQueue.global(qos: .utility).async {
                        do {
                            let result = try runPrediction(profile: &user)
                            predictionResult = result
                        } catch {
                            predictionError = error
                        }
                        predictionGroup.leave()
                    }
                    
                    // Wait for prediction with timeout
                    let timeoutResult = predictionGroup.wait(timeout: .now() + 30) // 30 second timeout
                    
                    if timeoutResult == .timedOut {
                        print("⏰ Prediction timed out after 30 seconds")
                        DispatchQueue.main.async {
                            self.isLoading = false
                        }
                        return
                    }
                    
                    if let error = predictionError {
                        throw error
                    }
                    
                    guard let (score, resultEmbedding) = predictionResult else {
                        throw NSError(domain: "PredictionError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Prediction returned no result"])
                    }
                    
                    print("✅ Prediction completed successfully, score: \(score)")
                    
                    DispatchQueue.main.async {
                        self.embeddings = resultEmbedding
                        self.isLoading = false
                        print("✅ UI updated with prediction results")
                    }
                } catch {
                    print("❌ Prediction failed: \(error)")
                    DispatchQueue.main.async {
                        self.isLoading = false
                    }
                }
            }
        }
    }
    
    // Check if user has performed any tests
    private var hasNoTestData: Bool {
        return user.walkingCount == 0 && user.tappingCount == 0 && user.voiceCount == 0
    }
    
    // Computed property to check if any recent data record was clamped
    private var hasDataBeenClamped: Bool {
        return user.dataRecords.last?.wasDataClamped ?? false
    }
}

#Preview {
    UserProfileView(user: UserProfile(
        username: "preview_user",
        name: "John Doe",
        age: 45,
        weight: 75.0,
        gender: "Male",
        id: "ABC123",
        pdRiskScore: 65,
        walkingCount: 5,
        tappingCount: 3,
        voiceCount: 2,
        dataRecords: [
            DataRecord(id: 1, timestamp: Date(), pdRiskScore: 60),
            DataRecord(id: 2, timestamp: Date().addingTimeInterval(-86400), pdRiskScore: 65),
            DataRecord(id: 3, timestamp: Date().addingTimeInterval(-172800), pdRiskScore: 70)
        ]
    ))
    .environmentObject(UserSession())
}
