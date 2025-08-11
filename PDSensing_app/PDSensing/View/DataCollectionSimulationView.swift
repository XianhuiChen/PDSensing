//
//  DataCollectionSimulationView.swift
//  PDSensing
//
//  Created for simulating data collection process
//

import SwiftUI

struct DataCollectionSimulationView: View {
    let activity: String
    @State private var isComplete = false
    @State private var showingProfile = false
    @EnvironmentObject var userSession: UserSession
    @State private var userProfile: UserProfile?
    
    var body: some View {
        VStack(spacing: 40) {
            Spacer()
            
            // Activity Icon
            Image(activityIconName)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 100, height: 100)
            
            VStack(spacing: 20) {
                if !isComplete {
                    // Collection in progress
                    Text("Collecting \(activity.lowercased()) data")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.primary)
                    
                    Text("Please follow the instructions on the testing manual")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                    
                    // Loading indicator
                    ProgressView()
                        .scaleEffect(1.5)
                        .padding(.top, 20)
                        
                } else {
                    // Collection complete
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.green)
                    
                    Text("Complete!")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
            }
            
            Spacer()
            
            if isComplete {
                Button(action: {
                    recordDataCollection(for: activity)
                    showingProfile = true
                }) {
                    Text("Continue")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                        .font(.title3)
                }
                .padding(.horizontal, 40)
            }
        }
        .padding()
        .navigationTitle("\(activity) Test")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadProfile()
            // Start 3-second timer
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                withAnimation(.easeInOut(duration: 0.5)) {
                    isComplete = true
                }
            }
        }
        .navigationDestination(isPresented: $showingProfile) {
            if let profile = userProfile {
                UserProfileView(user: profile)
                    .environmentObject(userSession)
            } else {
                Text("No profile loaded.")
            }
        }
    }
    
    private var activityIconName: String {
        switch activity {
        case "Walking":
            return "walking_icon"
        case "Tapping":
            return "tapping_icon"
        case "Voice":
            return "voice_icon"
        default:
            return "questionmark"
        }
    }
    

    
    private func loadProfile() {
        if let profile = loadUserProfile(username: userSession.username) {
            self.userProfile = profile
        }
    }
    
    private func recordDataCollection(for activity: String) {
        guard var profile = userProfile else { return }

        switch activity {
        case "Walking":
            profile.walkingCount += 1
        case "Tapping":
            profile.tappingCount += 1
        case "Voice":
            profile.voiceCount += 1
        default:
            break
        }
        
        saveUserProfile(profile)
        self.userProfile = profile
        print("Collected data for \(activity), walking: \(profile.walkingCount), tapping: \(profile.tappingCount), voice: \(profile.voiceCount)")
    }
}

#Preview {
    DataCollectionSimulationView(activity: "Walking")
        .environmentObject(UserSession())
} 