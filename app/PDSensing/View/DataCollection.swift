//
//  DataCollection.swift
//  PDSensing
//
//  Created by Betty Chen on 5/5/25.
//
import SwiftUI

struct DataCollectionView: View {
    @EnvironmentObject var userSession: UserSession
    @Binding var isLoggedIn: Bool  

    @State private var userProfile: UserProfile?
    @State private var showSimulation = false
    @State private var selectedActivity: String?
    

    var body: some View {
        VStack(spacing: 30) {
            ForEach(["Walking", "Tapping", "Voice"], id: \.self) { activity in
                Button(action: {
                    selectedActivity = activity
                    showSimulation = true
                }) {
                    Text(activity)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(12)
                        .font(.title3)
                }
            }
        }
        .padding()
        .navigationTitle("Collect Data")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: logout) {
                    Label("Log out", systemImage: "arrowshape.turn.up.left")
                        .foregroundColor(.red)
                }
            }
        }
        .onAppear {
            loadProfile()
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

    private func loadProfile() {
        if let profile = loadUserProfile(username: userSession.username) {
            self.userProfile = profile
        }
    }

    private func logout() {
        userSession.isLoggedIn = false
        userSession.username = ""
        isLoggedIn = false
        print("User logged out from Data Collection View")
    }
}

#Preview {
    NavigationView {
        DataCollectionView(isLoggedIn: .constant(true))
            .environmentObject(UserSession())
    }
}
