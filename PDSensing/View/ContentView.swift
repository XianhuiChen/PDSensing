///
//  ContentView.swift
//  PDSensing
//
//  Created by Xianhui Chen on 4/25/25.
//

import SwiftUI
import Charts

func generateSimpleID() -> String {
    let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return String((0..<6).map { _ in characters.randomElement()! })
}



struct ContentView: View {
    @StateObject private var userSession = UserSession()
    @State private var isLoggedIn = false
    @State private var showRegister = false
    @State private var users: [String: String] = ["AIMed": "PDSensing"]

    var body: some View {
        NavigationStack {
            if isLoggedIn {
                DataCollectionView(isLoggedIn: $isLoggedIn)
                    .environmentObject(userSession)
            } else {
                LoginView(
                    isLoggedIn: $isLoggedIn,
                    showRegister: $showRegister,
                    users: $users
                )
                .environmentObject(userSession)
            }
        }
        .sheet(isPresented: $showRegister) {
            RegisterView(users: $users)
        }
        .onAppear {
            createAdminProfileIfNeeded()
        }
    }
    
    private func createAdminProfileIfNeeded() {
        // Create default profile for admin user if it doesn't exist
        if loadUserProfile(username: "AIMed") == nil {
            let adminProfile = UserProfile(
                username: "AIMed",
                name: "Admin User",
                age: 30,
                weight: 70.0,
                gender: "Unspecified",
                id: generateSimpleID(),
                pdRiskScore: 0
            )
            saveUserProfile(adminProfile)
            print("Created default profile for admin user: AIMed")
        }
    }
}



struct InfoItem: View {
    let label: String
    let value: String
    var color: Color = .primary

    var body: some View {
        HStack(spacing: 4) {
            Text(label + ":")
                .font(.subheadline)
                .foregroundColor(.gray)
            Text(value)
                .font(.subheadline).bold()
                .foregroundColor(color)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(UserSession())
}
