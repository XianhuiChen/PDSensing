//
//  Login.swift
//  PDSensing
//
//  Created by Betty Chen on 5/5/25.
//

import SwiftUI
import Combine


class UserSession: ObservableObject {
    @Published var isLoggedIn: Bool = false
    @Published var username: String = ""
    @Published var embedding: [Float] = []
}

struct UserDatabase: Codable {
    var users: [String: String]
}


func loadUsersFromFile() -> [String: String] {
    let decoder = JSONDecoder()
    do {
        let documentsURL = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let fileURL = documentsURL.appendingPathComponent("users.json")
        let data = try Data(contentsOf: fileURL)
        let userDB = try decoder.decode(UserDatabase.self, from: data)
        return userDB.users
    } catch {
        print("Failed to load users: \(error)")
        return [:]
    }
}



func saveUsersToFile(users: [String: String]) {
    let userDB = UserDatabase(users: users)
    let encoder = JSONEncoder()
    do {
        let data = try encoder.encode(userDB)
        let documentsURL = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let fileURL = documentsURL.appendingPathComponent("users.json")
        try data.write(to: fileURL)
        print("Saved users to \(fileURL)")
    } catch {
        print("Failed to save users: \(error)")
    }
}
struct LoginView: View {
    @EnvironmentObject var userSession: UserSession
    
    @Binding var isLoggedIn: Bool
    @Binding var showRegister: Bool
    @Binding var users: [String: String]
    
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var loginFailed: Bool = false
    @State private var showAccountManager: Bool = false
    var body: some View {
        VStack(spacing: 20) {
            Text("PDSensing")
                .font(.largeTitle).bold()

            TextField("Username", text: $username)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)

            SecureField("Password", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())

            if loginFailed {
                Text("Invalid credentials").foregroundColor(.red).font(.caption)
            }

            Button(action: {
                if users[username] == password {
                    isLoggedIn = true
                    loginFailed = false
                    userSession.isLoggedIn = true
                    userSession.username = username
                    
                }
                else {
                    loginFailed = true
                }
            }) {
                Text("Login")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }

            Button("Don't have an account? Register here") {
                showRegister = true
            }
            .font(.footnote)

            Button("Manage Accounts") {
                showAccountManager = true
            }
            .font(.footnote)
            .foregroundColor(.orange)
            .disabled(!(username == "PDSensing" && password == "AIMed"))
        }
        .padding()
        .sheet(isPresented: $showAccountManager) {
            AccountManagerView(users: $users)
        }
    }
}

struct RegisterView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var users: [String: String]

    @State private var username: String = ""
    @State private var name: String = ""
    @State private var age: String = ""
    @State private var weight: String = ""
    @State private var gender: String = "Unspecified"
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
    @State private var error: String? = nil
    
    private let genderOptions = ["Male", "Female", "Other", "Unspecified"]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    Text("Create New Account")
                        .font(.title2).bold()

                    VStack(spacing: 16) {
                        TextField("Username", text: $username)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .autocapitalization(.none)
                        
                        TextField("Full Name", text: $name)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        
                        TextField("Age", text: $age)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.numberPad)
                        
                        TextField("Weight (kg)", text: $weight)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.decimalPad)
                        
                        Picker("Gender", selection: $gender) {
                            ForEach(genderOptions, id: \.self) { option in
                                Text(option).tag(option)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity, alignment: .leading)

                        SecureField("Password", text: $password)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .textContentType(.none)
                            .disableAutocorrection(true)

                        SecureField("Confirm Password", text: $confirmPassword)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                    }

                    if let error = error {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                    }

                    Button("Register") {
                        validateAndRegister()
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)

                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Register")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func validateAndRegister() {
        // Validation
        if username.isEmpty || name.isEmpty || password.isEmpty {
            error = "Username, name and password cannot be empty"
            return
        }
        
        if password != confirmPassword {
            error = "Passwords do not match"
            return
        }
        
        if users.keys.contains(username) {
            error = "Username already exists"
            return
        }
        
        guard let ageInt = Int(age), ageInt > 0 else {
            error = "Please enter a valid age"
            return
        }
        
        guard let weightDouble = Double(weight), weightDouble > 0 else {
            error = "Please enter a valid weight"
            return
        }
        
        // Create user account
        users[username] = password
        saveUsersToFile(users: users)
        
        // Create user profile
        let newProfile = UserProfile(
            username: username,
            name: name,
            age: ageInt,
            weight: weightDouble,
            gender: gender,
            id: generateSimpleID(),
            pdRiskScore: 0
        )
        saveUserProfile(newProfile)
        
        // Create user data folder
        createUserDataFolder(username: username)
        
        dismiss()
    }
    
    private func createUserDataFolder(username: String) {
        do {
            let documentsURL = try FileManager.default.url(
                for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
            )
            let userFolderURL = documentsURL.appendingPathComponent("data").appendingPathComponent(username)
            
            if !FileManager.default.fileExists(atPath: userFolderURL.path) {
                try FileManager.default.createDirectory(at: userFolderURL, withIntermediateDirectories: true, attributes: nil)
                print("Created user data folder: \(userFolderURL.path)")
                
                // Create default data files by copying from bundle if they exist
                createDefaultDataFiles(userFolderURL: userFolderURL)
            } else {
                print("User data folder already exists: \(userFolderURL.path)")
            }
        } catch {
            print("Failed to create user data folder: \(error)")
        }
    }
    
    private func createDefaultDataFiles(userFolderURL: URL) {
        guard let bundleURL = Bundle.main.resourceURL else { return }
        
        let dataFiles = ["walking.json", "tapping.json", "tapping_accel.json", "voice.json"]
        
        // Extract username from the destination folder path
        let username = userFolderURL.lastPathComponent
        
        for fileName in dataFiles {
            // Try to load from user-specific folder first (data/{username})
            let userSourceURL = bundleURL.appendingPathComponent("data").appendingPathComponent(username).appendingPathComponent(fileName)
            
            // Fallback to Samuel_Hale folder if user-specific folder doesn't exist
            let fallbackSourceURL = bundleURL.appendingPathComponent("data").appendingPathComponent("Samuel_Hale").appendingPathComponent(fileName)
            
            let destinationURL = userFolderURL.appendingPathComponent(fileName)
            
            do {
                if FileManager.default.fileExists(atPath: userSourceURL.path) {
                    try FileManager.default.copyItem(at: userSourceURL, to: destinationURL)
                    print("Copied \(fileName) from bundle data/\(username)")
                } else if FileManager.default.fileExists(atPath: fallbackSourceURL.path) {
                    try FileManager.default.copyItem(at: fallbackSourceURL, to: destinationURL)
                    print("Copied \(fileName) from bundle data/Samuel_Hale (fallback)")
                } else {
                    print("Warning: Could not find source file for \(fileName) for user \(username)")
                }
            } catch {
                print("Failed to copy \(fileName): \(error)")
            }
        }
    }
}

struct AccountManagerView: View {
    @Binding var users: [String: String]
    
    var body: some View {
        NavigationView {
            List {
                ForEach(users.keys.sorted(), id: \.self) { username in
                    HStack {
                        Text(username)
                        Spacer()
                        Button(role: .destructive) {
                            users.removeValue(forKey: username)
                            saveUsersToFile(users: users)
                            deleteUserProfile(username: username)
                        } label: {
                            Image(systemName: "trash")
                        }
                    }
                }
            }
            .navigationTitle("Manage Accounts")
        }
    }
}
func deleteUserProfile(username: String) {
    do {
        let documentsURL = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let fileURL = documentsURL.appendingPathComponent("user_\(username)_profile.json")
        try FileManager.default.removeItem(at: fileURL)
        print("Deleted profile for \(username)")
    } catch {
        print("Failed to delete profile: \(error)")
    }
}

#Preview("Login View") {
    LoginView(
        isLoggedIn: .constant(false),
        showRegister: .constant(false),
        users: .constant(["testuser": "password123", "admin": "admin"])
    )
    .environmentObject(UserSession())
}

#Preview("Register View") {
    RegisterView(users: .constant(["existing_user": "password"]))
}

#Preview("Account Manager View") {
    AccountManagerView(users: .constant([
        "user1": "password1",
        "user2": "password2",
        "admin": "admin123"
    ]))
}
