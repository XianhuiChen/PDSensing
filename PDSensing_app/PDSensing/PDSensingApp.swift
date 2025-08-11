//
//  PDSensingApp.swift
//  PDSensing
//
//  Created by Betty Chen on 4/25/25.
//

import SwiftUI


@main
struct PDSensingApp: App {
    @StateObject private var userSession = UserSession()
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)                .environmentObject(userSession)
        }
    }
}



