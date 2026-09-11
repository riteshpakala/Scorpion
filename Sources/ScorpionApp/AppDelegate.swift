//
//  AppDelegate.swift
//  ScorpionApp
//

import AppKit
import ScorpionFlux2
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    let model = ScanViewModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Composition root: executor families available to this app.
        Flux2Registration.register()
        NSApp.mainMenu = MainMenu.build(target: self)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1080, height: 760),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = "Scorpion"
        window.minSize = NSSize(width: 820, height: 560)
        window.contentView = NSHostingView(rootView: ScanView(model: model))
        window.center()
        window.setFrameAutosaveName("ScorpionMainWindow")
        window.makeKeyAndOrderFront(nil)
        self.window = window
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    @objc func openImage(_ sender: Any?) { model.chooseImage() }
    @objc func exportReport(_ sender: Any?) { model.exportReport() }
    @objc func startScan(_ sender: Any?) { model.run() }
}
