import SwiftUI
import FoundationModels

struct ContentView: View {
    @State private var prompt = ""
    @State private var baseResponse = ""
    @State private var adapterResponse = ""
    @State private var isBaseGenerating = false
    @State private var isAdapterGenerating = false
    @State private var baseError: String?
    @State private var adapterError: String?
    @State private var adapterURL: URL?
    @State private var showFilePicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Adapter file picker
                HStack {
                    if let url = adapterURL {
                        Label(url.lastPathComponent, systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .lineLimit(1)
                    } else {
                        Label("No adapter loaded", systemImage: "xmark.circle")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Select .fmadapter") {
                        showFilePicker = true
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal)

                // Prompt input
                HStack {
                    TextField("พิมพ์ข้อความ...", text: $prompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .submitLabel(.send)
                        .onSubmit(generate)

                    Button {
                        generate()
                    } label: {
                        Image(systemName: "paperplane.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(prompt.isEmpty || isBaseGenerating || isAdapterGenerating)
                }
                .padding(.horizontal)

                // Side-by-side responses
                HStack(spacing: 12) {
                    ResponseBox(
                        title: "Base Model",
                        response: baseResponse,
                        error: baseError,
                        isGenerating: isBaseGenerating
                    )

                    ResponseBox(
                        title: "Thai Adapter",
                        response: adapterResponse,
                        error: adapterError,
                        isGenerating: isAdapterGenerating
                    )
                }
                .padding(.horizontal)
            }
            .navigationTitle("FM Thai Demo")
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [.folder, .package],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    adapterURL = url
                }
            }
        }
    }

    private func generate() {
        let currentPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !currentPrompt.isEmpty else { return }

        baseResponse = ""
        adapterResponse = ""
        baseError = nil
        adapterError = nil

        // Generate both in parallel
        Task { await generateBase(prompt: currentPrompt) }
        Task { await generateAdapter(prompt: currentPrompt) }
    }

    private func generateBase(prompt: String) async {
        isBaseGenerating = true
        defer { isBaseGenerating = false }

        do {
            let session = LanguageModelSession()
            let stream = session.streamResponse(to: prompt)
            for try await partial in stream {
                baseResponse = partial.content
            }
        } catch {
            baseError = error.localizedDescription
        }
    }

    private func generateAdapter(prompt: String) async {
        isAdapterGenerating = true
        defer { isAdapterGenerating = false }

        guard let url = adapterURL else {
            adapterError = "No adapter loaded. Select a .fmadapter file."
            return
        }

        do {
            let adapter = try SystemLanguageModel.Adapter(fileURL: url)
            let model = SystemLanguageModel(adapter: adapter)
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(to: prompt)
            for try await partial in stream {
                adapterResponse = partial.content
            }
        } catch {
            adapterError = error.localizedDescription
        }
    }
}

struct ResponseBox: View {
    let title: String
    let response: String
    let error: String?
    let isGenerating: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            ScrollView {
                if let error {
                    Text(error)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if response.isEmpty && !isGenerating {
                    Text("Response will appear here")
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Text(response)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.fill.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    ContentView()
}
