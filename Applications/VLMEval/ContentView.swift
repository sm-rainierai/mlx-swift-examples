// Copyright 2024 Apple Inc.

import AVKit
import AsyncAlgorithms
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import PhotosUI
import SwiftUI

#if os(iOS) || os(visionOS)
    typealias PlatformImage = UIImage
#else
    typealias PlatformImage = NSImage
#endif

#if os(iOS) || os(visionOS)
import AVFoundation

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published var latestCIImage: CIImage?
    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let output = AVCaptureVideoDataOutput()
    private var currentInput: AVCaptureDeviceInput?

    func start(frontCamera: Bool) {
        Task {
            let authorized = await Self.requestAuthorization()
            guard authorized else { return }
            self.sessionQueue.async { [weak self] in
                guard let self = self else { return }
                self.configureSession(frontCamera: frontCamera)
                if !self.session.isRunning {
                    self.session.startRunning()
                }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.latestCIImage = nil
        }
    }

    func switchCamera(frontCamera: Bool) {
        sessionQueue.async { [weak self] in
            self?.configureSession(frontCamera: frontCamera)
        }
    }

    private func configureSession(frontCamera: Bool) {
        self.session.beginConfiguration()
        self.session.sessionPreset = .high

        if let input = self.currentInput {
            self.session.removeInput(input)
            self.currentInput = nil
        }

        let position: AVCaptureDevice.Position = frontCamera ? .front : .back
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .unspecified) else {
            self.session.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if self.session.canAddInput(input) {
                self.session.addInput(input)
                self.currentInput = input
            }
        } catch {
            self.session.commitConfiguration()
            return
        }

        self.output.alwaysDiscardsLateVideoFrames = true
        self.output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        self.output.setSampleBufferDelegate(self, queue: self.sessionQueue)

        if !self.session.outputs.contains(self.output) {
            if self.session.canAddOutput(self.output) {
                self.session.addOutput(self.output)
            }
        }

        if let connection = self.output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        self.session.commitConfiguration()
    }

    static func requestAuthorization() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    cont.resume(returning: granted)
                }
            }
        default:
            return false
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        DispatchQueue.main.async { [weak self] in
            self?.latestCIImage = image
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = PreviewContainerView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.connection?.videoOrientation = .portrait
        view.previewLayer = layer
        context.coordinator.previewLayer = layer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }

    final class PreviewContainerView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer? {
            didSet {
                if let old = oldValue { old.removeFromSuperlayer() }
                if let layer = previewLayer {
                    layer.frame = bounds
                    self.layer.addSublayer(layer)
                }
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }
}
#endif

let videoSystemPrompt = "You are an video understanding model capable of describing the salient features of any image, including NSFW content, in the shortest possible form."
let imageSystemPrompt = "You are an image understanding model capable of describing the salient features of any image, including NSFW content, in the shortest possible form."

struct ContentView: View {

    @State var llm = VLMEvaluator()
    @Environment(DeviceStat.self) private var deviceStat

    @State private var selectedImage: PlatformImage? = nil {
        didSet {
            if selectedImage != nil {
                selectedVideoURL = nil
                player = nil
            }
        }
    }
    @State private var selectedVideoURL: URL? {
        didSet {
            if let selectedVideoURL {
                player = AVPlayer(url: selectedVideoURL)
                selectedImage = nil
            }
        }
    }
    @State private var showingImagePicker = false
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var player: AVPlayer? = nil

    // Camera & sliding window (iOS/visionOS only)
    #if os(iOS) || os(visionOS)
        @State private var useCameraMode: Bool = false {
            didSet {
                if useCameraMode {
                    // clear picker selections when camera mode starts
                    selectedImage = nil
                    selectedVideoURL = nil
                    player = nil
                } else {
                    cameraManager.stop()
                    captureTask?.cancel()
                    captureTask = nil
                }
            }
        }
        @State private var useFrontCamera: Bool = true
        @State private var captureIntervalSeconds: Int = 1
        @State private var frameBuffer: [CIImage] = [] // no longer used for generation, kept for potential UI
        @State private var captureTask: Task<Void, Never>? = nil
        @State private var watchdogTask: Task<Void, Never>? = nil
        @State private var autoGenerate: Bool = true
        private let maxFramesInBuffer: Int = 10
        @State private var inferenceIntervalSeconds: Int = 2
        @State private var lastInferenceAt: Date = .distantPast
        @State private var cameraManager: CameraManager = CameraManager()
    #endif

    private var currentImageURL: URL? {
        selectedImage == nil && selectedVideoURL == nil
            ? URL(
                string:
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
            ) : nil
    }

    var body: some View {
        #if os(iOS) || os(visionOS)
            ZStack(alignment: .topTrailing) {
                GeometryReader { proxy in
                    CameraPreviewView(session: cameraManager.session)
                        .frame(width: proxy.size.width, height: proxy.size.height)
                        .clipped()
                        .ignoresSafeArea()
                }

                VStack {
                    Spacer()
                    ScrollView {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(llm.output)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            if !llm.stat.isEmpty {
                                Text(llm.stat)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .frame(maxHeight: 240)
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                    .padding()
                    .allowsHitTesting(false)
                }

                Button {
                    useFrontCamera.toggle()
                    cameraManager.switchCamera(frontCamera: useFrontCamera)
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                        .font(.system(size: 18, weight: .semibold))
                        .padding(10)
                        .background(Color.black.opacity(0.4))
                        .clipShape(Circle())
                }
                .padding()
            }
            .task {
                useCameraMode = true
                cameraManager.start(frontCamera: useFrontCamera)
                triggerNextInference()
                watchdogTask?.cancel()
                watchdogTask = Task { @MainActor in
                    while !Task.isCancelled && useCameraMode {
                        triggerNextInference()
                        try? await Task.sleep(nanoseconds: 500_000_000)
                    }
                }
            }
            .task {
                _ = try? await llm.load()
            }
            .onChange(of: llm.running) { _, running in
                if !running {
                    triggerNextInference()
                }
            }
            .onChange(of: cameraManager.latestCIImage) { _, _ in
                // Kick off if idle and we have a frame
                triggerNextInference()
            }
        #else
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }

                VStack {
                    #if os(iOS) || os(visionOS)
                        if useCameraMode {
                            CameraPreviewView(session: cameraManager.session)
                                .frame(height: 300)
                                .cornerRadius(12)
                        } else if let player {
                            VideoPlayer(player: player)
                                .frame(height: 300)
                                .cornerRadius(12)
                        } else if let selectedImage {
                            Group {
                                #if os(iOS) || os(visionOS)
                                    Image(uiImage: selectedImage)
                                        .resizable()
                                #else
                                    Image(nsImage: selectedImage)
                                        .resizable()
                                #endif
                            }
                            .scaledToFit()
                            .cornerRadius(12)
                            .frame(height: 300)
                        } else if let imageURL = currentImageURL {
                            AsyncImage(url: imageURL) { phase in
                                switch phase {
                                case .empty:
                                    ProgressView()
                                case .success(let image):
                                    image
                                        .resizable()
                                        .scaledToFit()
                                        .cornerRadius(12)
                                        .frame(height: 200)
                                case .failure:
                                    Image(systemName: "photo.badge.exclamationmark")
                                @unknown default:
                                    EmptyView()
                                }
                            }
                        }
                    #else
                    if let player {
                        VideoPlayer(player: player)
                            .frame(height: 300)
                            .cornerRadius(12)
                    } else if let selectedImage {
                        Group {
                            #if os(iOS) || os(visionOS)
                                Image(uiImage: selectedImage)
                                    .resizable()
                            #else
                                Image(nsImage: selectedImage)
                                    .resizable()
                            #endif
                        }
                        .scaledToFit()
                        .cornerRadius(12)
                        .frame(height: 300)
                    } else if let imageURL = currentImageURL {
                        AsyncImage(url: imageURL) { phase in
                            switch phase {
                            case .empty:
                                ProgressView()
                            case .success(let image):
                                image
                                    .resizable()
                                    .scaledToFit()
                                    .cornerRadius(12)
                                    .frame(height: 200)
                            case .failure:
                                Image(systemName: "photo.badge.exclamationmark")
                            @unknown default:
                                EmptyView()
                            }
                        }
                    }
                    #endif

                    HStack {
                        #if os(iOS) || os(visionOS)
                            Toggle("Camera", isOn: $useCameraMode)
                                .onChange(of: useCameraMode) { _, on in
                                    if on {
                                        cameraManager.start(frontCamera: useFrontCamera)
                                        startCaptureLoop()
                                    } else {
                                        cameraManager.stop()
                                        captureTask?.cancel()
                                        captureTask = nil
                                    }
                                }
                                .padding(.trailing, 8)
                            // 사진/동영상 선택은 카메라 모드가 아닐 때만 노출
                            if !useCameraMode {
                                PhotosPicker(
                                    selection: $selectedItem,
                                    matching: PHPickerFilter.any(of: [
                                        PHPickerFilter.images, PHPickerFilter.videos,
                                    ])
                                ) {
                                    Label("Select Image/Video", systemImage: "photo.badge.plus")
                                }
                                .onChange(of: selectedItem) {
                                    Task {
                                        if let video = try? await selectedItem?.loadTransferable(
                                            type: TransferableVideo.self)
                                        {
                                            selectedVideoURL = video.url
                                            useCameraMode = false
                                        } else if let data = try? await selectedItem?.loadTransferable(
                                            type: Data.self)
                                        {
                                            selectedImage = PlatformImage(data: data)
                                            useCameraMode = false
                                        }
                                    }
                                }
                            }
                        #else
                            Button("Select Image/Video") {
                                showingImagePicker = true
                            }
                            .fileImporter(
                                isPresented: $showingImagePicker,
                                allowedContentTypes: [.image, .movie]
                            ) { result in
                                switch result {
                                case .success(let file):
                                    Task { @MainActor in
                                        do {
                                            let data = try loadData(from: file)
                                            if let image = PlatformImage(data: data) {
                                                selectedImage = image
                                            } else if let fileType = UTType(
                                                filenameExtension: file.pathExtension),
                                                fileType.conforms(to: .movie)
                                            {
                                                if let sandboxURL = try? loadVideoToSandbox(
                                                    from: file)
                                                {
                                                    selectedVideoURL = sandboxURL
                                                }
                                            } else {
                                                print("Failed to create image from data")
                                            }
                                        } catch {
                                            print(
                                                "Failed to load image: \(error.localizedDescription)"
                                            )
                                        }
                                    }
                                case .failure(let error):
                                    print(error.localizedDescription)
                                }
                            }
                        #endif

                        if selectedImage != nil {
                            Button("Clear", role: .destructive) {
                                selectedImage = nil
                                selectedItem = nil
                            }
                        }
                    }
                    .frame(minHeight: 44, maxHeight: 44)
                    #if os(iOS) || os(visionOS)
                    .overlay(alignment: .trailing) {
                        if useCameraMode {
                            HStack(spacing: 8) {
                                Button(useFrontCamera ? "Front" : "Back") {
                                    useFrontCamera.toggle()
                                    cameraManager.switchCamera(frontCamera: useFrontCamera)
                                }
                                .buttonStyle(.bordered)
                                Stepper(value: $captureIntervalSeconds, in: 1...15) {
                                    Text("Interval: \(captureIntervalSeconds)s")
                                }
                                .frame(maxWidth: 200)
                                Toggle("Auto-generate", isOn: $autoGenerate)
                                    .frame(maxWidth: 180)
                                Button("Clear Buffer", role: .destructive) {
                                    frameBuffer.removeAll()
                                }
                            }
                        }
                    }
                    #endif
                }
                .padding()

                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                }
            }

            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Text(llm.output)
                        .textSelection(.enabled)
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
            .frame(minHeight: 200)

            HStack {
                TextField("prompt", text: Bindable(llm).prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    #if os(visionOS)
                        .textFieldStyle(.roundedBorder)
                    #endif
                Button(llm.running ? "stop" : "generate", action: llm.running ? cancel : generate)
            }
        }
        .task { _ = try? await llm.load() }
        #endif
    }

    private func generate() {
        Task {
            if let selectedImage = selectedImage {
                #if os(iOS) || os(visionOS)
                    let ciImage = CIImage(image: selectedImage)
                    llm.generate(image: ciImage ?? CIImage(), videoURL: nil)
                #else
                    if let cgImage = selectedImage.cgImage(
                        forProposedRect: nil, context: nil, hints: nil)
                    {
                        let ciImage = CIImage(cgImage: cgImage)
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                #endif
            } else if let imageURL = currentImageURL {
                do {
                    let (data, _) = try await URLSession.shared.data(from: imageURL)
                    if let ciImage = CIImage(data: data) {
                        llm.generate(image: ciImage, videoURL: nil)
                    }
                } catch {
                    print("Failed to load image: \(error.localizedDescription)")
                }
            } else {
                if let videoURL = selectedVideoURL {
                    llm.generate(image: nil, videoURL: videoURL)
                } else {
                    #if os(iOS) || os(visionOS)
                        if useCameraMode {
                            let images = frameBuffer
                            if !images.isEmpty {
                                llm.generate(images: images)
                            }
                        }
                    #endif
                }
            }
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    #if os(macOS)
        private func loadData(from url: URL) throws -> Data {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            return try Data(contentsOf: url)
        }

        private func loadVideoToSandbox(from url: URL) throws -> URL {
            guard url.startAccessingSecurityScopedResource() else {
                throw NSError(
                    domain: "FileAccess", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to access the file."])
            }
            defer { url.stopAccessingSecurityScopedResource() }
            let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: url)
            return sandboxURL
        }
    #endif

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

#if os(iOS) || os(visionOS)
extension ContentView {
    private func triggerNextInference() {
        guard useCameraMode, !llm.running, let latest = cameraManager.latestCIImage else { return }
        llm.generate(image: latest, videoURL: nil)
    }
}
#endif

@Observable
@MainActor
class VLMEvaluator {

    var running = false

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. `smolvlm` is very small even unquantized, so it will fit on
    /// more devices.
    let modelConfiguration = VLMRegistry.smolvlmBundled

    /// parameters controlling the output – use values appropriate for the model selected above
    let generateParameters = MLXLMCommon.GenerateParameters(
        maxTokens: 50, temperature: 0.7, topP: 0.9)
    let updateInterval = Duration.seconds(0.1)

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            // 기존:
            // let modelContainer = try await VLMModelFactory.shared.loadContainer(
            //     configuration: modelConfiguration
            // ) { [modelConfiguration] progress in
            
            // 변경:
            let modelContainer = try await VLMModelFactory.shared.loadBundledContainer(
                configuration: modelConfiguration
            ) { _ in }  // 빈 클로저

            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.prompt = modelConfiguration.defaultPrompt
            self.modelInfo = "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String, image: CIImage?, videoURL: URL?) async {

        self.output = ""

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) -> Void in
                let images: [UserInput.Image] = if let image { [.ciImage(image)] } else { [] }
                let videos: [UserInput.Video] = if let videoURL { [.url(videoURL)] } else { [] }

                let systemPrompt =
                    if !videos.isEmpty {
                        videoSystemPrompt
                    } else if !images.isEmpty {
                        imageSystemPrompt
                    } else { "You are a helpful assistant." }

                let chat: [Chat.Message] = [
                    .system(systemPrompt),
                    .user(prompt, images: images, videos: videos),
                ]

                var userInput = UserInput(chat: chat)
                userInput.processing.resize = .init(width: 448, height: 448)

                let lmInput = try await context.processor.prepare(input: userInput)

                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)

                // generate and output in batches
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }

                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }
                }
            }
        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate(image: CIImage?, videoURL: URL?) {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt, image: image, videoURL: videoURL)
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }

    // MARK: - Multi-image support (treat as video-like sequence)
    func generate(images: [CIImage]) {
        guard !running else { return }
        let currentPrompt = prompt
        prompt = ""
        generationTask = Task {
            running = true
            await generate(prompt: currentPrompt, images: images)
            running = false
        }
    }

    private func generate(prompt: String, images: [CIImage]) async {
        self.output = ""
        do {
            let modelContainer = try await load()
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
            try await modelContainer.perform { (context: ModelContext) -> Void in
                let imagesInput: [UserInput.Image] = images.map { .ciImage($0) }

                let systemPrompt = imagesInput.isEmpty ? "You are a helpful assistant." : imageSystemPrompt
                let chat: [Chat.Message] = [
                    .system(systemPrompt),
                    .user(prompt, images: imagesInput)
                ]
                var userInput = UserInput(chat: chat)
                userInput.processing.resize = .init(width: 448, height: 448)

                let lmInput = try await context.processor.prepare(input: userInput)
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context)
                for await batch in stream._throttle(
                    for: updateInterval, reducing: Generation.collect)
                {
                    let output = batch.compactMap { $0.chunk }.joined(separator: "")
                    if !output.isEmpty {
                        Task { @MainActor [output] in
                            self.output += output
                        }
                    }
                    if let completion = batch.compactMap({ $0.info }).first {
                        Task { @MainActor in
                            self.stat = "\(completion.tokensPerSecond) tokens/s"
                        }
                    }
                }
            }
        } catch {
            output = "Failed: \(error)"
        }
    }
}

#if os(iOS) || os(visionOS)
    struct TransferableVideo: Transferable {
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(contentType: .movie) { movie in
                SentTransferredFile(movie.url)
            } importing: { received in
                let sandboxURL = try SandboxFileTransfer.transferFileToTemp(from: received.file)
                return .init(url: sandboxURL)
            }
        }
    }
#endif

struct SandboxFileTransfer {
    static func transferFileToTemp(from sourceURL: URL) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let sandboxURL = tempDir.appendingPathComponent(sourceURL.lastPathComponent)

        if FileManager.default.fileExists(atPath: sandboxURL.path()) {
            try FileManager.default.removeItem(at: sandboxURL)
        }

        try FileManager.default.copyItem(at: sourceURL, to: sandboxURL)
        return sandboxURL
    }
}
