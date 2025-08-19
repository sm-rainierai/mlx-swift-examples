// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

public enum VLMError: LocalizedError {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case singleVideoAllowed
    case singleMediaTypeAllowed
    case imageProcessingFailure(String)
    case processing(String)

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return String(localized: "An image is required for this operation.")
        case .maskRequired:
            return String(localized: "An image mask is required for this operation.")
        case .singleImageAllowed:
            return String(localized: "Only a single image is allowed for this operation.")
        case .singleVideoAllowed:
            return String(localized: "Only a single video is allowed for this operation.")
        case .singleMediaTypeAllowed:
            return String(
                localized:
                    "Only a single media type (image or video) is allowed for this operation.")
        case .imageProcessingFailure(let details):
            return String(localized: "Failed to process the image: \(details)")
        case .processing(let details):
            return String(localized: "Processing error: \(details)")
        }
    }
}

public struct BaseProcessorConfiguration: Codable, Sendable {
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
    { url in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return modelInit(configuration)
    }
}

private func create<C: Codable, P>(
    _ configurationType: C.Type,
    _ processorInit: @escaping (
        C,
        any Tokenizer
    ) -> P
) -> (URL, any Tokenizer) throws -> P {
    { url, tokenizer in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return processorInit(configuration, tokenizer)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
public class VLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {

    /// Shared instance with default model types.
    public static let shared: VLMTypeRegistry = .init(creators: all())

    /// All predefined model types
    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
            "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
            "qwen2_5_vl": create(Qwen25VLConfiguration.self, Qwen25VL.init),
            "idefics3": create(Idefics3Configuration.self, Idefics3.init),
            "gemma3": create(Gemma3Configuration.self, Gemma3.init),
            "smolvlm": create(SmolVLM2Configuration.self, SmolVLM2.init),
        ]
    }
}

public class VLMProcessorTypeRegistry: ProcessorTypeRegistry, @unchecked Sendable {

    /// Shared instance with default processor types.
    public static let shared: VLMProcessorTypeRegistry = .init(creators: all())

    /// All predefined processor types.
    private static func all() -> [String: @Sendable (URL, any Tokenizer) throws ->
        any UserInputProcessor]
    {
        [
            "PaliGemmaProcessor": create(
                PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
            "Qwen2VLProcessor": create(
                Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
            "Qwen2_5_VLProcessor": create(
                Qwen25VLProcessorConfiguration.self, Qwen25VLProcessor.init),
            "Idefics3Processor": create(
                Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
            "Gemma3Processor": create(
                Gemma3ProcessorConfiguration.self, Gemma3Processor.init),
            "SmolVLMProcessor": create(
                SmolVLMProcessorConfiguration.self, SmolVLMProcessor.init),
        ]
    }
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class VLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared: VLMRegistry = .init(modelConfigurations: all())

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2_5VL3BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let smolvlminstruct4bit = ModelConfiguration(
        id: "mlx-community/SmolVLM-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let gemma3_4B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-4b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_12B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-12b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_27B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-27b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let smolvlm = ModelConfiguration(
        id: "EZCon/SmolVLM2-500M-Video-Instruct-4bit-mlx",
        defaultPrompt:
            "What is the main action or notable event happening in this segment? Describe it in one brief sentence."
    )

    static public func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
            qwen2_5VL3BInstruct4Bit,
            smolvlminstruct4bit,
            gemma3_4B_qat_4bit,
            gemma3_12B_qat_4bit,
            gemma3_27B_qat_4bit,
            smolvlm,
        ]
    }

}

@available(*, deprecated, renamed: "VLMRegistry", message: "Please use VLMRegistry directly.")
public typealias ModelRegistry = VLMRegistry

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await VLMModelFactory.shared.loadContainer(
///     configuration: VLMRegistry.paligemma3bMix4488bit)
/// ```
public class VLMModelFactory: ModelFactory {

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = VLMModelFactory(
        typeRegistry: VLMTypeRegistry.shared, processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry: ProcessorTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

        // load the generic config to understand which model and how to load the weights
        let configurationURL = modelDirectory.appending(
            component: "config.json"
        )

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder().decode(
                BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try typeRegistry.createModel(
                configuration: configurationURL, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let tokenizer = try await loadTokenizer(
            configuration: configuration,
            hub: hub
        )

        let processorConfigurationURL = modelDirectory.appending(
            component: "preprocessor_config.json"
        )

        let baseProcessorConfig: BaseProcessorConfiguration
        do {
            baseProcessorConfig = try JSONDecoder().decode(
                BaseProcessorConfiguration.self,
                from: Data(contentsOf: processorConfigurationURL)
            )
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                processorConfigurationURL.lastPathComponent, configuration.name, error)
        }

        let processor = try processorRegistry.createModel(
            configuration: processorConfigurationURL,
            processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer)

        return .init(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)
    }

}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
        VLMModelFactory.shared
    }
}

// VLMModelFactory extension for bundled model loading (수정된 버전)
extension VLMModelFactory {
    
    /// 번들된 모델을 로드하는 메서드 (개별 파일들이 번들 루트에 있는 경우)
    public func loadBundledContainer(
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        
        print("=== 번들 모델 로딩 시작 ===")
        
        // 번들 루트를 모델 디렉토리로 사용
        guard let bundleURL = Bundle.main.resourceURL else {
            throw ModelFactoryError.unsupportedModelType("Bundle resource URL not found")
        }
        
        let modelDirectory = bundleURL
        print("모델 디렉토리: \(modelDirectory.path)")
        
        // 필수 파일들 존재 확인
        let requiredFiles = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors"
        ]
        
        print("=== 필수 파일 확인 ===")
        for fileName in requiredFiles {
            let fileURL = modelDirectory.appendingPathComponent(fileName)
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("\(exists ? "✅" : "❌") \(fileName)")
            
            if !exists {
                throw ModelFactoryError.unsupportedModelType("\(fileName) not found in bundle")
            }
        }
        
        // config.json 로드
        let configurationURL = modelDirectory.appendingPathComponent("config.json")
        let baseConfig: BaseConfiguration
        do {
            print("config.json 로딩...")
            baseConfig = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: configurationURL)
            )
            print("✅ config.json 로딩 완료 - 모델 타입: \(baseConfig.modelType)")
        } catch let error as DecodingError {
            print("❌ config.json 디코딩 실패: \(error)")
            throw ModelFactoryError.configurationDecodingError("config.json", configuration.name, error)
        }
        
        // 모델 생성
        let model: LanguageModel
        do {
            print("모델 인스턴스 생성...")
            model = try typeRegistry.createModel(
                configuration: configurationURL,
                modelType: baseConfig.modelType
            )
            print("✅ 모델 인스턴스 생성 완료")
        } catch let error as DecodingError {
            print("❌ 모델 생성 실패: \(error)")
            throw ModelFactoryError.configurationDecodingError("config.json", configuration.name, error)
        }
        
        // weights 로드
        do {
            print("모델 가중치 로딩...")
            try loadWeights(
                modelDirectory: modelDirectory,
                model: model,
                perLayerQuantization: baseConfig.perLayerQuantization
            )
            print("✅ 모델 가중치 로딩 완료")
        } catch {
            print("❌ 가중치 로딩 실패: \(error)")
            throw error
        }
        
        // tokenizer 로드
        let tokenizer: any Tokenizer
        do {
            print("토크나이저 로딩...")
            tokenizer = try loadBundledTokenizer(from: modelDirectory)
            print("✅ 토크나이저 로딩 완료")
        } catch {
            print("❌ 토크나이저 로딩 실패: \(error)")
            throw error
        }
        
        // processor 로드
        let processorConfigurationURL = modelDirectory.appendingPathComponent("preprocessor_config.json")
        guard FileManager.default.fileExists(atPath: processorConfigurationURL.path) else {
            throw ModelFactoryError.unsupportedProcessorType("preprocessor_config.json not found in bundle")
        }
        
        let baseProcessorConfig: BaseProcessorConfiguration
        do {
            print("프로세서 설정 로딩...")
            baseProcessorConfig = try JSONDecoder().decode(
                BaseProcessorConfiguration.self,
                from: Data(contentsOf: processorConfigurationURL)
            )
            print("✅ 프로세서 설정 로딩 완료 - 타입: \(baseProcessorConfig.processorClass)")
        } catch let error as DecodingError {
            print("❌ 프로세서 설정 디코딩 실패: \(error)")
            throw ModelFactoryError.configurationDecodingError("preprocessor_config.json", configuration.name, error)
        }
        
        let processor: any UserInputProcessor
        do {
            print("프로세서 인스턴스 생성...")
            processor = try processorRegistry.createModel(
                configuration: processorConfigurationURL,
                processorType: baseProcessorConfig.processorClass,
                tokenizer: tokenizer
            )
            print("✅ 프로세서 인스턴스 생성 완료")
        } catch {
            print("❌ 프로세서 생성 실패: \(error)")
            throw error
        }
        
        // Progress 완료 표시
        let progress = Progress(totalUnitCount: 1)
        progress.completedUnitCount = 1
        progressHandler(progress)
        
        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
        
        print("✅ 모델 컨테이너 생성 완료")
        return ModelContainer(context: context)
    }
    
    /// 번들에서 토크나이저 로드
    private func loadBundledTokenizer(from modelDirectory: URL) throws -> any Tokenizer {
        let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = modelDirectory.appendingPathComponent("tokenizer.json")
        
        // tokenizer_config.json 로드
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw ModelFactoryError.unsupportedModelType("tokenizer_config.json not found in bundle")
        }
        
        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try JSONDecoder().decode(Config.self, from: tokenizerConfigData)
        
        // tokenizer.json 로드
        guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
            throw ModelFactoryError.unsupportedModelType("tokenizer.json not found in bundle")
        }
        
        let tokenizerDataData = try Data(contentsOf: tokenizerDataURL)
        let tokenizerData = try JSONDecoder().decode(Config.self, from: tokenizerDataData)
        
        // AutoTokenizer를 사용하여 적절한 토크나이저 생성
        return try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )
    }
}
