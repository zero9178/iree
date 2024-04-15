
#include "iree/compiler/PluginAPI/Client.h"

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"

#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "compiler/plugins/target/LLVMCPU/StaticLibraryGenerator.h"

#include "compiler/plugins/target/LLVMCPU/Builtins/Device.h"
#include "compiler/src/iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

class QuidditchTargetDevice final : public IREE::HAL::TargetDevice {
public:
  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("quidditch")
        ->getDefaultExecutableTargets(context, "quidditch", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("quidditch"),
                                            configAttr, executableTargetAttrs);
  }
};

class QuidditchTargetBackend final : public IREE::HAL::TargetBackend {
public:
  [[nodiscard]] std::string getLegacyDefaultDeviceID() const override {
    return "quidditch";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(IREE::HAL::ExecutableTargetAttr::get(
        context, "quidditch", "device_target"));
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    if (variantOp.isExternal())
      return;

    addCommonTargetExecutablePreprocessingPasses(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    if (variantOp.isExternal())
      return;

    // Go from linalg on tensor to linalg on buffer.
    passManager.addPass(createTileAndDistributeToWorkgroupsPass());
    auto &nestedModulePM = passManager.nest<ModuleOp>();
    nestedModulePM.addNestedPass<func::FuncOp>(
        createConvertToDestinationPassingStylePass());
    nestedModulePM.addNestedPass<func::FuncOp>(
        createFoldAffineMinInDistributedLoopsPass());
    nestedModulePM.addPass(createCanonicalizerPass());
    nestedModulePM.addPass(createCSEPass());
    nestedModulePM.addNestedPass<func::FuncOp>(
        createFuseTensorPadWithConsumerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(
        createConcretizePadResultShapePass());
    nestedModulePM.addNestedPass<func::FuncOp>(
        IREE::LinalgExt::createTileAndDecomposeWinogradTransformPass());
    addIREEComprehensiveBufferizePasses(nestedModulePM);
    nestedModulePM.addPass(createEraseHALDescriptorTypeFromMemRefPass());

    {
      auto &modulePm = nestedModulePM;
      modulePm.addNestedPass<func::FuncOp>(
          IREE::LinalgExt::createLinalgExtToLoopsPass());
      modulePm.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
      modulePm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      modulePm.addPass(createConvertBf16ArithToF32Pass());
      modulePm.addPass(createConvertBf16ToUInt16BuffersPass());
      modulePm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      modulePm.addNestedPass<func::FuncOp>(createCSEPass());

      // Handled tensor-type constants.
      modulePm.addPass(arith::createConstantBufferizePass());
      modulePm.addPass(createFoldTensorExtractOpPass());

      // Handle complex operation conversion.
      modulePm.addPass(createConvertComplexToStandardPass());

      // math dialect elementry functions -> polynomial form.
      modulePm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

      modulePm.addNestedPass<func::FuncOp>(
          createHoistStaticallyBoundAllocationsPass());

      modulePm.addNestedPass<func::FuncOp>(
          createIREEExpandStridedMetadataPass());
      modulePm.addNestedPass<func::FuncOp>(createCleanupBufferAllocViewPass());

      // Checking stack allocation before converting to CF dialect is easier.
      modulePm.addPass(createLLVMCPUCheckIRBeforeLLVMConversionPass());

      // SCF -> CF
      modulePm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
      modulePm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      modulePm.addNestedPass<func::FuncOp>(createCSEPass());

      // (HAL, IREE, Linalg, CF) -> LLVM
      modulePm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
      modulePm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
      modulePm.addPass(memref::createFoldMemRefAliasOpsPass());
      modulePm.addPass(createEmulateNarrowTypePass());
      modulePm.addPass(createCanonicalizerPass());
      modulePm.addPass(createCSEPass());

      modulePm.addPass(createConvertToLLVMPass());
      modulePm.addPass(createReconcileUnrealizedCastsPass());

      // We rely on MLIR symbol visibility being correct after this point and
      // need to mirror the LLVM linkage that was assigned during conversion.
      modulePm.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

      modulePm.addPass(createCanonicalizerPass());
      modulePm.addPass(createCSEPass());
      modulePm.addNestedPass<LLVM::LLVMFuncOp>(createAddFastMathFlagsPass());
    }
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // TODO: This is a 'builtin.module' containing a 'func.func' containing the
    //       kernel which consists of memref, linalg, scf and some 'hal' ops for
    //       I/O. It should be lowered to an object file that is later linked
    //       into the final executable.
    [[maybe_unused]] ModuleOp module = variantOp.getInnerModule();
    module.dump();

    // TODO: Be inspired by or use the static library export in the LLVMCPU
    //       target: It just exports the library name for the purpose of the
    //       runtime and generates an entry point for the runtime.
    //
    //       See runtime/src/iree/hal/local/loaders/static_library_loader.c
    //       and compiler/plugins/target/LLVMCPU/LLVMCPUTarget.cpp:360
    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    llvm::LLVMContext context;

    std::string errorMessage;
    auto llvmTarget = llvm::TargetRegistry::lookupTarget(
        "riscv32-unknown-unknown-elf", errorMessage);
    if (!llvmTarget)
      return failure();

    std::unique_ptr<llvm::TargetMachine> targetMachine(
        llvmTarget->createTargetMachine(
            "riscv32-unknown-unknown-elf", "generic-rv32",
            /*cpu features=*/"", llvm::TargetOptions(),
            llvm::Reloc::Model::PIC_, llvm::CodeModel::Medium,
            llvm::CodeGenOptLevel::Aggressive));

    const llvm::Triple &targetTriple = targetMachine->getTargetTriple();
    variantOp.getInnerModule()->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        executableBuilder.getStringAttr(targetTriple.str()));

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(variantOp.getInnerModule(),
                                                    context, libraryName);
    if (!llvmModule) {
      return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                      "dialect to the native llvm::Module";
    }

    // Configure the functions in the module. This may override defaults set
    // during the MLIR->LLVM conversion.
    for (auto &func : *llvmModule) {
      // Enable frame pointers to ensure that stack unwinding works, e.g. in
      // Tracy. In principle this could also be achieved by enabling unwind
      // tables, but we tried that and that didn't work in Tracy (which uses
      // libbacktrace), while enabling frame pointers worked.
      // https://github.com/openxla/iree/issues/3957
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");

      // Our dispatches are all hot - that's kind of the point.
      // This may favor more aggressive optimizations.
      func.addFnAttr("hot");
    }

    // Build the IREE HAL executable library metadata. The runtime uses this to
    // find the entry point functions and their information.
    IREE::HAL::LibraryBuilder::Mode libraryBuilderMode =
        IREE::HAL::LibraryBuilder::Mode::NONE;
    IREE::HAL::LibraryBuilder libraryBuilder(
        llvmModule.get(), libraryBuilderMode,
        IREE::HAL::LibraryBuilder::Version::LATEST);

    // Declare dynamically imported functions.
    auto importsAttrName =
        StringAttr::get(variantOp.getContext(), "hal.executable.imports");
    if (auto importsAttr =
            variantOp->getAttrOfType<ArrayAttr>(importsAttrName)) {
      for (auto importAttr : importsAttr.getAsValueRange<ArrayAttr>()) {
        auto nameAttr = llvm::cast<StringAttr>(importAttr[0]);
        auto weakAttr = llvm::cast<BoolAttr>(importAttr[1]);
        libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
      }
      variantOp->removeAttr(importsAttrName);
    }

    // Declare exported entry points.
    auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
    for (auto exportOp :
         variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
      if (!llvmFunc)
        continue;
      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);

      // Tag the function parameters in case they got removed during conversion.
      // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
      for (unsigned i = 0; i <= 2; ++i) {
        llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
        llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
        llvmFunc->addParamAttr(i, align16);
      }

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      int64_t localMemorySize = exportOp.getWorkgroupLocalMemory()
                                    .value_or(APInt(64, 0))
                                    .getSExtValue();

      IREE::HAL::LibraryBuilder::SourceLocation sourceLocation;
      SmallVector<IREE::HAL::LibraryBuilder::SourceLocation> stageLocations;
      libraryBuilder.addExport(
          exportOp.getName(), std::move(sourceLocation),
          std::move(stageLocations), /*tag=*/"",
          IREE::HAL::LibraryBuilder::DispatchAttrs{localMemorySize}, llvmFunc);
    }

    // Embed source files (if present).
    if (auto sourcesAttr = variantOp.getSourcesAttr()) {
      for (auto sourceAttr : sourcesAttr.getValue()) {
        if (auto resourceAttr = dyn_cast_if_present<DenseResourceElementsAttr>(
                sourceAttr.getValue())) {
          auto handle = resourceAttr.getRawHandle();
          SmallVector<char> rawData;
          llvm::append_range(rawData, handle.getBlob()->getData());
          libraryBuilder.addSourceFile(sourceAttr.getName(),
                                       std::move(rawData));
        }
      }
    }

    auto queryFunctionName = std::string("iree_hal_executable_library_query");

    // Static library query functions must be unique to support multiple
    // libraries in the same namespace.
    queryFunctionName = libraryName + "_library_query";

    auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setDSOLocal(false);
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    queryLibraryFunc->setDLLStorageClass(
        llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

    // Specialize the module to our target machine.
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());

    // Statically link libraries into our module prior to LLVM optimizations.
    // This approximates LTO.
    llvm::Linker moduleLinker(*llvmModule);

    // Link any bitcode files specified on the command line.
    if (failed(IREE::HAL::linkCmdlineBitcodeFiles(
            variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
            *targetMachine, context))) {
      return failure();
    }

    // Link any bitcode objects specified in executable.object attributes and
    // specialize them for the current config.
    if (failed(IREE::HAL::linkBitcodeObjects(
            variantOp.getLoc(), moduleLinker, llvm::Linker::LinkOnlyNeeded,
            *targetMachine, variantOp.getObjectsAttr(), context))) {
      return failure();
    }

    // Link our libdevice after all codegen and user objects as they may
    // reference it. Some of the functions in here are only known used after
    // we perform LLVM ISel and need to be pulled in whether they are used or
    // not.
    if (failed(IREE::HAL::linkBitcodeModule(
            variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
            *targetMachine, "libdevice",
            IREE::HAL::loadDeviceBitcode(targetMachine.get(), context),
            [&](llvm::Module &module) {
              IREE::HAL::specializeDeviceModule(variantOp, module,
                                                *targetMachine);
            }))) {
      return mlir::emitError(variantOp.getLoc())
             << "failed linking in builtin library for target triple '"
             << targetTriple.str() << "'";
    }

    // Strip any compiler identifiers that may have snuck in. We let the linker
    // tag the module.
    auto *llvmIdent = llvmModule->getNamedMetadata("llvm.ident");
    if (llvmIdent)
      llvmIdent->clearOperands();

    /*
    // LLVM opt passes that perform code generation optimizations/transformation
    // similar to what a frontend would do.
    if (failed(
            runLLVMIRPasses(target, targetMachine.get(), llvmModule.get()))) {
      return variantOp.emitError()
             << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
                "targeting '"
             << targetTriple.str() << "'";
    }
     */

    // Fixup visibility from any symbols we may link in - we want to hide all
    // but the query entry point.
    // Note: can't move this before runLLVMIRPasses at the moment, as further
    // symbol references may still be created past this point, namely to math
    // functions, e.g. `llvm.frem` lowering to a call to `fmodf`.
    SetVector<llvm::Function *> preservedFuncs;
    preservedFuncs.insert(queryLibraryFunc);
    // fixupVisibility(*llvmModule, preservedFuncs);

    SmallVector<IREE::HAL::Artifact> objectFiles;

    // Emit the base object file containing the bulk of our code.
    // This must come first such that we have the proper library linking order.
    {
      // NOTE: today we just use a single object file, however if we wanted to
      // scale code generation and linking we'd want to generate one per
      // function (or something like that). A single object file is also
      // instrumental to static library generation (which only supports one
      // object file per library).
      std::string objectData;
      if (failed(IREE::HAL::runEmitObjFilePasses(
              targetMachine.get(), llvmModule.get(),
              llvm::CodeGenFileType::ObjectFile, &objectData))) {
        return variantOp.emitError()
               << "failed to compile LLVM-IR module to an object file";
      }
      auto objectFile = IREE::HAL::Artifact::createTemporary(libraryName, "o");
      auto &os = objectFile.outputFile->os();
      os << objectData;
      os.flush();
      os.close();
      objectFiles.push_back(std::move(objectFile));
    }

    // If custom object files were specified then add those to our artifact set.
    // These will either be combined into the resulting static library or linked
    // statically into the resulting dynamic library.
    SmallVector<IREE::HAL::ExecutableObjectAttr> linkerObjectAttrs;
    IREE::HAL::ExecutableObjectAttr::filterObjects(variantOp.getObjectsAttr(),
                                                   {".o", ".obj", ".a", ".lib"},
                                                   linkerObjectAttrs);
    for (auto [index, attr] : llvm::enumerate(linkerObjectAttrs)) {
      auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(attr);
      if (auto dataAttr = objectAttr.getData()) {
        objectFiles.push_back(IREE::HAL::Artifact::createTemporary(
            objectFiles.front().path + "_object_" + std::to_string(index),
            llvm::sys::path::extension(objectAttr.getPath())));
      } else {
        auto absolutePath = objectAttr.getAbsolutePath();
        if (failed(absolutePath)) {
          llvm::errs()
              << "ERROR: referenced object file not found on any path; use "
                 "--iree-hal-executable-object-search-path= to add search "
                 "paths: "
              << objectAttr << "\n";
          return failure();
        }
        objectFiles.push_back(IREE::HAL::Artifact::fromFile(*absolutePath));
      }
    }

    return serializeStaticLibraryExecutable(options, variantOp,
                                            executableBuilder, libraryName,
                                            queryFunctionName, objectFiles);
  }

  LogicalResult serializeStaticLibraryExecutable(
      const SerializationOptions &options,
      IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
      const std::string &libraryName, const std::string &queryFunctionName,
      const SmallVector<IREE::HAL::Artifact> &objectFiles) {
    if (objectFiles.size() != 1) {
      // Static library output only supports single object libraries.
      return variantOp.emitError() << "generating static libraries from "
                                      "multiple object files is not supported";
    }

    // Copy the static object file to the specified output along with
    // generated header file.
    if (!IREE::HAL::outputStaticLibrary(libraryName, queryFunctionName,
                                        "quidditch.a", objectFiles[0].path)) {
      return variantOp.emitError() << "static library generation failed";
    }

    // Embed the library name in the executable binary op. This informs the
    // loader which static library to load for the target binary.
    std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                           libraryName.end());
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(), "static",
        libraryNameVector);

    return success();
  }
};

class QuidditchSession final
    : public PluginSession<QuidditchSession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("quidditch",
                []() { return std::make_shared<QuidditchTargetDevice>(); });
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
    targets.add("quidditch",
                []() { return std::make_shared<QuidditchTargetBackend>(); });
  }
};
} // namespace

extern "C" bool iree_register_compiler_plugin_hal_target_quidditch(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<QuidditchSession>("hal_target_quidditch");
  return true;
}
