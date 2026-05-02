/*
 ██████╗ ██████╗ ████████╗██╗██████╗ ██╗███╗   ███╗
██╔═══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║████╗ ████║
██║   ██║██████╔╝   ██║   ██║██████╔╝██║██╔████╔██║
██║   ██║██╔═══╝    ██║   ██║██╔═══╝ ██║██║╚██╔╝██║
╚██████╔╝██║        ██║   ██║██║     ██║██║ ╚═╝ ██║
 ╚═════╝ ╚═╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝                                                
*/

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "pimopt/Analysis/DataLayout/DataLayoutPass.h"

using namespace mlir;

static llvm::cl::opt<bool> useDataLayoutPass("data-layout-pass",
                       llvm::cl::desc("Use Data Layout Pass"),
                       llvm::cl::init(false));

static llvm::cl::opt<std::string> traceOutputPath("trace-output-path",
                                             llvm::cl::desc("Specify the output path for the generated simulation trace file"),
                                             llvm::cl::value_desc("path"),
                                             llvm::cl::init("./Result/Layer_group_0_results.txt"));

static llvm::cl::opt<std::string> modelDebugPath("model-debug-file",
                                             llvm::cl::desc("Specify the output path to store the milp Model"),
                                             llvm::cl::value_desc("path-milp"),
                                             llvm::cl::init(""));

static llvm::cl::opt<int>
    deviceType("target-device-type",
               llvm::cl::desc("Specify the target device type"),
               llvm::cl::value_desc("devicetype"), llvm::cl::init(0));

static llvm::cl::opt<int> memAlloScheme("memory-allocation-scheme",
                                    llvm::cl::desc("Specify the memory allocation scheme for MILP optimization"),
                                    llvm::cl::value_desc("memory-alloc"),
                                    llvm::cl::init(0));

static llvm::cl::opt<int> transCoeffMethod("trans-coeff-method",
                                    llvm::cl::desc("Specify the assumption made for the transformation coefficients for a single loop variable"),
                                    llvm::cl::value_desc("trans-coeff"),
                                    llvm::cl::init(4));

static llvm::cl::opt<int> numCountingMethod("number-counting-method",
                                    llvm::cl::desc("Specify the estimation methods for the input number"),
                                    llvm::cl::value_desc("number-counting"),
                                    llvm::cl::init(0));

static llvm::cl::opt<int> storageMethod("storage-method",
                                    llvm::cl::desc("Specify the storage method for the column"),
                                    llvm::cl::value_desc("storage-method"),
                                    llvm::cl::init(1));

static llvm::cl::opt<int> objMethod("obj-method",
                                    llvm::cl::desc("Specify the objective function for the milp"),
                                    llvm::cl::value_desc("obj-approx"),
                                    llvm::cl::init(0));

static llvm::cl::opt<std::string>
    configArchPath("config-arch-path",
                   llvm::cl::desc("Specify path to architecture json file"),
                   llvm::cl::value_desc("arch-json"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    configKnobPath("config-knobs-path",
                   llvm::cl::desc("Specify path to tuninig knob json file"),
                   llvm::cl::value_desc("knob-json"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::ml_program::MLProgramDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect>();

  auto [inputFilename, outputFilename] = mlir::registerAndParseCLIOptions(
      argc, argv, "PIM Optimization Pass", registry);

  std::string errorMessage;
  auto inputBuffer = openInputFile(inputFilename, &errorMessage);
  if (!inputBuffer) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();
  config.allowUnregisteredDialects(true);
  config.setPassPipelineSetupFn([&](mlir::PassManager &pm) {
    if (useDataLayoutPass) {
      auto dataLayoutPass = std::make_unique<pim::DataLayoutPass>();

      dataLayoutPass->setOutputPath(traceOutputPath);
      dataLayoutPass->setDeviceType(deviceType);
      dataLayoutPass->setMemAlloc(memAlloScheme);
      dataLayoutPass->setTransCoeffMethod(transCoeffMethod);
      dataLayoutPass->setNumCountingMethod(numCountingMethod);
      dataLayoutPass->setStorageMethod(storageMethod);
      dataLayoutPass->setMilpPath(modelDebugPath);
      dataLayoutPass->setArchPath(configArchPath);
      dataLayoutPass->setKnobPath(configKnobPath);
      dataLayoutPass->setObjMethod(objMethod);

      pm.addPass(std::move(dataLayoutPass));
    }

    return mlir::success();
  });

  if (failed(mlir::MlirOptMain(output->os(), std::move(inputBuffer), registry,
                               config))) {
    llvm::errs() << "Error: failed to process the input MLIR file.\n";
    return 1;
  }

  output->keep();
  return 0;
}
