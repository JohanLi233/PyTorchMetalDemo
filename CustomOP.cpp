#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/SingleHeader/Metal.hpp"
#include "CustomOP.metal"

#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <unordered_map>

// Metal Context class to manage Metal resources
class MetalContext {
private:
  static std::unique_ptr<MetalContext> instance;
  
  // Metal resources
  MTL::Device* device;
  MTL::CommandQueue* commandQueue;
  std::unordered_map<std::string, MTL::ComputePipelineState*> pipelineCache;
  MTL::Library* library;

  // Private constructor
  MetalContext() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("Unable to create Metal device");
    }
    
    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
      device->release();
      throw std::runtime_error("Unable to create command queue");
    }
    
    // Load Metal kernel library
    NS::Error* error = nullptr;
    library = device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error);
    if (!library) {
      std::string errorMsg = "Unable to create kernel library";
      if (error) {
        errorMsg += ": " + std::string(error->localizedDescription()->utf8String());
      }
      commandQueue->release();
      device->release();
      throw std::runtime_error(errorMsg);
    }
  }

public:
  // Prevent copying
  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;
  
  // Destructor
  ~MetalContext() {
    // Release all cached pipelines
    for (auto& pair : pipelineCache) {
      if (pair.second) {
        pair.second->release();
      }
    }
    
    if (library) {
      library->release();
    }
    
    if (commandQueue) {
      commandQueue->release();
    }
    
    if (device) {
      device->release();
    }
  }
  
  // Get singleton instance
  static MetalContext& getInstance() {
    if (!instance) {
      instance = std::unique_ptr<MetalContext>(new MetalContext());
    }
    return *instance;
  }
  
  // Get device
  MTL::Device* getDevice() const {
    return device;
  }
  
  // Get command queue
  MTL::CommandQueue* getCommandQueue() const {
    return commandQueue;
  }
  
  // Get or create compute pipeline state
  MTL::ComputePipelineState* getPipelineState(const std::string& functionName) {
    // If pipeline is already cached, return it
    if (pipelineCache.find(functionName) != pipelineCache.end()) {
      return pipelineCache[functionName];
    }
    
    // Otherwise create new pipeline
    NS::Error* error = nullptr;
    MTL::Function* function = library->newFunction(NS::String::string(functionName.c_str(), NS::UTF8StringEncoding));
    if (!function) {
      throw std::runtime_error("Unable to find function: " + functionName);
    }
    
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(function, &error);
    function->release();
    
    if (!pipelineState) {
      std::string errorMsg = "Unable to create compute pipeline state: " + functionName;
      if (error) {
        errorMsg += ": " + std::string(error->localizedDescription()->utf8String());
      }
      throw std::runtime_error(errorMsg);
    }
    
    // Cache and return
    pipelineCache[functionName] = pipelineState;
    return pipelineState;
  }
  
  // Create command buffer and encoder
  std::pair<MTL::CommandBuffer*, MTL::ComputeCommandEncoder*> createCommandBufferAndEncoder() {
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    if (!commandBuffer) {
      throw std::runtime_error("Unable to create command buffer");
    }
    
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    if (!encoder) {
      commandBuffer->release();
      throw std::runtime_error("Unable to create compute command encoder");
    }
    
    return {commandBuffer, encoder};
  }
};

// Initialize singleton pointer
std::unique_ptr<MetalContext> MetalContext::instance = nullptr;

// Helper function: Calculate thread groups configuration
void calculateThreadGroups(MTL::ComputePipelineState* pipelineState, int64_t numElements, 
                          uint& threadgroups, uint& threadsPerThreadgroup) {
  threadsPerThreadgroup = pipelineState->maxTotalThreadsPerThreadgroup();
  threadgroups = (static_cast<uint>(numElements) + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
}

// Custom add operation implementation
torch::Tensor custom_add(torch::Tensor input1, torch::Tensor input2)
{
  TORCH_CHECK(input1.device().is_mps(), "Input1 must be an MPS tensor");
  TORCH_CHECK(input2.device().is_mps(), "Input2 must be an MPS tensor");
  TORCH_CHECK(input1.dtype() == torch::kFloat && input2.dtype() == torch::kFloat,
              "Input tensors must be of float type");
  TORCH_CHECK(input1.sizes() == input2.sizes(), "Input tensors must have the same shape");

  auto shape = input1.sizes();
  int64_t numElements = input1.numel();
  
  try {
    // Get Metal context
    MetalContext& context = MetalContext::getInstance();
    MTL::Device* device = context.getDevice();
    
    // Allocate buffers
    MTL::Buffer* in1Buffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* in2Buffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* outBuffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* sizeBuffer = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    
    if (!in1Buffer || !in2Buffer || !outBuffer || !sizeBuffer) {
      if (in1Buffer) in1Buffer->release();
      if (in2Buffer) in2Buffer->release();
      if (outBuffer) outBuffer->release();
      if (sizeBuffer) sizeBuffer->release();
      throw std::runtime_error("Unable to create buffers");
    }
    
    // Copy input data into the Metal buffers
    {
      torch::Tensor input1_cpu = input1.to(torch::kCPU);
      torch::Tensor input2_cpu = input2.to(torch::kCPU);
      
      float* in1Ptr = static_cast<float*>(in1Buffer->contents());
      float* in2Ptr = static_cast<float*>(in2Buffer->contents());
      std::memcpy(in1Ptr, input1_cpu.data_ptr<float>(), numElements * sizeof(float));
      std::memcpy(in2Ptr, input2_cpu.data_ptr<float>(), numElements * sizeof(float));
    }
    
    // Set the size
    uint* sizePtr = static_cast<uint*>(sizeBuffer->contents());
    *sizePtr = static_cast<uint>(numElements);
    
    // Get compute pipeline state
    MTL::ComputePipelineState* pipelineState = context.getPipelineState("custom_add");
    
    // Create command buffer and encoder
    auto [commandBuffer, encoder] = context.createCommandBufferAndEncoder();
    
    // Set pipeline and buffers
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(in1Buffer, 0, 0);
    encoder->setBuffer(in2Buffer, 0, 1);
    encoder->setBuffer(outBuffer, 0, 2);
    encoder->setBuffer(sizeBuffer, 0, 3);
    
    // Calculate and set thread groups configuration
    uint threadgroups, threadsPerThreadgroup;
    calculateThreadGroups(pipelineState, numElements, threadgroups, threadsPerThreadgroup);
    encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
    encoder->endEncoding();
    
    // Run the command buffer
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back into a new MPS tensor
    torch::Tensor outputTensor = torch::from_blob(
        outBuffer->contents(), shape, torch::kFloat)
        .clone()
        .to(input1.device());
    
    // Release Metal resources
    in1Buffer->release();
    in2Buffer->release();
    outBuffer->release();
    sizeBuffer->release();
    encoder->release();
    commandBuffer->release();
    
    return outputTensor;
  } catch (const std::exception& e) {
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

// Custom fill operation implementation
torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // Ensure the input tensor is on the MPS device and is float type
  TORCH_CHECK(input.device().is_mps(), "Input must be an MPS tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of float type");
  
  auto shape = input.sizes();
  int64_t numElements = input.numel();
  TORCH_CHECK(numElements > 0, "Input tensor must have at least one element");
  
  try {
    // Get Metal context
    MetalContext& context = MetalContext::getInstance();
    MTL::Device* device = context.getDevice();
    
    // Create Metal buffers:
    // - outputBuffer: Will store the filled values
    // - fillValBuffer: Holds the fill value
    // - sizeBuffer: Holds the number of elements
    MTL::Buffer* outputBuffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* fillValBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* sizeBuffer = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    
    if (!outputBuffer || !fillValBuffer || !sizeBuffer) {
      if (outputBuffer) outputBuffer->release();
      if (fillValBuffer) fillValBuffer->release();
      if (sizeBuffer) sizeBuffer->release();
      throw std::runtime_error("Unable to create required buffers");
    }
    
    // Initialize the fill value
    float* fillValPtr = static_cast<float*>(fillValBuffer->contents());
    *fillValPtr = fill_val;
    
    // Set the size buffer
    uint* sizePtr = static_cast<uint*>(sizeBuffer->contents());
    *sizePtr = static_cast<uint>(numElements);
    
    // Get compute pipeline state
    MTL::ComputePipelineState* pipelineState = context.getPipelineState("custom_fill");
    
    // Create command buffer and encoder
    auto [commandBuffer, encoder] = context.createCommandBufferAndEncoder();
    
    // Set up the compute command encoder
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(outputBuffer, 0, 0);  // output data
    encoder->setBuffer(fillValBuffer, 0, 1); // fill_val
    encoder->setBuffer(sizeBuffer, 0, 2);    // data_size
    
    // Determine threadgroup configuration
    uint threadgroups, threadsPerThreadgroup;
    calculateThreadGroups(pipelineState, numElements, threadgroups, threadsPerThreadgroup);
    encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
    encoder->endEncoding();
    
    // Commit and wait for completion
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Create a new tensor from the output buffer.
    // Use from_blob() on CPU and then move it back to MPS for consistency.
    torch::Tensor outputTensor = torch::from_blob(
        outputBuffer->contents(), shape, torch::kFloat)
        .clone()
        .to(input.device());
    
    // Release Metal resources
    outputBuffer->release();
    fillValBuffer->release();
    sizeBuffer->release();
    encoder->release();
    commandBuffer->release();
    
    return outputTensor;
  } catch (const std::exception& e) {
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
  m.def("custom_add", &custom_add, "Custom add function for MPS");
}
