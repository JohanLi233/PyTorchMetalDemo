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

// Helper function: Check tensor constraints
void checkTensorConstraints(const torch::Tensor& tensor, bool checkMPS = true, bool checkFloat = true) {
  if (checkMPS) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor must be an MPS tensor");
  }
  if (checkFloat) {
    TORCH_CHECK(tensor.dtype() == torch::kFloat, "Tensor must be of float type");
  }
}

// Helper function: Create Metal buffer
MTL::Buffer* createMetalBuffer(MTL::Device* device, size_t size, MTL::ResourceOptions options = MTL::ResourceStorageModeShared) {
  MTL::Buffer* buffer = device->newBuffer(size, options);
  if (!buffer) {
    throw std::runtime_error("Unable to create Metal buffer");
  }
  return buffer;
}

// Helper function: Copy tensor data to Metal buffer
void copyTensorToBuffer(const torch::Tensor& tensor, MTL::Buffer* buffer, int64_t numElements) {
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU);
  float* bufferPtr = static_cast<float*>(buffer->contents());
  std::memcpy(bufferPtr, tensor_cpu.data_ptr<float>(), numElements * sizeof(float));
}

// Helper function: Execute Metal computation
torch::Tensor executeMetalComputation(
    const std::string& kernelName,
    std::vector<MTL::Buffer*> buffers,
    torch::IntArrayRef shape,
    int64_t numElements,
    const torch::Device& device) {
  
  try {
    // Get Metal context and pipeline state
    MetalContext& context = MetalContext::getInstance();
    MTL::ComputePipelineState* pipelineState = context.getPipelineState(kernelName);
    
    // Create command buffer and encoder
    auto [commandBuffer, encoder] = context.createCommandBufferAndEncoder();
    
    // Set pipeline and buffers
    encoder->setComputePipelineState(pipelineState);
    for (size_t i = 0; i < buffers.size(); i++) {
      encoder->setBuffer(buffers[i], 0, i);
    }
    
    // Calculate and set thread groups configuration
    uint threadgroups, threadsPerThreadgroup;
    calculateThreadGroups(pipelineState, numElements, threadgroups, threadsPerThreadgroup);
    encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
    encoder->endEncoding();
    
    // Run the command buffer
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Copy result back into a new MPS tensor (assuming output is at index 2 for binary ops, index 0 for unary ops)
    MTL::Buffer* outputBuffer = (kernelName == "custom_fill") ? buffers[0] : buffers[2];
    torch::Tensor outputTensor = torch::from_blob(
        outputBuffer->contents(), shape, torch::kFloat)
        .clone()
        .to(device);
    
    // Release Metal resources
    for (auto* buffer : buffers) {
      buffer->release();
    }
    encoder->release();
    commandBuffer->release();
    
    return outputTensor;
  } catch (const std::exception& e) {
    // Clean up buffers on error
    for (auto* buffer : buffers) {
      buffer->release();
    }
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

// Custom add operation implementation
torch::Tensor custom_add(torch::Tensor input1, torch::Tensor input2)
{
  // Validate inputs
  checkTensorConstraints(input1);
  checkTensorConstraints(input2);
  TORCH_CHECK(input1.sizes() == input2.sizes(), "Input tensors must have the same shape");

  auto shape = input1.sizes();
  int64_t numElements = input1.numel();
  
  try {
    // Get Metal context
    MetalContext& context = MetalContext::getInstance();
    MTL::Device* device = context.getDevice();
    
    // Allocate buffers
    MTL::Buffer* in1Buffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* in2Buffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* outBuffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* sizeBuffer = createMetalBuffer(device, sizeof(uint));
    
    // Copy input data into the Metal buffers
    copyTensorToBuffer(input1, in1Buffer, numElements);
    copyTensorToBuffer(input2, in2Buffer, numElements);
    
    // Set the size
    uint* sizePtr = static_cast<uint*>(sizeBuffer->contents());
    *sizePtr = static_cast<uint>(numElements);
    
    // Execute computation
    std::vector<MTL::Buffer*> buffers = {in1Buffer, in2Buffer, outBuffer, sizeBuffer};
    return executeMetalComputation("custom_add", buffers, shape, numElements, input1.device());
    
  } catch (const std::exception& e) {
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

// Custom fill operation implementation
torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // Ensure the input tensor is on the MPS device and is float type
  checkTensorConstraints(input);
  
  auto shape = input.sizes();
  int64_t numElements = input.numel();
  TORCH_CHECK(numElements > 0, "Input tensor must have at least one element");
  
  try {
    // Get Metal context
    MetalContext& context = MetalContext::getInstance();
    MTL::Device* device = context.getDevice();
    
    // Create Metal buffers
    MTL::Buffer* outputBuffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* fillValBuffer = createMetalBuffer(device, sizeof(float));
    MTL::Buffer* sizeBuffer = createMetalBuffer(device, sizeof(uint));
    
    // Initialize the fill value
    float* fillValPtr = static_cast<float*>(fillValBuffer->contents());
    *fillValPtr = fill_val;
    
    // Set the size buffer
    uint* sizePtr = static_cast<uint*>(sizeBuffer->contents());
    *sizePtr = static_cast<uint>(numElements);
    
    // Execute computation
    std::vector<MTL::Buffer*> buffers = {outputBuffer, fillValBuffer, sizeBuffer};
    return executeMetalComputation("custom_fill", buffers, shape, numElements, input.device());
    
  } catch (const std::exception& e) {
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

// Custom multiply operation implementation
torch::Tensor custom_multiply(torch::Tensor input1, torch::Tensor input2)
{
  // Validate inputs
  checkTensorConstraints(input1);
  checkTensorConstraints(input2);
  TORCH_CHECK(input1.sizes() == input2.sizes(), "Input tensors must have the same shape");

  auto shape = input1.sizes();
  int64_t numElements = input1.numel();
  
  try {
    // Get Metal context
    MetalContext& context = MetalContext::getInstance();
    MTL::Device* device = context.getDevice();
    
    // Allocate buffers
    MTL::Buffer* in1Buffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* in2Buffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* outBuffer = createMetalBuffer(device, numElements * sizeof(float));
    MTL::Buffer* sizeBuffer = createMetalBuffer(device, sizeof(uint));
    
    // Copy input data into the Metal buffers
    copyTensorToBuffer(input1, in1Buffer, numElements);
    copyTensorToBuffer(input2, in2Buffer, numElements);
    
    // Set the size
    uint* sizePtr = static_cast<uint*>(sizeBuffer->contents());
    *sizePtr = static_cast<uint>(numElements);
    
    // Execute computation
    std::vector<MTL::Buffer*> buffers = {in1Buffer, in2Buffer, outBuffer, sizeBuffer};
    return executeMetalComputation("custom_multiply", buffers, shape, numElements, input1.device());
    
  } catch (const std::exception& e) {
    std::cerr << "Metal execution error: " << e.what() << std::endl;
    throw; // Re-throw the exception
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
  m.def("custom_add", &custom_add, "Custom add function for MPS");
  m.def("custom_multiply", &custom_multiply, "Custom multiply function for MPS");
}
