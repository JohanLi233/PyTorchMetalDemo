#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/SingleHeader/Metal.hpp"
#include "CustomOP.metal"

#include <torch/extension.h>
#include <iostream>
#include <memory>

// Custom deleter for Metal objects
struct MTLDeviceDeleter
{
  void operator()(MTL::Device *device) const
  {
    if (device)
      device->release();
  }
};

struct MTLBufferDeleter
{
  void operator()(MTL::Buffer *buffer) const
  {
    if (buffer)
      buffer->release();
  }
};

struct MTLLibraryDeleter
{
  void operator()(MTL::Library *library) const
  {
    if (library)
      library->release();
  }
};

struct MTLFunctionDeleter
{
  void operator()(MTL::Function *function) const
  {
    if (function)
      function->release();
  }
};

struct MTLComputePipelineStateDeleter
{
  void operator()(MTL::ComputePipelineState *pipelineState) const
  {
    if (pipelineState)
      pipelineState->release();
  }
};

struct MTLCommandQueueDeleter
{
  void operator()(MTL::CommandQueue *commandQueue) const
  {
    if (commandQueue)
      commandQueue->release();
  }
};

struct MTLCommandBufferDeleter
{
  void operator()(MTL::CommandBuffer *commandBuffer) const
  {
    if (commandBuffer)
      commandBuffer->release();
  }
};

struct MTLComputeCommandEncoderDeleter
{
  void operator()(MTL::ComputeCommandEncoder *encoder) const
  {
    if (encoder)
      encoder->release();
  }
};

torch::Tensor custom_add(torch::Tensor input1, torch::Tensor input2)
{
  TORCH_CHECK(input1.device().is_mps(), "Input1 must be an MPS tensor");
  TORCH_CHECK(input2.device().is_mps(), "Input2 must be an MPS tensor");
  TORCH_CHECK(input1.dtype() == torch::kFloat && input2.dtype() == torch::kFloat,
              "Input tensors must be of float type");
  TORCH_CHECK(input1.sizes() == input2.sizes(), "Input tensors must have the same shape");

  // Create Metal device
  std::unique_ptr<MTL::Device, MTLDeviceDeleter> device(MTL::CreateSystemDefaultDevice());
  TORCH_CHECK(device, "Unable to create MTL device");

  auto shape = input1.sizes();
  int64_t numElements = input1.numel();

  // Allocate buffers
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> in1Buffer(
      device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> in2Buffer(
      device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> outBuffer(
      device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> sizeBuffer(
      device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared));

  TORCH_CHECK(in1Buffer && in2Buffer && outBuffer && sizeBuffer, "Unable to create buffers");

  // Copy input data into the Metal buffers
  {
    torch::Tensor input1_cpu = input1.to(torch::kCPU);
    torch::Tensor input2_cpu = input2.to(torch::kCPU);

    float *in1Ptr = static_cast<float *>(in1Buffer->contents());
    float *in2Ptr = static_cast<float *>(in2Buffer->contents());
    std::memcpy(in1Ptr, input1_cpu.data_ptr<float>(), numElements * sizeof(float));
    std::memcpy(in2Ptr, input2_cpu.data_ptr<float>(), numElements * sizeof(float));
  }

  // Set the size
  uint *sizePtr = static_cast<uint *>(sizeBuffer->contents());
  *sizePtr = static_cast<uint>(numElements);

  // Load the Metal kernel
  NS::Error *error = nullptr;
  std::unique_ptr<MTL::Library, MTLLibraryDeleter> library(
      device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error));
  TORCH_CHECK(library, "Unable to create kernel library, error: " + std::string(error->localizedDescription()->utf8String()));

  std::unique_ptr<MTL::Function, MTLFunctionDeleter> function(
      library->newFunction(NS::String::string("custom_add", NS::UTF8StringEncoding)));
  std::unique_ptr<MTL::ComputePipelineState, MTLComputePipelineStateDeleter> pipelineState(
      device->newComputePipelineState(function.get(), &error));
  TORCH_CHECK(pipelineState, "Unable to create compute pipeline state, error: " + std::string(error->localizedDescription()->utf8String()));

  // Create command queue and buffer
  std::unique_ptr<MTL::CommandQueue, MTLCommandQueueDeleter> commandQueue(device->newCommandQueue());
  std::unique_ptr<MTL::CommandBuffer, MTLCommandBufferDeleter> commandBuffer(commandQueue->commandBuffer());
  std::unique_ptr<MTL::ComputeCommandEncoder, MTLComputeCommandEncoderDeleter> encoder(commandBuffer->computeCommandEncoder());

  // Set pipeline and buffers
  encoder->setComputePipelineState(pipelineState.get());
  encoder->setBuffer(in1Buffer.get(), 0, 0);
  encoder->setBuffer(in2Buffer.get(), 0, 1);
  encoder->setBuffer(outBuffer.get(), 0, 2);
  encoder->setBuffer(sizeBuffer.get(), 0, 3);

  // Dispatch
  const uint threadsPerThreadgroup = pipelineState->maxTotalThreadsPerThreadgroup();
  const uint threadgroups = (numElements + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

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

  return outputTensor;
}

torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // Ensure the input tensor is on the MPS device and is float type
  TORCH_CHECK(input.device().is_mps(), "Input must be an MPS tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of float type");

  // Create Metal device
  std::unique_ptr<MTL::Device, MTLDeviceDeleter> device(MTL::CreateSystemDefaultDevice());
  TORCH_CHECK(device, "Unable to create MTL device");

  // Gather shape and size information
  auto shape = input.sizes();
  int64_t numElements = input.numel();
  TORCH_CHECK(numElements > 0, "Input tensor must have at least one element");

  // Create Metal buffers:
  // - outputBuffer: Will store the filled values
  // - fillValBuffer: Holds the fill value
  // - sizeBuffer: Holds the number of elements
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> outputBuffer(
      device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> fillValBuffer(
      device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> sizeBuffer(
      device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared));

  TORCH_CHECK(outputBuffer && fillValBuffer && sizeBuffer, "Unable to create required buffers");

  // Initialize the fill value
  float *fillValPtr = static_cast<float *>(fillValBuffer->contents());
  *fillValPtr = fill_val;

  // Set the size buffer
  uint *sizePtr = static_cast<uint *>(sizeBuffer->contents());
  *sizePtr = static_cast<uint>(numElements);

  // Load Metal kernel from CUSTOM_KERNEL string
  NS::Error *error = nullptr;
  std::unique_ptr<MTL::Library, MTLLibraryDeleter> library(
      device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error));
  TORCH_CHECK(library, "Unable to create kernel library" +
                           (error ? (": " + std::string(error->localizedDescription()->utf8String())) : ""));

  std::unique_ptr<MTL::Function, MTLFunctionDeleter> function(
      library->newFunction(NS::String::string("custom_fill", NS::UTF8StringEncoding)));
  TORCH_CHECK(function, "Unable to find 'custom_fill' function in library");

  std::unique_ptr<MTL::ComputePipelineState, MTLComputePipelineStateDeleter> pipelineState(
      device->newComputePipelineState(function.get(), &error));
  TORCH_CHECK(pipelineState, "Unable to create compute pipeline state" +
                                 (error ? (": " + std::string(error->localizedDescription()->utf8String())) : ""));

  // Create command queue and command buffer
  std::unique_ptr<MTL::CommandQueue, MTLCommandQueueDeleter> commandQueue(device->newCommandQueue());
  TORCH_CHECK(commandQueue, "Unable to create command queue");

  std::unique_ptr<MTL::CommandBuffer, MTLCommandBufferDeleter> commandBuffer(commandQueue->commandBuffer());
  TORCH_CHECK(commandBuffer, "Unable to create command buffer");

  // Create and set up the compute command encoder
  std::unique_ptr<MTL::ComputeCommandEncoder, MTLComputeCommandEncoderDeleter> encoder(commandBuffer->computeCommandEncoder());
  TORCH_CHECK(encoder, "Unable to create compute command encoder");

  encoder->setComputePipelineState(pipelineState.get());
  encoder->setBuffer(outputBuffer.get(), 0, 0);  // output data
  encoder->setBuffer(fillValBuffer.get(), 0, 1); // fill_val
  encoder->setBuffer(sizeBuffer.get(), 0, 2);    // data_size

  // Determine threadgroup configuration
  const uint threadsPerThreadgroup = pipelineState->maxTotalThreadsPerThreadgroup();
  const uint threadgroups = (static_cast<uint>(numElements) + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

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

  return outputTensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
  m.def("custom_add", &custom_add, "Custom add function for MPS");
}
