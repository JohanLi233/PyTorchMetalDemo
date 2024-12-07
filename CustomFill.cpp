#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/SingleHeader/Metal.hpp"
#include "CustomFill.metal"

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

torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // Ensure the input tensor is on the MPS device
  TORCH_CHECK(input.device().is_mps(), "Input must be an MPS tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of float type");

  // Create Metal device
  std::unique_ptr<MTL::Device, MTLDeviceDeleter> device(MTL::CreateSystemDefaultDevice());
  TORCH_CHECK(device, "Unable to create MTL device");

  // Get the shape and size information of the tensor
  auto shape = input.sizes();
  int64_t numElements = input.numel();

  // Create Metal buffers
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> inputBuffer(
      device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> fillValBuffer(
      device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared));
  std::unique_ptr<MTL::Buffer, MTLBufferDeleter> sizeBuffer(
      device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared));

  TORCH_CHECK(inputBuffer && fillValBuffer && sizeBuffer, "Unable to create buffers");

  // Copy input data to Metal buffer
  memcpy(inputBuffer->contents(), input.data_ptr(), numElements * sizeof(float));

  // Copy fill value to buffer
  float *fillValPtr = static_cast<float *>(fillValBuffer->contents());
  *fillValPtr = fill_val;

  // Copy the number of elements to the size buffer
  uint *sizePtr = static_cast<uint *>(sizeBuffer->contents());
  *sizePtr = static_cast<uint>(numElements);

  // Load Metal kernel
  NS::Error *error = nullptr;
  std::unique_ptr<MTL::Library, MTLLibraryDeleter> library(
      device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error));
  TORCH_CHECK(library, "Unable to create kernel library, error: " + std::string(error->localizedDescription()->utf8String()));

  std::unique_ptr<MTL::Function, MTLFunctionDeleter> function(
      library->newFunction(NS::String::string("custom_fill", NS::UTF8StringEncoding)));
  std::unique_ptr<MTL::ComputePipelineState, MTLComputePipelineStateDeleter> pipelineState(
      device->newComputePipelineState(function.get(), &error));
  TORCH_CHECK(pipelineState, "Unable to create compute pipeline state, error: " + std::string(error->localizedDescription()->utf8String()));

  // Create command queue and buffer
  std::unique_ptr<MTL::CommandQueue, MTLCommandQueueDeleter> commandQueue(device->newCommandQueue());
  std::unique_ptr<MTL::CommandBuffer, MTLCommandBufferDeleter> commandBuffer(commandQueue->commandBuffer());
  std::unique_ptr<MTL::ComputeCommandEncoder, MTLComputeCommandEncoderDeleter> encoder(commandBuffer->computeCommandEncoder());

  encoder->setComputePipelineState(pipelineState.get());
  encoder->setBuffer(inputBuffer.get(), 0, 0);   // data
  encoder->setBuffer(fillValBuffer.get(), 0, 1); // fill_val
  encoder->setBuffer(sizeBuffer.get(), 0, 2);    // data_size

  // Dispatch threadgroups dynamically based on number of elements
  const uint threadsPerThreadgroup = pipelineState->maxTotalThreadsPerThreadgroup();
  const uint threadgroups = (numElements + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();

  // Commit the command buffer and wait for completion
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // Copy result back to MPS tensor
  torch::Tensor outputTensor = torch::from_blob(
                                   inputBuffer->contents(), shape, torch::kFloat)
                                   .clone()
                                   .to(input.device());

  return outputTensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
}
