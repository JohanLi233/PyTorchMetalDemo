#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/SingleHeader/Metal.hpp"
#include "CustomFill.metal"

#include <torch/extension.h>
#include <iostream>

static inline const MTL::Buffer *getMTLBufferStorage(const torch::Tensor &tensor)
{
  return reinterpret_cast<const MTL::Buffer *>(tensor.storage().data()); // Cast to MTL::Buffer*
}

torch::Tensor custom_fill(torch::Tensor input, float fill_val)
{
  // Ensure the input tensor is on the MPS device
  TORCH_CHECK(input.device().is_mps(), "Input must be an MPS tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of float type");

  // Create Metal device
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device)
  {
    throw std::runtime_error("Unable to create MTL device");
  }

  // Get the shape and size information of the tensor
  auto shape = input.sizes();
  int64_t numElements = input.numel();

  // Create Metal buffers
  MTL::Buffer *inputBuffer = device->newBuffer(numElements * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *fillValBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *sizeBuffer = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
  if (!inputBuffer || !fillValBuffer || !sizeBuffer)
  {
    throw std::runtime_error("Unable to create buffers");
  }

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
  MTL::Library *library = device->newLibrary(NS::String::string(CUSTOM_KERNEL, NS::UTF8StringEncoding), nullptr, &error);
  if (!library)
  {
    throw std::runtime_error(std::string("Unable to create kernel library, error: ") + error->localizedDescription()->utf8String());
  }

  MTL::Function *function = library->newFunction(NS::String::string("custom_fill", NS::UTF8StringEncoding));
  MTL::ComputePipelineState *pipelineState = device->newComputePipelineState(function, &error);
  if (!pipelineState)
  {
    throw std::runtime_error(std::string("Unable to create compute pipeline state, error: ") + error->localizedDescription()->utf8String());
  }

  // Create command queue and command buffer
  MTL::CommandQueue *commandQueue = device->newCommandQueue();
  MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(pipelineState);
  encoder->setBuffer(inputBuffer, 0, 0);   // data
  encoder->setBuffer(fillValBuffer, 0, 1); // fill_val
  encoder->setBuffer(sizeBuffer, 0, 2);    // data_size

  // Set thread groups and number of threads
  const uint threadsPerThreadgroup = 1024;
  const uint threadgroups = (numElements + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();

  // Submit command buffer and wait until completed
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  // Copy result back to mps
  torch::Tensor outputTensor = torch::from_blob(inputBuffer->contents(), shape, torch::kFloat).clone().to(input.device());

  // Clean up resources
  library->release();
  function->release();
  pipelineState->release();
  commandQueue->release();
  inputBuffer->release();
  fillValBuffer->release();
  sizeBuffer->release();
  commandBuffer->release();
  encoder->release();
  device->release();

  return outputTensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("custom_fill", &custom_fill, "Custom fill function for MPS");
}
