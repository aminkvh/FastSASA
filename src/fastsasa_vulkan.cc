#include "fastsasa_vulkan_internal.h"
#include "fastsasa_cpu.h"

/* volk loads the Vulkan loader at runtime so FastSASA binaries do not carry a
 * hard DT_NEEDED dependency on libvulkan.so.1. */
#include "volk.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

#include "fastsasa_vk_cell_count_spv.h"
#include "fastsasa_vk_cell_scan_spv.h"
#include "fastsasa_vk_cell_fill_spv.h"
#include "fastsasa_vk_cell_gather_spv.h"
#include "fastsasa_vk_sr_cell_list_32_spv.h"
#include "fastsasa_vk_sr_cell_list_64_spv.h"
#include "fastsasa_vk_sr_cell_list_sg_spv.h"
#include "fastsasa_vk_lee_richards_cell_spv.h"
#include "fastsasa_vk_lee_richards_reduce_spv.h"
#include "fastsasa_vk_cell_count_fp64_spv.h"
#include "fastsasa_vk_cell_fill_fp64_spv.h"
#include "fastsasa_vk_sr_cell_list_fp64_32_spv.h"
#include "fastsasa_vk_sr_cell_list_fp64_64_spv.h"
#include "fastsasa_vk_sr_cell_list_fp64_sg_spv.h"
#include "fastsasa_vk_lee_richards_cell_fp64_spv.h"
#include "fastsasa_vk_lee_richards_reduce_fp64_spv.h"
#include "fastsasa_vk_sr_exposed_points_fp64_64_spv.h"

constexpr uint32_t kBinThreads = 128;
constexpr uint32_t kSrThreads = 64;
constexpr uint64_t kMaxCells = 16ull * 1024ull * 1024ull;
constexpr double kPi = 3.14159265358979323846;

/* Mirrors the CUDA backend's kill switch: the FP32-prefiltered FP64 path is
 * on by default and FASTSASA_SR_FP64_HYBRID=0 restores pure-FP64 behavior by
 * widening the uncertainty margin so every test re-runs in FP64. */
bool sr_fp64_hybrid_enabled()
{
    const char *value = std::getenv("FASTSASA_SR_FP64_HYBRID");
    if (value == nullptr || value[0] == '\0') return true;
    return std::strcmp(value, "0") != 0;
}

/* Conservative FP32 uncertainty margin; see the CUDA launch site for the
 * derivation. Coordinates are frame-local, so magnitudes are bounded by the
 * cell-grid extent. */
double sr_hybrid_margin(double cell_size, uint32_t dim_x, uint32_t dim_y,
                        uint32_t dim_z)
{
    const double eps32 = 5.9604644775390625e-8; /* 2^-24 */
    const uint32_t max_dim = std::max(dim_x, std::max(dim_y, dim_z));
    const double extent = cell_size * (double)(max_dim + 1u);
    const double dmax = 6.0 * cell_size;
    const double delta = 8.0 * eps32 * extent;

    if (!sr_fp64_hybrid_enabled()) return 1.0e30;
    return 2.0 * (2.0 * dmax * delta + delta * delta + 8.0 * eps32 * dmax * dmax);
}

/* The SR shader returns the exact exposed-point count; the area formula runs
 * on the host in FP64 in the same operation order as the CUDA kernels. */
double sr_count_to_area(double exposed_count,
                        double radius,
                        double probe_radius,
                        double point_count)
{
    const double expanded_radius = radius + probe_radius;
    return 4.0 * kPi * expanded_radius * expanded_radius * exposed_count /
           point_count;
}

template <typename Real>
struct Vec4T {
    Real x, y, z, w;
};

template <typename Real>
struct ParametersT {
    uint32_t atom_count;
    uint32_t point_count;
    uint32_t center_count;
    uint32_t base_index;
    uint32_t dim_x;
    uint32_t dim_y;
    uint32_t dim_z;
    Real probe_radius;
    Real cell_size;
    Real min_x;
    Real min_y;
    Real min_z;
    Real margin;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    VkDeviceSize capacity = 0;
};

void check(VkResult result, const char *operation)
{
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(static_cast<int>(result)));
    }
}

uint32_t find_memory_type(VkPhysicalDevice physical,
                          uint32_t allowed,
                          VkMemoryPropertyFlags required)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((allowed & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("Vulkan device has no compatible memory type");
}

/*
 * Memory for buffers the host reads back. Plain HOST_VISIBLE|HOST_COHERENT
 * memory is typically write-combined and uncached, so reading results
 * element by element costs a full uncached transaction each (measured at
 * ~1.6 ms per 13,658-atom frame, more than the SR compute itself). Prefer a
 * HOST_CACHED type when the device offers one.
 */
VkMemoryPropertyFlags readback_memory_flags(VkPhysicalDevice physical)
{
    const VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((properties.memoryTypes[i].propertyFlags & cached) == cached) return cached;
    }
    return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

VkPhysicalDevice select_device(VkInstance instance,
                               int requested_index,
                               uint32_t *queue_family,
                               VkPhysicalDeviceProperties *selected_properties)
{
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr),
          "vkEnumeratePhysicalDevices");
    if (count == 0) throw std::runtime_error("no Vulkan devices found");
    std::vector<VkPhysicalDevice> devices(count);
    const VkResult enumerated =
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
    if (enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) {
        check(enumerated, "vkEnumeratePhysicalDevices");
    }
    devices.resize(count);

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    uint32_t fallback_family = 0;
    VkPhysicalDeviceProperties fallback_properties{};
    int compute_index = 0;
    for (VkPhysicalDevice candidate : devices) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
        for (uint32_t family = 0; family < family_count; ++family) {
            if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            /* FastSASA requires Vulkan 1.1 device behavior; probing 1.0-only
             * ICDs with 1.1 queries can crash old drivers. */
            if (properties.apiVersion < VK_API_VERSION_1_1) break;
            if (requested_index == compute_index) {
                *queue_family = family;
                *selected_properties = properties;
                return candidate;
            }
            ++compute_index;
            if (requested_index < 0 &&
                properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                *queue_family = family;
                *selected_properties = properties;
                return candidate;
            }
            if (fallback == VK_NULL_HANDLE) {
                fallback = candidate;
                fallback_family = family;
                fallback_properties = properties;
            }
            break;
        }
    }
    if (requested_index >= 0) {
        throw std::runtime_error("requested Vulkan compute device index is unavailable");
    }
    if (fallback == VK_NULL_HANDLE) {
        throw std::runtime_error("no Vulkan 1.1 compute queue found");
    }
    *queue_family = fallback_family;
    *selected_properties = fallback_properties;
    return fallback;
}

} // namespace

struct fastsasa_vk_context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    uint32_t subgroup_size = 0;
    uint32_t sr_workgroup_size = 64;
    uint32_t sr_atoms_per_group = 1;
    bool subgroup_sr = false;
    bool shader_float64 = false;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule count_shader = VK_NULL_HANDLE;
    VkShaderModule scan_shader = VK_NULL_HANDLE;
    VkShaderModule fill_shader = VK_NULL_HANDLE;
    VkShaderModule gather_shader = VK_NULL_HANDLE;
    VkShaderModule sr32_shader = VK_NULL_HANDLE;
    VkShaderModule sr64_shader = VK_NULL_HANDLE;
    VkShaderModule sr_sg_shader = VK_NULL_HANDLE;
    VkShaderModule lr_shader = VK_NULL_HANDLE;
    VkShaderModule lr_reduce_shader = VK_NULL_HANDLE;
    VkShaderModule count_fp64_shader = VK_NULL_HANDLE;
    VkShaderModule fill_fp64_shader = VK_NULL_HANDLE;
    VkShaderModule sr_fp64_32_shader = VK_NULL_HANDLE;
    VkShaderModule sr_fp64_64_shader = VK_NULL_HANDLE;
    VkShaderModule sr_fp64_sg_shader = VK_NULL_HANDLE;
    VkShaderModule lr_fp64_shader = VK_NULL_HANDLE;
    VkShaderModule lr_reduce_fp64_shader = VK_NULL_HANDLE;
    VkShaderModule sr_exposed_points_fp64_shader = VK_NULL_HANDLE;
    VkPipeline count_pipeline = VK_NULL_HANDLE;
    VkPipeline scan_pipeline = VK_NULL_HANDLE;
    VkPipeline fill_pipeline = VK_NULL_HANDLE;
    VkPipeline gather_pipeline = VK_NULL_HANDLE;
    VkPipeline sr32_pipeline = VK_NULL_HANDLE;
    VkPipeline sr64_pipeline = VK_NULL_HANDLE;
    VkPipeline sr_sg_pipeline = VK_NULL_HANDLE;
    VkPipeline lr_pipeline = VK_NULL_HANDLE;
    VkPipeline lr_reduce_pipeline = VK_NULL_HANDLE;
    VkPipeline count_fp64_pipeline = VK_NULL_HANDLE;
    VkPipeline fill_fp64_pipeline = VK_NULL_HANDLE;
    VkPipeline sr_fp64_32_pipeline = VK_NULL_HANDLE;
    VkPipeline sr_fp64_64_pipeline = VK_NULL_HANDLE;
    VkPipeline sr_fp64_sg_pipeline = VK_NULL_HANDLE;
    VkPipeline lr_fp64_pipeline = VK_NULL_HANDLE;
    VkPipeline lr_reduce_fp64_pipeline = VK_NULL_HANDLE;
    VkPipeline sr_exposed_points_fp64_pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    Buffer atoms;
    Buffer points;
    Buffer heads;       /* cell_offsets: cells + 1 exclusive prefix sums */
    Buffer next;        /* cell_atoms: atom indices grouped by cell */
    Buffer cell_counts; /* per-cell counts, then fill cursors */
    Buffer atoms_shadow_sorted; /* FP32 shadow in cell order */
    Buffer centers;
    Buffer areas;
    Buffer exposed_points;         /* atom_count * point_count uints */
    Buffer staging_atoms;
    Buffer staging_points;
    Buffer staging_centers;
    Buffer staging_areas;
    Buffer staging_exposed_points;
    Buffer atoms_shadow;
    Buffer staging_atoms_shadow;
    std::string error;
};

namespace {

void check_storage_range(const fastsasa_vk_context *context,
                         VkDeviceSize bytes,
                         const char *name)
{
    if (bytes > context->properties.limits.maxStorageBufferRange) {
        throw std::runtime_error(std::string(name) +
                                 " exceeds the Vulkan storage-buffer limit");
    }
}

/* Issues a 1D dispatch in chunks that respect maxComputeWorkGroupCount[0]
 * (spec minimum 65,535). Each chunk re-pushes the constants with the shader
 * index offset advanced by indices_per_group per submitted workgroup, so
 * arbitrarily large systems run on spec-minimum devices. */
template <typename Real>
void dispatch_chunked(fastsasa_vk_context *context,
                      ParametersT<Real> parameters,
                      uint64_t total_groups,
                      uint32_t indices_per_group)
{
    const uint64_t group_limit =
        context->properties.limits.maxComputeWorkGroupCount[0];
    uint64_t issued = 0;
    while (issued < total_groups) {
        const uint32_t groups = static_cast<uint32_t>(
            std::min<uint64_t>(total_groups - issued, group_limit));
        parameters.base_index =
            static_cast<uint32_t>(issued * indices_per_group);
        vkCmdPushConstants(context->command, context->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters),
                           &parameters);
        vkCmdDispatch(context->command, groups, 1, 1);
        issued += groups;
    }
}


/* Records the contiguous cell-list build (count, exclusive scan, fill) for
 * the atoms currently in the device buffer; the caller has already
 * bound the descriptor set. Ends with the grid visible to compute. */
template <typename Real>
void record_cell_list_build(fastsasa_vk_context *context,
                            const ParametersT<Real> &parameters,
                            uint32_t atom_count,
                            uint64_t cell_count,
                            bool use_fp64,
                            bool gather_shadow)
{
    VkMemoryBarrier compute_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    compute_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdFillBuffer(context->command, context->cell_counts.handle, 0,
                    sizeof(int32_t) * cell_count, 0u);
    VkMemoryBarrier transfer_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    transfer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                     VK_ACCESS_TRANSFER_READ_BIT;
    transfer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &transfer_barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      use_fp64 ? context->count_fp64_pipeline : context->count_pipeline);
    vkCmdBindDescriptorSets(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            context->pipeline_layout, 0, 1,
                            &context->descriptor_set, 0, nullptr);
    dispatch_chunked(context, parameters,
                     (atom_count + kBinThreads - 1u) / kBinThreads, kBinThreads);
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &compute_barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      context->scan_pipeline);
    dispatch_chunked(context, parameters, 1u, 1u);
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &compute_barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      use_fp64 ? context->fill_fp64_pipeline : context->fill_pipeline);
    dispatch_chunked(context, parameters,
                     (atom_count + kBinThreads - 1u) / kBinThreads, kBinThreads);
    vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &compute_barrier, 0, nullptr, 0, nullptr);
    if (gather_shadow) {
        vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          context->gather_pipeline);
        dispatch_chunked(context, parameters,
                         (atom_count + kBinThreads - 1u) / kBinThreads, kBinThreads);
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &compute_barrier, 0, nullptr, 0, nullptr);
    }
}

void destroy_buffer(fastsasa_vk_context *context, Buffer *buffer)
{
    if (buffer->mapped != nullptr) vkUnmapMemory(context->device, buffer->memory);
    if (buffer->handle != VK_NULL_HANDLE) vkDestroyBuffer(context->device, buffer->handle, nullptr);
    if (buffer->memory != VK_NULL_HANDLE) vkFreeMemory(context->device, buffer->memory, nullptr);
    *buffer = {};
}

void ensure_buffer(fastsasa_vk_context *context,
                   Buffer *buffer,
                   VkDeviceSize bytes,
                   VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags memory_flags)
{
    if (bytes == 0) bytes = 4;
    if (buffer->capacity >= bytes) return;
    destroy_buffer(context, buffer);

    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(context->device, &info, nullptr, &buffer->handle),
          "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context->device, buffer->handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = find_memory_type(context->physical,
                                                   requirements.memoryTypeBits,
                                                   memory_flags);
    check(vkAllocateMemory(context->device, &allocation, nullptr, &buffer->memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(context->device, buffer->handle, buffer->memory, 0),
          "vkBindBufferMemory");
    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check(vkMapMemory(context->device, buffer->memory, 0, VK_WHOLE_SIZE, 0,
                          &buffer->mapped),
              "vkMapMemory");
    }
    buffer->capacity = bytes;
}

VkShaderModule make_shader(fastsasa_vk_context *context,
                           const unsigned char *bytes,
                           size_t byte_count)
{
    if (bytes == nullptr || byte_count == 0 || byte_count % 4 != 0) {
        throw std::runtime_error("invalid embedded SPIR-V shader");
    }
    /* The xxd-generated byte arrays carry no alignment guarantee, while
     * VkShaderModuleCreateInfo::pCode must be 4-byte aligned. */
    std::vector<uint32_t> words(byte_count / 4);
    std::memcpy(words.data(), bytes, byte_count);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = byte_count;
    info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(context->device, &info, nullptr, &module),
          "vkCreateShaderModule");
    return module;
}

VkPipeline make_pipeline(fastsasa_vk_context *context, VkShaderModule shader)
{
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = stage;
    info.layout = context->pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(context->device, VK_NULL_HANDLE, 1, &info,
                                   nullptr, &pipeline),
          "vkCreateComputePipelines");
    return pipeline;
}

bool instance_extension_available(const char *name)
{
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) !=
            VK_SUCCESS || count == 0) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    const VkResult listed = vkEnumerateInstanceExtensionProperties(
        nullptr, &count, extensions.data());
    if (listed != VK_SUCCESS && listed != VK_INCOMPLETE) return false;
    for (uint32_t i = 0; i < count; ++i) {
        if (std::strcmp(extensions[i].extensionName, name) == 0) return true;
    }
    return false;
}

void initialize(fastsasa_vk_context *context, int device_index)
{
    if (volkInitialize() != VK_SUCCESS) {
        throw std::runtime_error(
            "the Vulkan loader (libvulkan.so.1) is not installed");
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "FastSASA Vulkan backend";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    /* Lets MoltenVK-style portability drivers enumerate when present. */
    const char *portability = "VK_KHR_portability_enumeration";
    if (instance_extension_available(portability)) {
        instance_info.flags |= 0x00000001; /* ..ENUMERATE_PORTABILITY_BIT_KHR */
        instance_info.enabledExtensionCount = 1;
        instance_info.ppEnabledExtensionNames = &portability;
    }
    check(vkCreateInstance(&instance_info, nullptr, &context->instance),
          "vkCreateInstance");
    volkLoadInstance(context->instance);

    context->physical = select_device(context->instance, device_index,
                                      &context->queue_family, &context->properties);
    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &subgroup;
    if (vkGetPhysicalDeviceProperties2 == nullptr) {
        throw std::runtime_error("the Vulkan loader does not expose Vulkan 1.1");
    }
    vkGetPhysicalDeviceProperties2(context->physical, &properties2);
    VkPhysicalDeviceFeatures available_features{};
    vkGetPhysicalDeviceFeatures(context->physical, &available_features);
    context->shader_float64 = available_features.shaderFloat64 == VK_TRUE;
    context->subgroup_size = subgroup.subgroupSize;
    context->sr_workgroup_size = 64u;
    {
        const VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                              VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        /* Software devices (llvmpipe) emulate subgroup operations; the
         * one-atom-per-workgroup shaders are far faster there. */
        context->subgroup_sr = context->properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU &&
                               (subgroup.supportedOperations & needed) == needed &&
                               (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                               subgroup.subgroupSize >= 8u && subgroup.subgroupSize <= 256u &&
                               256u % subgroup.subgroupSize == 0u &&
                               context->properties.limits.maxComputeWorkGroupInvocations >= 256u &&
                               context->properties.limits.maxComputeWorkGroupSize[0] >= 256u;
    }
    if (context->properties.limits.maxPushConstantsSize < sizeof(ParametersT<double>) ||
        context->properties.limits.maxComputeWorkGroupInvocations < kBinThreads ||
        context->properties.limits.maxComputeWorkGroupSize[0] < kBinThreads) {
        throw std::runtime_error("Vulkan compute limits are below FastSASA requirements");
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = context->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderFloat64 = context->shader_float64 ? VK_TRUE : VK_FALSE;
    device_info.pEnabledFeatures = &enabled_features;
    check(vkCreateDevice(context->physical, &device_info, nullptr, &context->device),
          "vkCreateDevice");
    vkGetDeviceQueue(context->device, context->queue_family, 0, &context->queue);

    VkDescriptorSetLayoutBinding bindings[9]{};
    for (uint32_t i = 0; i < 9; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 9;
    descriptor_info.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(context->device, &descriptor_info, nullptr,
                                      &context->descriptor_layout),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                              (uint32_t)sizeof(ParametersT<double>)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &context->descriptor_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(context->device, &layout_info, nullptr,
                                 &context->pipeline_layout),
          "vkCreatePipelineLayout");

    context->count_shader = make_shader(context, fastsasa_vk_cell_count_spv,
                                        fastsasa_vk_cell_count_spv_len);
    context->scan_shader = make_shader(context, fastsasa_vk_cell_scan_spv,
                                       fastsasa_vk_cell_scan_spv_len);
    context->fill_shader = make_shader(context, fastsasa_vk_cell_fill_spv,
                                       fastsasa_vk_cell_fill_spv_len);
    context->gather_shader = make_shader(context, fastsasa_vk_cell_gather_spv,
                                         fastsasa_vk_cell_gather_spv_len);
    context->sr32_shader = make_shader(context, fastsasa_vk_sr_cell_list_32_spv,
                                       fastsasa_vk_sr_cell_list_32_spv_len);
    context->sr64_shader = make_shader(context, fastsasa_vk_sr_cell_list_64_spv,
                                       fastsasa_vk_sr_cell_list_64_spv_len);
    if (context->subgroup_sr) {
        context->sr_sg_shader = make_shader(context, fastsasa_vk_sr_cell_list_sg_spv,
                                            fastsasa_vk_sr_cell_list_sg_spv_len);
    }
    context->lr_shader = make_shader(context, fastsasa_vk_lee_richards_cell_spv,
                                     fastsasa_vk_lee_richards_cell_spv_len);
    context->lr_reduce_shader = make_shader(context, fastsasa_vk_lee_richards_reduce_spv,
                                            fastsasa_vk_lee_richards_reduce_spv_len);
    context->count_pipeline = make_pipeline(context, context->count_shader);
    context->scan_pipeline = make_pipeline(context, context->scan_shader);
    context->fill_pipeline = make_pipeline(context, context->fill_shader);
    context->gather_pipeline = make_pipeline(context, context->gather_shader);
    context->sr32_pipeline = make_pipeline(context, context->sr32_shader);
    context->sr64_pipeline = make_pipeline(context, context->sr64_shader);
    if (context->subgroup_sr) context->sr_sg_pipeline = make_pipeline(context, context->sr_sg_shader);
    context->lr_pipeline = make_pipeline(context, context->lr_shader);
    context->lr_reduce_pipeline = make_pipeline(context, context->lr_reduce_shader);
    if (context->shader_float64) {
        context->count_fp64_shader = make_shader(
            context, fastsasa_vk_cell_count_fp64_spv,
            fastsasa_vk_cell_count_fp64_spv_len);
        context->fill_fp64_shader = make_shader(
            context, fastsasa_vk_cell_fill_fp64_spv,
            fastsasa_vk_cell_fill_fp64_spv_len);
        context->sr_fp64_32_shader = make_shader(
            context, fastsasa_vk_sr_cell_list_fp64_32_spv,
            fastsasa_vk_sr_cell_list_fp64_32_spv_len);
        context->sr_fp64_64_shader = make_shader(
            context, fastsasa_vk_sr_cell_list_fp64_64_spv,
            fastsasa_vk_sr_cell_list_fp64_64_spv_len);
        if (context->subgroup_sr) {
            context->sr_fp64_sg_shader = make_shader(
                context, fastsasa_vk_sr_cell_list_fp64_sg_spv,
                fastsasa_vk_sr_cell_list_fp64_sg_spv_len);
        }
        context->lr_fp64_shader = make_shader(
            context, fastsasa_vk_lee_richards_cell_fp64_spv,
            fastsasa_vk_lee_richards_cell_fp64_spv_len);
        context->lr_reduce_fp64_shader = make_shader(
            context, fastsasa_vk_lee_richards_reduce_fp64_spv,
            fastsasa_vk_lee_richards_reduce_fp64_spv_len);
        context->count_fp64_pipeline = make_pipeline(context, context->count_fp64_shader);
        context->fill_fp64_pipeline = make_pipeline(context, context->fill_fp64_shader);
        context->sr_fp64_32_pipeline = make_pipeline(context, context->sr_fp64_32_shader);
        context->sr_fp64_64_pipeline = make_pipeline(context, context->sr_fp64_64_shader);
        if (context->subgroup_sr) context->sr_fp64_sg_pipeline = make_pipeline(context, context->sr_fp64_sg_shader);
        context->lr_fp64_pipeline = make_pipeline(context, context->lr_fp64_shader);
        context->lr_reduce_fp64_pipeline = make_pipeline(
            context, context->lr_reduce_fp64_shader);
        context->sr_exposed_points_fp64_shader = make_shader(
            context, fastsasa_vk_sr_exposed_points_fp64_64_spv,
            fastsasa_vk_sr_exposed_points_fp64_64_spv_len);
        context->sr_exposed_points_fp64_pipeline = make_pipeline(
            context, context->sr_exposed_points_fp64_shader);
    }

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(context->device, &pool_info, nullptr,
                                 &context->descriptor_pool),
          "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = context->descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &context->descriptor_layout;
    check(vkAllocateDescriptorSets(context->device, &set_info,
                                   &context->descriptor_set),
          "vkAllocateDescriptorSets");

    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = context->queue_family;
    check(vkCreateCommandPool(context->device, &command_pool_info, nullptr,
                              &context->command_pool),
          "vkCreateCommandPool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = context->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(context->device, &command_info, &context->command),
          "vkAllocateCommandBuffers");
}

void update_descriptors(fastsasa_vk_context *context)
{
    Buffer *buffers[9] = {&context->atoms, &context->points, &context->heads,
                          &context->next, &context->centers, &context->areas,
                          &context->atoms_shadow, &context->cell_counts,
                          &context->atoms_shadow_sorted};
    VkDescriptorBufferInfo infos[9]{};
    VkWriteDescriptorSet writes[9]{};
    for (uint32_t i = 0; i < 9; ++i) {
        infos[i] = {buffers[i]->handle, 0, buffers[i]->capacity};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = context->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(context->device, 9, writes, 0, nullptr);
}

VkPipeline select_sr_pipeline(fastsasa_vk_context *context,
                              uint32_t point_count,
                              bool use_fp64)
{
    context->sr_workgroup_size =
        context->subgroup_size > 0 && context->subgroup_size <= 32 && point_count <= 32
            ? 32u
            : 64u;
    /* One atom per subgroup (256-lane workgroups) is the default where the
     * device supports subgroup arithmetic: atoms retire independently, which
     * is measurably better on dense systems and never worse in our tests.
     * FASTSASA_VK_SR_LANES=64|32 selects the one-atom-per-workgroup shaders. */
    bool subgroup_mode = context->subgroup_sr;
    if (const char *lanes = std::getenv("FASTSASA_VK_SR_LANES")) {
        subgroup_mode = false;
        if (std::strcmp(lanes, "32") == 0 && context->subgroup_size <= 32) context->sr_workgroup_size = 32u;
        if (std::strcmp(lanes, "64") == 0) context->sr_workgroup_size = 64u;
        if (std::strcmp(lanes, "sg") == 0 && context->subgroup_sr) subgroup_mode = true;
    }
    context->sr_atoms_per_group = subgroup_mode ? 256u / context->subgroup_size : 1u;
    if (subgroup_mode) {
        return use_fp64 ? context->sr_fp64_sg_pipeline : context->sr_sg_pipeline;
    }
    if (use_fp64) {
        return context->sr_workgroup_size == 32
                   ? context->sr_fp64_32_pipeline
                   : context->sr_fp64_64_pipeline;
    }
    return context->sr_workgroup_size == 32 ? context->sr32_pipeline
                                            : context->sr64_pipeline;
}

/*
 * FP64 Lee-Richards slices that need more arcs than the shader's local
 * buffer are written as NaN; those atoms are recomputed here with the CPU
 * reference itself, so the Vulkan result stays bit-identical to it.
 */
static void
recompute_nan_lee_richards(const double *xyz,
                           const double *radii,
                           uint32_t atom_count,
                           uint32_t slice_count,
                           double probe_radius,
                           const uint32_t *centers,
                           uint32_t center_count,
                           double *sasa)
{
    std::vector<double> x, y, z, expanded;

    for (uint32_t center = 0; center < center_count; ++center) {
        double area = 0.0;

        if (!std::isnan(sasa[center])) continue;
        if (x.empty()) {
            x.resize(atom_count);
            y.resize(atom_count);
            z.resize(atom_count);
            expanded.resize(atom_count);
            for (uint32_t atom = 0; atom < atom_count; ++atom) {
                x[atom] = xyz[3u * atom];
                y[atom] = xyz[3u * atom + 1u];
                z[atom] = xyz[3u * atom + 2u];
                expanded[atom] = radii[atom] + probe_radius;
            }
        }
        if (fastsasa_cpu_lee_richards_atom(static_cast<int>(atom_count),
                                         static_cast<int>(slice_count),
                                         x.data(), y.data(), z.data(),
                                         expanded.data(),
                                         static_cast<int>(centers[center]),
                                         &area) != 0) {
            throw std::runtime_error("host Lee-Richards recompute failed");
        }
        sasa[center] = area;
    }
}

template <typename Real>
int calculate(fastsasa_vk_context *context,
              const double *xyz,
              const double *radii,
              uint32_t atom_count,
              const double *sphere_xyz,
              uint32_t point_count,
              double probe_radius,
              const uint32_t *requested_centers,
              uint32_t center_count,
              double *sasa,
              bool lee_richards)
{
    if (context == nullptr || xyz == nullptr || radii == nullptr ||
        (!lee_richards && sphere_xyz == nullptr) || sasa == nullptr || atom_count == 0 ||
        point_count == 0 || center_count == 0 || !std::isfinite(probe_radius) ||
        probe_radius < 0.0) {
        if (context != nullptr) context->error = "invalid Shrake-Rupley arguments";
        return 1;
    }

    try {
        const bool use_fp64 = std::is_same<Real, double>::value;
        /* The FP32 shadow serves the FP64 prefilter and is the FP32 path's
         * neighbour record, so it is built for every SR run. */
        const bool sr_hybrid = !lee_richards;
        std::vector<Vec4T<Real>> atoms(atom_count);
        std::vector<Vec4T<float>> shadow(sr_hybrid ? atom_count : 1u);
        std::vector<Vec4T<Real>> points(lee_richards ? 1u : point_count);
        std::vector<uint32_t> centers(center_count);
        Real min_x = std::numeric_limits<Real>::infinity();
        Real min_y = min_x;
        Real min_z = min_x;
        Real max_x = -min_x;
        Real max_y = -min_x;
        Real max_z = -min_x;
        Real max_expanded_radius = Real(0);
        for (uint32_t i = 0; i < atom_count; ++i) {
            const Real x = static_cast<Real>(xyz[3u * i]);
            const Real y = static_cast<Real>(xyz[3u * i + 1u]);
            const Real z = static_cast<Real>(xyz[3u * i + 2u]);
            const Real radius = static_cast<Real>(radii[i]);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                !std::isfinite(radius) || radius <= 0.0f) {
                throw std::runtime_error("coordinates and radii must be finite; radii must be positive");
            }
            atoms[i] = {x, y, z, radius};
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            min_z = std::min(min_z, z);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            max_z = std::max(max_z, z);
            max_expanded_radius = std::max(
                max_expanded_radius, radius + static_cast<Real>(probe_radius));
        }
        if (sr_hybrid) {
            /* The FP32 shadow is grid-local (the FP64 coordinates are not
             * shifted), which keeps its magnitudes inside the extent the
             * prefilter margin is derived from. */
            for (uint32_t i = 0; i < atom_count; ++i) {
                const double expanded = static_cast<double>(atoms[i].w) + probe_radius;
                shadow[i] = {(float)(atoms[i].x - min_x), (float)(atoms[i].y - min_y),
                             (float)(atoms[i].z - min_z), (float)(expanded * expanded)};
            }
        }
        if (lee_richards) {
            points[0] = {};
        } else {
            for (uint32_t i = 0; i < point_count; ++i) {
                points[i] = {static_cast<Real>(sphere_xyz[3u * i]),
                             static_cast<Real>(sphere_xyz[3u * i + 1u]),
                             static_cast<Real>(sphere_xyz[3u * i + 2u]), Real(0)};
            }
        }
        for (uint32_t i = 0; i < center_count; ++i) {
            const uint32_t center = requested_centers == nullptr ? i : requested_centers[i];
            if (center >= atom_count) throw std::runtime_error("center index is outside atom array");
            centers[i] = center;
        }

        const Real cell_size = Real(2) * max_expanded_radius;
        auto dimension = [cell_size](Real minimum, Real maximum) {
            const double cells = std::floor((static_cast<double>(maximum) - minimum) /
                                            cell_size) + 1.0;
            if (cells < 1.0 || cells > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("invalid cell-grid dimension");
            }
            return static_cast<uint32_t>(cells);
        };
        const uint32_t dim_x = dimension(min_x, max_x);
        const uint32_t dim_y = dimension(min_y, max_y);
        const uint32_t dim_z = dimension(min_z, max_z);
        const uint64_t cell_count = static_cast<uint64_t>(dim_x) * dim_y * dim_z;
        if (cell_count == 0 || cell_count > kMaxCells) {
            throw std::runtime_error("cell grid is too large; unwrap or filter sparse coordinates");
        }

        const VkDeviceSize atom_bytes = sizeof(Vec4T<Real>) * (VkDeviceSize)atom_count;
        const VkDeviceSize point_storage_bytes = lee_richards
            ? sizeof(Real) * static_cast<uint64_t>(center_count) * point_count
            : sizeof(Vec4T<Real>) * points.size();
        const VkDeviceSize head_bytes = sizeof(int32_t) * cell_count;
        const VkDeviceSize next_bytes = sizeof(int32_t) * (VkDeviceSize)atom_count;
        const VkDeviceSize center_bytes = sizeof(uint32_t) * (VkDeviceSize)center_count;
        const VkDeviceSize area_bytes = sizeof(Real) * (VkDeviceSize)center_count;
        check_storage_range(context, atom_bytes, "atom buffer");
        check_storage_range(context, point_storage_bytes, "point buffer");
        check_storage_range(context, head_bytes, "cell-head buffer");
        check_storage_range(context, next_bytes, "cell-link buffer");
        check_storage_range(context, center_bytes, "center buffer");
        check_storage_range(context, area_bytes, "area buffer");

        const VkMemoryPropertyFlags device_memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkMemoryPropertyFlags host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ensure_buffer(context, &context->atoms, sizeof(Vec4T<Real>) * atom_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->points, point_storage_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->cell_counts, sizeof(int32_t) * cell_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->heads, sizeof(int32_t) * (cell_count + 1u),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->next, sizeof(int32_t) * atom_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        ensure_buffer(context, &context->centers, sizeof(uint32_t) * center_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->areas, sizeof(Real) * center_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_memory);
        ensure_buffer(context, &context->staging_atoms, sizeof(Vec4T<Real>) * atom_count,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));
        ensure_buffer(context, &context->staging_points, sizeof(Vec4T<Real>) * points.size(),
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_centers, sizeof(uint32_t) * center_count,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_areas, sizeof(Real) * center_count,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      readback_memory_flags(context->physical));
        ensure_buffer(context, &context->atoms_shadow,
                      sr_hybrid ? sizeof(Vec4T<float>) * atom_count : 4u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->atoms_shadow_sorted,
                      sr_hybrid ? sizeof(Vec4T<float>) * atom_count : 4u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        ensure_buffer(context, &context->staging_atoms_shadow,
                      sr_hybrid ? sizeof(Vec4T<float>) * atom_count : 4u,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));
        if (sr_hybrid) {
            std::memcpy(context->staging_atoms_shadow.mapped, shadow.data(),
                        sizeof(Vec4T<float>) * atom_count);
        }
        std::memcpy(context->staging_atoms.mapped, atoms.data(),
                    sizeof(Vec4T<Real>) * atom_count);
        std::memcpy(context->staging_points.mapped, points.data(),
                    sizeof(Vec4T<Real>) * points.size());
        std::memcpy(context->staging_centers.mapped, centers.data(),
                    sizeof(uint32_t) * center_count);
        update_descriptors(context);

        const ParametersT<Real> parameters{
            atom_count, point_count, center_count, 0u, dim_x, dim_y, dim_z,
            static_cast<Real>(probe_radius), cell_size, min_x, min_y, min_z,
            static_cast<Real>(sr_hybrid
                ? sr_hybrid_margin(static_cast<double>(cell_size),
                                   dim_x, dim_y, dim_z)
                : 0.0)};
        check(vkResetCommandBuffer(context->command, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(context->command, &begin), "vkBeginCommandBuffer");

        VkBufferCopy atom_copy{0, 0, sizeof(Vec4T<Real>) * atom_count};
        VkBufferCopy point_copy{0, 0, sizeof(Vec4T<Real>) * points.size()};
        VkBufferCopy center_copy{0, 0, sizeof(uint32_t) * center_count};
        vkCmdCopyBuffer(context->command, context->staging_atoms.handle,
                        context->atoms.handle, 1, &atom_copy);
        if (sr_hybrid) {
            VkBufferCopy shadow_copy{0, 0, sizeof(Vec4T<float>) * atom_count};
            vkCmdCopyBuffer(context->command, context->staging_atoms_shadow.handle,
                            context->atoms_shadow.handle, 1, &shadow_copy);
        }
        if (!lee_richards) {
            /* In Lee-Richards mode the points binding is slice scratch and can
             * be smaller than one Vec4; copying the dummy point would read and
             * write out of bounds. */
            vkCmdCopyBuffer(context->command, context->staging_points.handle,
                            context->points.handle, 1, &point_copy);
        }
        vkCmdCopyBuffer(context->command, context->staging_centers.handle,
                        context->centers.handle, 1, &center_copy);
        record_cell_list_build(context, parameters, atom_count, cell_count, use_fp64, sr_hybrid);

        if (lee_richards) {
            vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              use_fp64 ? context->lr_fp64_pipeline
                                       : context->lr_pipeline);
            const uint64_t slices = static_cast<uint64_t>(center_count) * point_count;
            dispatch_chunked(context, parameters,
                             (slices + kSrThreads - 1u) / kSrThreads, kSrThreads);
            VkMemoryBarrier slice_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            slice_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            slice_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &slice_barrier, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              use_fp64 ? context->lr_reduce_fp64_pipeline
                                       : context->lr_reduce_pipeline);
            dispatch_chunked(context, parameters,
                             (center_count + kSrThreads - 1u) / kSrThreads,
                             kSrThreads);
        } else {
            vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              select_sr_pipeline(context, point_count, use_fp64));
            /* One workgroup per center; base_index advances per workgroup. */
            dispatch_chunked(context, parameters,
                             (center_count + context->sr_atoms_per_group - 1u) /
                                 context->sr_atoms_per_group,
                             context->sr_atoms_per_group);
        }

        VkMemoryBarrier output_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &output_barrier,
                             0, nullptr, 0, nullptr);
        VkBufferCopy area_copy{0, 0, sizeof(Real) * center_count};
        vkCmdCopyBuffer(context->command, context->areas.handle,
                        context->staging_areas.handle, 1, &area_copy);
        VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier,
                             0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(context->command), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &context->command;
        check(vkQueueSubmit(context->queue, 1, &submit, VK_NULL_HANDLE),
              "vkQueueSubmit");
        check(vkQueueWaitIdle(context->queue), "vkQueueWaitIdle");
        const Real *areas = static_cast<const Real *>(context->staging_areas.mapped);
        for (uint32_t center = 0; center < center_count; ++center) {
            sasa[center] = lee_richards
                ? static_cast<double>(areas[center])
                : sr_count_to_area(static_cast<double>(areas[center]),
                                   radii[centers[center]], probe_radius,
                                   point_count);
        }
        if (lee_richards && use_fp64) {
            recompute_nan_lee_richards(xyz, radii, atom_count, point_count, probe_radius,
                                       centers.data(), center_count, sasa);
        }
        context->error.clear();
        return 0;
    } catch (const std::exception &error) {
        context->error = error.what();
        return 1;
    }
}

template <typename Real>
int calculate_frames(fastsasa_vk_context *context,
                     const double *frame_xyz,
                     const double *radii,
                     uint32_t frame_count,
                     uint32_t atom_count,
                     const double *sphere_xyz,
                     uint32_t resolution,
                     double probe_radius,
                     const uint32_t *requested_centers,
                     uint32_t center_count,
                     double *frame_sasa,
                     bool lee_richards)
{
    if (context == nullptr || frame_xyz == nullptr || radii == nullptr ||
        frame_sasa == nullptr || frame_count == 0 || atom_count == 0 ||
        center_count == 0 || resolution == 0 ||
        !std::isfinite(probe_radius) || probe_radius < 0.0 ||
        (!lee_richards && sphere_xyz == nullptr)) {
        if (context != nullptr) context->error = "invalid frame arguments";
        return 1;
    }
    try {
        const auto profile_t0 = std::chrono::steady_clock::now();
        const bool use_fp64 = std::is_same<Real, double>::value;
        /* The FP32 shadow serves the FP64 prefilter and is the FP32 path's
         * neighbour record, so it is built for every SR run. */
        const bool sr_hybrid = !lee_richards;
        /* Atom records and the FP32 shadow are written straight into the
         * (host-cached) staging buffers: per-batch temporaries of
         * frames x atoms records cost more in allocation and page faults
         * than the GPU work on large systems. */
        const size_t record_count = static_cast<size_t>(frame_count) * atom_count;
        const size_t shadow_count = sr_hybrid ? record_count : 1u;
        ensure_buffer(context, &context->staging_atoms,
                      sizeof(Vec4T<Real>) * record_count,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));
        ensure_buffer(context, &context->staging_atoms_shadow,
                      sr_hybrid ? sizeof(Vec4T<float>) * shadow_count : 4u,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));
        Vec4T<Real> *const atoms = static_cast<Vec4T<Real> *>(context->staging_atoms.mapped);
        Vec4T<float> *const shadow = static_cast<Vec4T<float> *>(context->staging_atoms_shadow.mapped);
        std::vector<Vec4T<Real>> points(lee_richards ? 1u : resolution);
        std::vector<uint32_t> centers(center_count);
        std::vector<ParametersT<Real>> parameters(frame_count);
        uint64_t max_cell_count = 0;
        Real max_expanded_radius = Real(0);
        for (uint32_t atom = 0; atom < atom_count; ++atom) {
            if (!std::isfinite(radii[atom]) || radii[atom] <= 0.0f) {
                throw std::runtime_error("radii must be finite and positive");
            }
            max_expanded_radius = std::max(
                max_expanded_radius,
                static_cast<Real>(radii[atom] + probe_radius));
        }
        const Real cell_size = Real(2) * max_expanded_radius;
        for (uint32_t frame = 0; frame < frame_count; ++frame) {
            Real min_x = std::numeric_limits<Real>::infinity();
            Real min_y = min_x;
            Real min_z = min_x;
            Real max_x = -min_x;
            Real max_y = -min_x;
            Real max_z = -min_x;
            for (uint32_t atom = 0; atom < atom_count; ++atom) {
                const size_t input = 3ull * (static_cast<size_t>(frame) * atom_count + atom);
                const Real x = static_cast<Real>(frame_xyz[input]);
                const Real y = static_cast<Real>(frame_xyz[input + 1u]);
                const Real z = static_cast<Real>(frame_xyz[input + 2u]);
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    throw std::runtime_error("frame coordinates must be finite");
                }
                atoms[static_cast<size_t>(frame) * atom_count + atom] =
                    {x, y, z, static_cast<Real>(radii[atom])};
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                min_z = std::min(min_z, z);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
                max_z = std::max(max_z, z);
            }
            if (sr_hybrid) {
                /* Grid-local FP32 shadow; see calculate(). */
                for (uint32_t atom = 0; atom < atom_count; ++atom) {
                    const size_t slot = static_cast<size_t>(frame) * atom_count + atom;
                    const double expanded = radii[atom] + probe_radius;
                    shadow[slot] = {(float)(atoms[slot].x - min_x), (float)(atoms[slot].y - min_y),
                                    (float)(atoms[slot].z - min_z), (float)(expanded * expanded)};
                }
            }
            auto dimension = [cell_size](Real minimum, Real maximum) {
                const double value = std::floor((static_cast<double>(maximum) - minimum) /
                                                cell_size) + 1.0;
                if (value < 1.0 || value > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("invalid frame cell-grid dimension");
                }
                return static_cast<uint32_t>(value);
            };
            const uint32_t dim_x = dimension(min_x, max_x);
            const uint32_t dim_y = dimension(min_y, max_y);
            const uint32_t dim_z = dimension(min_z, max_z);
            const uint64_t cells = static_cast<uint64_t>(dim_x) * dim_y * dim_z;
            if (cells == 0 || cells > kMaxCells) {
                throw std::runtime_error("frame cell grid is too large");
            }
            max_cell_count = std::max(max_cell_count, cells);
            parameters[frame] = {atom_count, resolution, center_count, 0u,
                                 dim_x, dim_y, dim_z,
                                 static_cast<Real>(probe_radius), cell_size,
                                 min_x, min_y, min_z,
                                 static_cast<Real>(sr_hybrid
                                     ? sr_hybrid_margin(
                                           static_cast<double>(cell_size),
                                           dim_x, dim_y, dim_z)
                                     : 0.0)};
        }
        if (lee_richards) {
            points[0] = {};
        } else {
            for (uint32_t point = 0; point < resolution; ++point) {
                points[point] = {
                    static_cast<Real>(sphere_xyz[3u * point]),
                    static_cast<Real>(sphere_xyz[3u * point + 1u]),
                    static_cast<Real>(sphere_xyz[3u * point + 2u]), Real(0)};
            }
        }
        for (uint32_t center = 0; center < center_count; ++center) {
            centers[center] = requested_centers == nullptr ? center : requested_centers[center];
            if (centers[center] >= atom_count) {
                throw std::runtime_error("center index is outside atom array");
            }
        }

        const VkDeviceSize atom_bytes = sizeof(Vec4T<Real>) * (VkDeviceSize)atom_count;
        const VkDeviceSize point_storage_bytes = lee_richards
            ? sizeof(Real) * static_cast<uint64_t>(center_count) * resolution
            : sizeof(Vec4T<Real>) * points.size();
        const VkDeviceSize head_bytes = sizeof(int32_t) * max_cell_count;
        const VkDeviceSize next_bytes = sizeof(int32_t) * (VkDeviceSize)atom_count;
        const VkDeviceSize center_bytes = sizeof(uint32_t) * (VkDeviceSize)center_count;
        const VkDeviceSize area_bytes = sizeof(Real) * (VkDeviceSize)center_count;
        check_storage_range(context, atom_bytes, "atom buffer");
        check_storage_range(context, point_storage_bytes, "point buffer");
        check_storage_range(context, head_bytes, "cell-head buffer");
        check_storage_range(context, next_bytes, "cell-link buffer");
        check_storage_range(context, center_bytes, "center buffer");
        check_storage_range(context, area_bytes, "area buffer");

        const VkMemoryPropertyFlags device_memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkMemoryPropertyFlags host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ensure_buffer(context, &context->atoms, sizeof(Vec4T<Real>) * atom_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->points, point_storage_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->cell_counts, sizeof(int32_t) * max_cell_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->heads, sizeof(int32_t) * (max_cell_count + 1u),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->next, sizeof(int32_t) * atom_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        ensure_buffer(context, &context->centers, sizeof(uint32_t) * center_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->areas, sizeof(Real) * center_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      device_memory);
        ensure_buffer(context, &context->staging_points,
                      sizeof(Vec4T<Real>) * points.size(),
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_centers, sizeof(uint32_t) * center_count,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_areas,
                      sizeof(Real) * static_cast<uint64_t>(frame_count) * center_count,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      readback_memory_flags(context->physical));
        ensure_buffer(context, &context->atoms_shadow,
                      sr_hybrid ? sizeof(Vec4T<float>) * atom_count : 4u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->atoms_shadow_sorted,
                      sr_hybrid ? sizeof(Vec4T<float>) * atom_count : 4u,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        std::memcpy(context->staging_points.mapped, points.data(),
                    sizeof(Vec4T<Real>) * points.size());
        std::memcpy(context->staging_centers.mapped, centers.data(),
                    sizeof(uint32_t) * center_count);
        update_descriptors(context);

        const auto profile_t1 = std::chrono::steady_clock::now();
        check(vkResetCommandBuffer(context->command, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(context->command, &begin), "vkBeginCommandBuffer");
        VkBufferCopy point_copy{0, 0, sizeof(Vec4T<Real>) * points.size()};
        VkBufferCopy center_copy{0, 0, sizeof(uint32_t) * center_count};
        if (!lee_richards) {
            /* See calculate(): the LR points binding is slice scratch and can
             * be smaller than one Vec4. */
            vkCmdCopyBuffer(context->command, context->staging_points.handle,
                            context->points.handle, 1, &point_copy);
        }
        vkCmdCopyBuffer(context->command, context->staging_centers.handle,
                        context->centers.handle, 1, &center_copy);

        for (uint32_t frame = 0; frame < frame_count; ++frame) {
            const uint64_t cell_count = static_cast<uint64_t>(parameters[frame].dim_x) *
                                        parameters[frame].dim_y * parameters[frame].dim_z;
            VkBufferCopy atom_copy{
                sizeof(Vec4T<Real>) * static_cast<uint64_t>(frame) * atom_count,
                0, sizeof(Vec4T<Real>) * atom_count};
            vkCmdCopyBuffer(context->command, context->staging_atoms.handle,
                            context->atoms.handle, 1, &atom_copy);
            if (sr_hybrid) {
                VkBufferCopy shadow_copy{
                    sizeof(Vec4T<float>) * static_cast<uint64_t>(frame) * atom_count,
                    0, sizeof(Vec4T<float>) * atom_count};
                vkCmdCopyBuffer(context->command,
                                context->staging_atoms_shadow.handle,
                                context->atoms_shadow.handle, 1, &shadow_copy);
            }
            record_cell_list_build(context, parameters[frame], atom_count, cell_count, use_fp64, sr_hybrid);
            if (lee_richards) {
                vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  use_fp64 ? context->lr_fp64_pipeline
                                           : context->lr_pipeline);
                const uint64_t slices = static_cast<uint64_t>(center_count) * resolution;
                dispatch_chunked(context, parameters[frame],
                                 (slices + kSrThreads - 1u) / kSrThreads,
                                 kSrThreads);
                VkMemoryBarrier slice_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                slice_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                slice_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(context->command,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                     &slice_barrier, 0, nullptr, 0, nullptr);
                vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  use_fp64 ? context->lr_reduce_fp64_pipeline
                                           : context->lr_reduce_pipeline);
                dispatch_chunked(context, parameters[frame],
                                 (center_count + kSrThreads - 1u) / kSrThreads,
                                 kSrThreads);
            } else {
                vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  select_sr_pipeline(context, resolution, use_fp64));
                dispatch_chunked(context, parameters[frame],
                                 (center_count + context->sr_atoms_per_group - 1u) /
                                     context->sr_atoms_per_group,
                                 context->sr_atoms_per_group);
            }
            VkMemoryBarrier compute_to_transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            compute_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            compute_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                 &compute_to_transfer, 0, nullptr, 0, nullptr);
            VkBufferCopy area_copy{0,
                sizeof(Real) * static_cast<uint64_t>(frame) * center_count,
                sizeof(Real) * center_count};
            vkCmdCopyBuffer(context->command, context->areas.handle,
                            context->staging_areas.handle, 1, &area_copy);
        }
        VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier,
                             0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(context->command), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &context->command;
        check(vkQueueSubmit(context->queue, 1, &submit, VK_NULL_HANDLE),
              "vkQueueSubmit");
        check(vkQueueWaitIdle(context->queue), "vkQueueWaitIdle");
        const auto profile_t2 = std::chrono::steady_clock::now();
        const Real *areas = static_cast<const Real *>(context->staging_areas.mapped);
        const size_t result_count = static_cast<size_t>(frame_count) * center_count;
        for (size_t result = 0; result < result_count; ++result) {
            frame_sasa[result] = lee_richards
                ? static_cast<double>(areas[result])
                : sr_count_to_area(static_cast<double>(areas[result]),
                                   radii[centers[result % center_count]],
                                   probe_radius, resolution);
        }
        if (lee_richards && use_fp64) {
            for (uint32_t frame = 0; frame < frame_count; ++frame) {
                recompute_nan_lee_richards(
                    frame_xyz + 3ull * static_cast<size_t>(frame) * atom_count,
                    radii, atom_count, resolution, probe_radius,
                    centers.data(), center_count,
                    frame_sasa + static_cast<size_t>(frame) * center_count);
            }
        }
        if (std::getenv("FASTSASA_VK_PROFILE") != nullptr) {
            const auto profile_t3 = std::chrono::steady_clock::now();
            const auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr,
                         "vk frames=%u atoms=%u host_prep %.3f ms, record+gpu %.3f ms, readback %.3f ms\n",
                         frame_count, atom_count, ms(profile_t0, profile_t1),
                         ms(profile_t1, profile_t2), ms(profile_t2, profile_t3));
        }
        context->error.clear();
        return 0;
    } catch (const std::exception &error) {
        context->error = error.what();
        return 1;
    }
}

/* Same 9 bindings as update_descriptors(), except binding 5 points at
 * exposed_points/staging_exposed_points instead of areas/staging_areas.
 * Kept as its own function (not a parameter added to update_descriptors())
 * so the SASA path's binding wiring is untouched by this addition. */
void update_descriptors_exposed_points(fastsasa_vk_context *context)
{
    Buffer *buffers[9] = {&context->atoms, &context->points, &context->heads,
                          &context->next, &context->centers, &context->exposed_points,
                          &context->atoms_shadow, &context->cell_counts,
                          &context->atoms_shadow_sorted};
    VkDescriptorBufferInfo infos[9]{};
    VkWriteDescriptorSet writes[9]{};
    for (uint32_t i = 0; i < 9; ++i) {
        infos[i] = {buffers[i]->handle, 0, buffers[i]->capacity};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = context->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(context->device, 9, writes, 0, nullptr);
}

/*
 * FP64-only host dispatch for VMD-style surface-point export. Every atom is
 * its own centre - there is no compact-centre concept for this path, every
 * calc atom's surface points are wanted - and the output is the full
 * per-point exposed/buried mask (matching fastsasa_cpu_exposed_points()'s
 * layout exactly: exposed[atom * point_count + point]), not a reduced area.
 * That is why this does not reuse calculate<Real>(), which is coupled to the
 * SASA-area reduction: it reuses the same cell-list-build and buffer-growth
 * helpers calculate<Real>() uses, so the two paths cannot silently drift
 * apart on cell-grid construction, but owns its own dispatch and buffers.
 */
int calculate_exposed_points(fastsasa_vk_context *context,
                             const double *xyz,
                             const double *x,
                             const double *y,
                             const double *z,
                             const double *radii,
                             uint32_t atom_count,
                             const double *sphere_xyz,
                             uint32_t point_count,
                             double probe_radius,
                             unsigned char *exposed)
{
    if (context == nullptr || radii == nullptr ||
        (xyz == nullptr && (x == nullptr || y == nullptr || z == nullptr)) ||
        sphere_xyz == nullptr || exposed == nullptr || atom_count == 0 ||
        point_count == 0 || !std::isfinite(probe_radius) || probe_radius < 0.0) {
        if (context != nullptr) context->error = "invalid exposed-point arguments";
        return 1;
    }
    if (!fastsasa_vk_supports_fp64(context)) {
        context->error = "Vulkan device does not support shaderFloat64; surface-point "
                         "export falls back to the CPU backend";
        return 1;
    }

    try {
        std::vector<Vec4T<double>> atoms(atom_count);
        std::vector<Vec4T<float>> shadow(atom_count);
        std::vector<Vec4T<double>> points(point_count);
        std::vector<uint32_t> centers(atom_count);
        double min_x = std::numeric_limits<double>::infinity();
        double min_y = min_x;
        double min_z = min_x;
        double max_x = -min_x;
        double max_y = -min_x;
        double max_z = -min_x;
        double max_expanded_radius = 0.0;
        for (uint32_t i = 0; i < atom_count; ++i) {
            const double px = xyz != nullptr ? xyz[3u * i] : x[i];
            const double py = xyz != nullptr ? xyz[3u * i + 1u] : y[i];
            const double pz = xyz != nullptr ? xyz[3u * i + 2u] : z[i];
            const double radius = radii[i];
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) ||
                !std::isfinite(radius) || radius <= 0.0) {
                throw std::runtime_error("coordinates and radii must be finite; radii must be positive");
            }
            atoms[i] = {px, py, pz, radius};
            min_x = std::min(min_x, px);
            min_y = std::min(min_y, py);
            min_z = std::min(min_z, pz);
            max_x = std::max(max_x, px);
            max_y = std::max(max_y, py);
            max_z = std::max(max_z, pz);
            max_expanded_radius = std::max(max_expanded_radius, radius + probe_radius);
            centers[i] = i;
        }
        for (uint32_t i = 0; i < atom_count; ++i) {
            const double expanded = atoms[i].w + probe_radius;
            shadow[i] = {(float)(atoms[i].x - min_x), (float)(atoms[i].y - min_y),
                        (float)(atoms[i].z - min_z), (float)(expanded * expanded)};
        }
        for (uint32_t i = 0; i < point_count; ++i) {
            points[i] = {sphere_xyz[3u * i], sphere_xyz[3u * i + 1u],
                        sphere_xyz[3u * i + 2u], 0.0};
        }

        const double cell_size = 2.0 * max_expanded_radius;
        auto dimension = [cell_size](double minimum, double maximum) {
            const double cells = std::floor((maximum - minimum) / cell_size) + 1.0;
            if (cells < 1.0 || cells > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("invalid cell-grid dimension");
            }
            return static_cast<uint32_t>(cells);
        };
        const uint32_t dim_x = dimension(min_x, max_x);
        const uint32_t dim_y = dimension(min_y, max_y);
        const uint32_t dim_z = dimension(min_z, max_z);
        const uint64_t cell_count = static_cast<uint64_t>(dim_x) * dim_y * dim_z;
        if (cell_count == 0 || cell_count > kMaxCells) {
            throw std::runtime_error("cell grid is too large; unwrap or filter sparse coordinates");
        }

        const VkDeviceSize atom_bytes = sizeof(Vec4T<double>) * (VkDeviceSize)atom_count;
        const VkDeviceSize point_storage_bytes = sizeof(Vec4T<double>) * points.size();
        const VkDeviceSize head_bytes = sizeof(int32_t) * cell_count;
        const VkDeviceSize next_bytes = sizeof(int32_t) * (VkDeviceSize)atom_count;
        const VkDeviceSize center_bytes = sizeof(uint32_t) * (VkDeviceSize)atom_count;
        const VkDeviceSize shadow_bytes = sizeof(Vec4T<float>) * (VkDeviceSize)atom_count;
        const VkDeviceSize exposed_bytes = sizeof(uint32_t) *
            (VkDeviceSize)atom_count * (VkDeviceSize)point_count;
        check_storage_range(context, atom_bytes, "atom buffer");
        check_storage_range(context, point_storage_bytes, "point buffer");
        check_storage_range(context, head_bytes, "cell-head buffer");
        check_storage_range(context, next_bytes, "cell-link buffer");
        check_storage_range(context, center_bytes, "center buffer");
        check_storage_range(context, exposed_bytes, "exposed-point buffer");

        const VkMemoryPropertyFlags device_memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkMemoryPropertyFlags host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ensure_buffer(context, &context->atoms, atom_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->points, point_storage_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->cell_counts, sizeof(int32_t) * cell_count,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      device_memory);
        ensure_buffer(context, &context->heads, sizeof(int32_t) * (cell_count + 1u),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->next, next_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        ensure_buffer(context, &context->centers, center_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->exposed_points, exposed_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_memory);
        ensure_buffer(context, &context->staging_atoms, atom_bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));
        ensure_buffer(context, &context->staging_points, point_storage_bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_centers, center_bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory);
        ensure_buffer(context, &context->staging_exposed_points, exposed_bytes,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      readback_memory_flags(context->physical));
        ensure_buffer(context, &context->atoms_shadow, shadow_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_memory);
        ensure_buffer(context, &context->atoms_shadow_sorted, shadow_bytes,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_memory);
        ensure_buffer(context, &context->staging_atoms_shadow, shadow_bytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, readback_memory_flags(context->physical));

        std::memcpy(context->staging_atoms_shadow.mapped, shadow.data(), shadow_bytes);
        std::memcpy(context->staging_atoms.mapped, atoms.data(), atom_bytes);
        std::memcpy(context->staging_points.mapped, points.data(), point_storage_bytes);
        std::memcpy(context->staging_centers.mapped, centers.data(), center_bytes);
        update_descriptors_exposed_points(context);

        const ParametersT<double> parameters{
            atom_count, point_count, atom_count, 0u, dim_x, dim_y, dim_z,
            probe_radius, cell_size, min_x, min_y, min_z,
            sr_hybrid_margin(cell_size, dim_x, dim_y, dim_z)};
        check(vkResetCommandBuffer(context->command, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(context->command, &begin), "vkBeginCommandBuffer");

        VkBufferCopy atom_copy{0, 0, atom_bytes};
        VkBufferCopy shadow_copy{0, 0, shadow_bytes};
        VkBufferCopy point_copy{0, 0, point_storage_bytes};
        VkBufferCopy center_copy{0, 0, center_bytes};
        vkCmdCopyBuffer(context->command, context->staging_atoms.handle,
                        context->atoms.handle, 1, &atom_copy);
        vkCmdCopyBuffer(context->command, context->staging_atoms_shadow.handle,
                        context->atoms_shadow.handle, 1, &shadow_copy);
        vkCmdCopyBuffer(context->command, context->staging_points.handle,
                        context->points.handle, 1, &point_copy);
        vkCmdCopyBuffer(context->command, context->staging_centers.handle,
                        context->centers.handle, 1, &center_copy);
        record_cell_list_build(context, parameters, atom_count, cell_count,
                               /*use_fp64=*/true, /*gather_shadow=*/true);

        vkCmdBindPipeline(context->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          context->sr_exposed_points_fp64_pipeline);
        /* One workgroup per atom, matching the non-subgroup SR dispatch
         * geometry exactly (indices_per_group=1). */
        dispatch_chunked(context, parameters, atom_count, 1u);

        VkMemoryBarrier output_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &output_barrier,
                             0, nullptr, 0, nullptr);
        VkBufferCopy exposed_copy{0, 0, exposed_bytes};
        vkCmdCopyBuffer(context->command, context->exposed_points.handle,
                        context->staging_exposed_points.handle, 1, &exposed_copy);
        VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(context->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier,
                             0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(context->command), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &context->command;
        check(vkQueueSubmit(context->queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(context->queue), "vkQueueWaitIdle");

        const uint32_t *bits =
            static_cast<const uint32_t *>(context->staging_exposed_points.mapped);
        const size_t total_points = (size_t)atom_count * (size_t)point_count;
        for (size_t i = 0; i < total_points; ++i) {
            exposed[i] = bits[i] != 0u ? (unsigned char)1 : (unsigned char)0;
        }
        context->error.clear();
        return 0;
    } catch (const std::exception &error) {
        context->error = error.what();
        return 1;
    }
}

} // namespace

namespace {
thread_local std::string create_error;
} // namespace

extern "C" int fastsasa_vk_context_create(fastsasa_vk_context **output, int device_index)
{
    if (output == nullptr) return 1;
    *output = nullptr;
    fastsasa_vk_context *context = nullptr;
    try {
        context = new fastsasa_vk_context();
        initialize(context, device_index);
        *output = context;
        create_error.clear();
        return 0;
    } catch (const std::exception &error) {
        try {
            create_error = error.what();
        } catch (...) {
        }
        if (context != nullptr) fastsasa_vk_context_free(context);
        return 1;
    } catch (...) {
        if (context != nullptr) fastsasa_vk_context_free(context);
        return 1;
    }
}

extern "C" const char *fastsasa_vk_create_error(void)
{
    return create_error.c_str();
}

extern "C" void fastsasa_vk_context_free(fastsasa_vk_context *context)
{
    if (context == nullptr) return;
    if (context->device != VK_NULL_HANDLE) vkDeviceWaitIdle(context->device);
    destroy_buffer(context, &context->staging_atoms_shadow);
    destroy_buffer(context, &context->atoms_shadow);
    destroy_buffer(context, &context->staging_exposed_points);
    destroy_buffer(context, &context->exposed_points);
    destroy_buffer(context, &context->staging_areas);
    destroy_buffer(context, &context->staging_centers);
    destroy_buffer(context, &context->staging_points);
    destroy_buffer(context, &context->staging_atoms);
    destroy_buffer(context, &context->areas);
    destroy_buffer(context, &context->centers);
    destroy_buffer(context, &context->next);
    destroy_buffer(context, &context->heads);
    destroy_buffer(context, &context->cell_counts);
    destroy_buffer(context, &context->atoms_shadow_sorted);
    destroy_buffer(context, &context->points);
    destroy_buffer(context, &context->atoms);
    if (context->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(context->device, context->command_pool, nullptr);
    if (context->sr64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr64_pipeline, nullptr);
    if (context->sr_sg_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr_sg_pipeline, nullptr);
    if (context->sr32_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr32_pipeline, nullptr);
    if (context->lr_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->lr_pipeline, nullptr);
    if (context->lr_reduce_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->lr_reduce_pipeline, nullptr);
    if (context->count_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->count_pipeline, nullptr);
    if (context->scan_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->scan_pipeline, nullptr);
    if (context->fill_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->fill_pipeline, nullptr);
    if (context->gather_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->gather_pipeline, nullptr);
    if (context->sr_fp64_64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr_fp64_64_pipeline, nullptr);
    if (context->sr_fp64_sg_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr_fp64_sg_pipeline, nullptr);
    if (context->sr_fp64_32_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr_fp64_32_pipeline, nullptr);
    if (context->lr_fp64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->lr_fp64_pipeline, nullptr);
    if (context->lr_reduce_fp64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->lr_reduce_fp64_pipeline, nullptr);
    if (context->count_fp64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->count_fp64_pipeline, nullptr);
    if (context->fill_fp64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->fill_fp64_pipeline, nullptr);
    if (context->sr_exposed_points_fp64_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(context->device, context->sr_exposed_points_fp64_pipeline, nullptr);
    if (context->sr64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr64_shader, nullptr);
    if (context->sr_sg_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr_sg_shader, nullptr);
    if (context->sr32_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr32_shader, nullptr);
    if (context->lr_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->lr_shader, nullptr);
    if (context->lr_reduce_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->lr_reduce_shader, nullptr);
    if (context->count_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->count_shader, nullptr);
    if (context->scan_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->scan_shader, nullptr);
    if (context->fill_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->fill_shader, nullptr);
    if (context->gather_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->gather_shader, nullptr);
    if (context->sr_fp64_64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr_fp64_64_shader, nullptr);
    if (context->sr_fp64_sg_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr_fp64_sg_shader, nullptr);
    if (context->sr_fp64_32_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr_fp64_32_shader, nullptr);
    if (context->lr_fp64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->lr_fp64_shader, nullptr);
    if (context->lr_reduce_fp64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->lr_reduce_fp64_shader, nullptr);
    if (context->count_fp64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->count_fp64_shader, nullptr);
    if (context->fill_fp64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->fill_fp64_shader, nullptr);
    if (context->sr_exposed_points_fp64_shader != VK_NULL_HANDLE)
        vkDestroyShaderModule(context->device, context->sr_exposed_points_fp64_shader, nullptr);
    if (context->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(context->device, context->pipeline_layout, nullptr);
    if (context->descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(context->device, context->descriptor_pool, nullptr);
    if (context->descriptor_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(context->device, context->descriptor_layout, nullptr);
    if (context->device != VK_NULL_HANDLE) vkDestroyDevice(context->device, nullptr);
    if (context->instance != VK_NULL_HANDLE) vkDestroyInstance(context->instance, nullptr);
    delete context;
}

extern "C" const char *fastsasa_vk_device_name(const fastsasa_vk_context *context)
{
    return context == nullptr ? "" : context->properties.deviceName;
}

extern "C" const char *fastsasa_vk_last_error(const fastsasa_vk_context *context)
{
    return context == nullptr ? "invalid Vulkan context" : context->error.c_str();
}

extern "C" uint32_t fastsasa_vk_subgroup_size(const fastsasa_vk_context *context)
{
    return context == nullptr ? 0u : context->subgroup_size;
}

extern "C" uint32_t fastsasa_vk_sr_workgroup_size(const fastsasa_vk_context *context)
{
    return context == nullptr ? 0u : context->sr_workgroup_size;
}

extern "C" int fastsasa_vk_supports_fp64(const fastsasa_vk_context *context)
{
    const char *disabled = std::getenv("FASTSASA_TEST_VULKAN_NO_FP64");

    if (disabled != nullptr && disabled[0] != '\0' &&
        std::strcmp(disabled, "0") != 0) {
        return 0;
    }
    return context != nullptr && context->shader_float64 ? 1 : 0;
}

static int require_precision(fastsasa_vk_context *context, int use_fp64)
{
    if (context == nullptr) return 0;
    /* fastsasa_vk_supports_fp64 also honors the FP64 test override, so direct
     * backend calls and fastsasa_context_set_precision gate identically. */
    if (use_fp64 && !fastsasa_vk_supports_fp64(context)) {
        context->error = "Vulkan device does not support shaderFloat64; use FP32 or the CPU backend";
        return 0;
    }
    return 1;
}

extern "C" int fastsasa_vk_sr(fastsasa_vk_context *context,
                             const double *xyz,
                             const double *radii,
                             uint32_t atom_count,
                             const double *sphere_xyz,
                             uint32_t point_count,
                             double probe_radius,
                             int use_fp64,
                             double *sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate<double>(context, xyz, radii, atom_count, sphere_xyz,
                            point_count, probe_radius, nullptr, atom_count,
                            sasa, false)
        : calculate<float>(context, xyz, radii, atom_count, sphere_xyz,
                           point_count, probe_radius, nullptr, atom_count,
                           sasa, false);
}

extern "C" int fastsasa_vk_sr_exposed_points(fastsasa_vk_context *context,
                                            const double *xyz,
                                            const double *x,
                                            const double *y,
                                            const double *z,
                                            const double *radii,
                                            uint32_t atom_count,
                                            const double *sphere_xyz,
                                            uint32_t point_count,
                                            double probe_radius,
                                            unsigned char *exposed)
{
    return calculate_exposed_points(context, xyz, x, y, z, radii, atom_count,
                                    sphere_xyz, point_count, probe_radius, exposed);
}

extern "C" int fastsasa_vk_sr_centers(fastsasa_vk_context *context,
                                     const double *xyz,
                                     const double *radii,
                                     uint32_t atom_count,
                                     const double *sphere_xyz,
                                     uint32_t point_count,
                                     double probe_radius,
                                     const uint32_t *center_indices,
                                     uint32_t center_count,
                                     int use_fp64,
                                     double *sasa)
{
    if (center_indices == nullptr) {
        if (context != nullptr) context->error = "center_indices is null";
        return 1;
    }
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate<double>(context, xyz, radii, atom_count, sphere_xyz,
                            point_count, probe_radius, center_indices,
                            center_count, sasa, false)
        : calculate<float>(context, xyz, radii, atom_count, sphere_xyz,
                           point_count, probe_radius, center_indices,
                           center_count, sasa, false);
}

extern "C" int fastsasa_vk_lee_richards(fastsasa_vk_context *context,
                                       const double *xyz,
                                       const double *radii,
                                       uint32_t atom_count,
                                       uint32_t slice_count,
                                       double probe_radius,
                                       int use_fp64,
                                       double *sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate<double>(context, xyz, radii, atom_count, nullptr,
                            slice_count, probe_radius, nullptr, atom_count,
                            sasa, true)
        : calculate<float>(context, xyz, radii, atom_count, nullptr,
                           slice_count, probe_radius, nullptr, atom_count,
                           sasa, true);
}

extern "C" int fastsasa_vk_lee_richards_centers(fastsasa_vk_context *context,
                                               const double *xyz,
                                               const double *radii,
                                               uint32_t atom_count,
                                               uint32_t slice_count,
                                               double probe_radius,
                                               const uint32_t *center_indices,
                                               uint32_t center_count,
                                               int use_fp64,
                                               double *sasa)
{
    if (center_indices == nullptr) {
        if (context != nullptr) context->error = "center_indices is null";
        return 1;
    }
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate<double>(context, xyz, radii, atom_count, nullptr,
                            slice_count, probe_radius, center_indices,
                            center_count, sasa, true)
        : calculate<float>(context, xyz, radii, atom_count, nullptr,
                           slice_count, probe_radius, center_indices,
                           center_count, sasa, true);
}

extern "C" int fastsasa_vk_sr_frames(fastsasa_vk_context *context,
                                    const double *frame_xyz,
                                    const double *radii,
                                    uint32_t frame_count,
                                    uint32_t atom_count,
                                    const double *sphere_xyz,
                                    uint32_t point_count,
                                    double probe_radius,
                                    int use_fp64,
                                    double *frame_sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate_frames<double>(context, frame_xyz, radii, frame_count,
                                   atom_count, sphere_xyz, point_count,
                                   probe_radius, nullptr, atom_count,
                                   frame_sasa, false)
        : calculate_frames<float>(context, frame_xyz, radii, frame_count,
                                  atom_count, sphere_xyz, point_count,
                                  probe_radius, nullptr, atom_count,
                                  frame_sasa, false);
}

extern "C" int fastsasa_vk_sr_center_frames(fastsasa_vk_context *context,
                                           const double *frame_xyz,
                                           const double *radii,
                                           uint32_t frame_count,
                                           uint32_t atom_count,
                                           const double *sphere_xyz,
                                           uint32_t point_count,
                                           double probe_radius,
                                           const uint32_t *center_indices,
                                           uint32_t center_count,
                                           int use_fp64,
                                           double *frame_sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate_frames<double>(context, frame_xyz, radii, frame_count,
                                   atom_count, sphere_xyz, point_count,
                                   probe_radius, center_indices, center_count,
                                   frame_sasa, false)
        : calculate_frames<float>(context, frame_xyz, radii, frame_count,
                                  atom_count, sphere_xyz, point_count,
                                  probe_radius, center_indices, center_count,
                                  frame_sasa, false);
}

extern "C" int fastsasa_vk_lee_richards_center_frames(fastsasa_vk_context *context,
                                                     const double *frame_xyz,
                                                     const double *radii,
                                                     uint32_t frame_count,
                                                     uint32_t atom_count,
                                                     uint32_t slice_count,
                                                     double probe_radius,
                                                     const uint32_t *center_indices,
                                                     uint32_t center_count,
                                                     int use_fp64,
                                                     double *frame_sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate_frames<double>(context, frame_xyz, radii, frame_count,
                                   atom_count, nullptr, slice_count,
                                   probe_radius, center_indices, center_count,
                                   frame_sasa, true)
        : calculate_frames<float>(context, frame_xyz, radii, frame_count,
                                  atom_count, nullptr, slice_count,
                                  probe_radius, center_indices, center_count,
                                  frame_sasa, true);
}

extern "C" int fastsasa_vk_lee_richards_frames(fastsasa_vk_context *context,
                                              const double *frame_xyz,
                                              const double *radii,
                                              uint32_t frame_count,
                                              uint32_t atom_count,
                                              uint32_t slice_count,
                                              double probe_radius,
                                              int use_fp64,
                                              double *frame_sasa)
{
    if (!require_precision(context, use_fp64)) return 1;
    return use_fp64
        ? calculate_frames<double>(context, frame_xyz, radii, frame_count,
                                   atom_count, nullptr, slice_count,
                                   probe_radius, nullptr, atom_count,
                                   frame_sasa, true)
        : calculate_frames<float>(context, frame_xyz, radii, frame_count,
                                  atom_count, nullptr, slice_count,
                                  probe_radius, nullptr, atom_count,
                                  frame_sasa, true);
}
