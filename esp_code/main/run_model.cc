// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "esp_flash.h"
// #include "esp_partition.h"
// #include "esp_system.h"
// #include <cstdint>
// #include <cstddef>
// #include "model.h"

// // Tensor arena
// constexpr int kTensorArenaSize = 1024 * 256; // 256 KB
// static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// namespace
// {
//     const tflite::Model *model = nullptr;
//     tflite::MicroInterpreter *interpreter = nullptr;
//     // IMPORTANT: Add the operations used by your model here.
//     // If you are not sure, you can use AllOpsResolver, but it will increase the binary size.
//     // #include "tensorflow/lite/micro/all_ops_resolver.h"
//     // tflite::AllOpsResolver resolver;
//     tflite::MicroMutableOpResolver<11> resolver; // Adjust number of ops as needed
// } // namespace

// void check_flash_memory()
// {
//     // Get flash chip size
//     uint32_t flash_size;
//     esp_err_t err = esp_flash_get_size(NULL, &flash_size);
//     if (err != ESP_OK)
//     {
//         MicroPrintf("Failed to get flash size: %s", esp_err_to_name(err));
//         return;
//     }
//     MicroPrintf("Total Flash Size: %u bytes (%.2f MB)", flash_size, flash_size / (1024.0 * 1024.0));

//     // Find the factory partition (where the app and model are stored)
//     const esp_partition_t *app_partition = esp_partition_find_first(ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_APP_FACTORY, NULL);
//     if (app_partition == NULL)
//     {
//         MicroPrintf("Failed to find factory partition!");
//         return;
//     }

//     // Get partition size
//     uint32_t partition_size = app_partition->size;
//     MicroPrintf("Factory Partition Size: %u bytes (%.2f MB)", partition_size, partition_size / (1024.0 * 1024.0));

//     // Estimate used space (approximate, based on model size + code)
//     uint32_t binary_size_approx = model_quant_tflite_len + 0x10000; // Model size + ~64 KB for code
//     uint32_t remaining_space = partition_size - binary_size_approx;

//     MicroPrintf("Approximate Used Flash (model + code): %u bytes (%.2f MB)", binary_size_approx, binary_size_approx / (1024.0 * 1024.0));
//     MicroPrintf("Approximate Remaining Flash in Partition: %u bytes (%.2f MB)", remaining_space, remaining_space / (1024.0 * 1024.0));
// }

// void test_model()
// {
//     // Map the model
//     model = tflite::GetModel(model_quant_tflite);
//     if (model->version() != TFLITE_SCHEMA_VERSION)
//     {
//         MicroPrintf("Model schema %d != supported %d", model->version(), TFLITE_SCHEMA_VERSION);
//         return;
//     }
//     MicroPrintf("Model loaded successfully.");

//     // Add the operations used by your model here. For example:
//     resolver.AddConv2D();
//     resolver.AddMaxPool2D();
//     resolver.AddReshape();
//     resolver.AddFullyConnected();
//     resolver.AddDepthwiseConv2D();
//     resolver.AddAdd();
//     resolver.AddTranspose();
//     resolver.AddConcatenation();
//     resolver.AddLogistic();
//     resolver.AddRelu();
//     resolver.AddMul();

//     // Build an interpreter to run the model with.
//     static tflite::MicroInterpreter static_interpreter(
//         model, resolver, tensor_arena, kTensorArenaSize);
//     interpreter = &static_interpreter;

//     // Allocate memory from the tensor_arena for the model's tensors.
//     TfLiteStatus allocate_status = interpreter->AllocateTensors();
//     if (allocate_status != kTfLiteOk)
//     {
//         MicroPrintf("AllocateTensors() failed");
//         return;
//     }

//     MicroPrintf("Arena used: %zu bytes", interpreter->arena_used_bytes());
//     // Print all input tensors
//     for (size_t i = 0; i < interpreter->inputs_size(); i++)
//     {
//         TfLiteTensor *input = interpreter->input(i);
//         MicroPrintf("Input tensor %u:", i);
//         MicroPrintf("  Type: %s", TfLiteTypeGetName(input->type));
//         MicroPrintf("  Dimensions: %d", input->dims->size);
//         for (int j = 0; j < input->dims->size; j++)
//         {
//             MicroPrintf("    Dim %d: %d", j, input->dims->data[j]);
//         }
//     }

//     // Print all output tensors
//     for (size_t i = 0; i < interpreter->outputs_size(); i++)
//     {
//         TfLiteTensor *output = interpreter->output(i);
//         MicroPrintf("Output tensor %u:", i);
//         MicroPrintf("  Type: %s", TfLiteTypeGetName(output->type));
//         MicroPrintf("  Dimensions: %d", output->dims->size);
//         for (int j = 0; j < output->dims->size; j++)
//         {
//             MicroPrintf("    Dim %d: %d", j, output->dims->data[j]);
//         }
//     }
//     // Set dummy inputs (wave_input: [1, 1, 200], feat_input: [1, 7], FLOAT32)
//     TfLiteTensor *input_wave = interpreter->input(0);
//     TfLiteTensor *input_feat = interpreter->input(1);
//     if (input_wave->type == kTfLiteFloat32 && input_feat->type == kTfLiteFloat32)
//     {
//         for (size_t i = 0; i < input_wave->bytes / sizeof(float); i++)
//         {
//             input_wave->data.f[i] = 0.0f; // Dummy value (e.g., zeros)
//         }
//         for (size_t i = 0; i < input_feat->bytes / sizeof(float); i++)
//         {
//             input_feat->data.f[i] = 0.0f; // Dummy value (e.g., zeros)
//         }
//     }
//     else
//     {
//         MicroPrintf("Unsupported input type: wave=%s, feat=%s", TfLiteTypeGetName(input_wave->type), TfLiteTypeGetName(input_feat->type));
//         return;
//     }

//     // Run inference
//     if (interpreter->Invoke() != kTfLiteOk)
//     {
//         MicroPrintf("Inference failed!");
//         return;
//     }
//     MicroPrintf("Inference successful");

//     // Get and print output ([1, 1], FLOAT32)
//     TfLiteTensor *output = interpreter->output(0);
//     if (output->type == kTfLiteFloat32)
//     {
//         float value = output->data.f[0]; // Single float output
//         MicroPrintf("Output value: %f", value);
//     }
//     else
//     {
//         MicroPrintf("Unsupported output type: %s", TfLiteTypeGetName(output->type));
//     }
// }

// extern "C" void setup()
// {
//     tflite::InitializeTarget();
//     check_flash_memory(); // Check flash memory details
//     test_model();         // Test the model
// }

// // extern "C" void loop() {
// //   // Empty for now
// // }

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_flash.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include "model.h"

// Tensor arena in PSRAM
constexpr int kTensorArenaSize = 512 * 1024; 
static uint8_t* tensor_arena = nullptr;

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  tflite::MicroMutableOpResolver<11> resolver; 
} 

void check_flash_memory() {
  uint32_t flash_size;
  esp_err_t err = esp_flash_get_size(NULL, &flash_size);
  if (err != ESP_OK) {
    MicroPrintf("Failed to get flash size: %s", esp_err_to_name(err));
    return;
  }
  MicroPrintf("Total Flash Size: %u bytes (%.2f MB)", flash_size, flash_size / (1024.0 * 1024.0));

  const esp_partition_t* app_partition = esp_partition_find_first(ESP_PARTITION_TYPE_APP, ESP_PARTITION_SUBTYPE_APP_FACTORY, NULL);
  if (app_partition == NULL) {
    MicroPrintf("Failed to find factory partition!");
    return;
  }

  uint32_t partition_size = app_partition->size;
  MicroPrintf("Factory Partition Size: %u bytes (%.2f MB)", partition_size, partition_size / (1024.0 * 1024.0));

  uint32_t binary_size_approx = model_quant_int8_tflite_len + 0x10000;
  uint32_t remaining_space = partition_size - binary_size_approx;
  MicroPrintf("Approximate Used Flash (model + code): %u bytes (%.2f MB)", binary_size_approx, binary_size_approx / (1024.0 * 1024.0));
  MicroPrintf("Approximate Remaining Flash in Partition: %u bytes (%.2f MB)", remaining_space, remaining_space / (1024.0 * 1024.0));
}

void check_psram() {
  esp_err_t psram_init_result = esp_psram_init();
  if (psram_init_result != ESP_OK) {
    MicroPrintf("PSRAM initialization failed: %s", esp_err_to_name(psram_init_result));
    return;
  }
  if (!esp_psram_is_initialized()) {
    MicroPrintf("PSRAM is not initialized!");
    return;
  }
  size_t total_psram = esp_psram_get_size();
  size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t largest_free_block = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
  MicroPrintf("PSRAM Total Size: %u bytes (%.2f MB)", total_psram, total_psram / (1024.0 * 1024.0));
  MicroPrintf("PSRAM Free Size: %u bytes (%.2f MB)", free_psram, free_psram / (1024.0 * 1024.0));
  MicroPrintf("PSRAM Largest Free Block: %u bytes (%.2f MB)", largest_free_block, largest_free_block / (1024.0 * 1024.0));
}

void test_model() {
  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tensor_arena == nullptr) {
    MicroPrintf("Failed to allocate tensor arena in PSRAM!");
    return;
  }
  MicroPrintf("Tensor arena allocated in PSRAM: %u bytes", kTensorArenaSize);

  // Map the model
  model = tflite::GetModel(model_quant_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  MicroPrintf("Model loaded successfully.");

  // Add operations
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddTranspose();
  resolver.AddConcatenation();
  resolver.AddLogistic();
  resolver.AddRelu();
  resolver.AddMul();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  MicroPrintf("Arena used: %zu bytes", interpreter->arena_used_bytes());

  // Print input tensors
  for (size_t i = 0; i < interpreter->inputs_size(); i++) {
    TfLiteTensor* input = interpreter->input(i);
    MicroPrintf("Input tensor %u:", i);
    MicroPrintf("  Type: %s", TfLiteTypeGetName(input->type));
    MicroPrintf("  Dimensions: %d", input->dims->size);
    for (int j = 0; j < input->dims->size; j++) {
      MicroPrintf("    Dim %d: %d", j, input->dims->data[j]);
    }
  }

  // Print output tensors
  for (size_t i = 0; i < interpreter->outputs_size(); i++) {
    TfLiteTensor* output = interpreter->output(i);
    MicroPrintf("Output tensor %u:", i);
    MicroPrintf("  Type: %s", TfLiteTypeGetName(output->type));
    MicroPrintf("  Dimensions: %d", output->dims->size);
    for (int j = 0; j < output->dims->size; j++) {
      MicroPrintf("    Dim %d: %d", j, output->dims->data[j]);
    }
  }

  // Set dummy inputs (wave_input: [1, 1, 200], feat_input: [1, 7], FLOAT32)
  TfLiteTensor* input_wave = interpreter->input(0);
  TfLiteTensor* input_feat = interpreter->input(1);
  if (input_wave->type != kTfLiteInt8 || input_feat->type != kTfLiteInt8) {
    MicroPrintf("Unsupported input type: wave=%s, feat=%s", TfLiteTypeGetName(input_wave->type), TfLiteTypeGetName(input_feat->type));
    return;
  }

  // Run 100 inferences and measure time
  const int num_inferences = 100;
  int64_t total_time_us = 0;
  for (int i = 0; i < num_inferences; i++) {
    // Set dummy inputs (random values for variation)
    for (size_t j = 0; j < input_wave->bytes / sizeof(int); j++) {
      input_wave->data.f[j] = static_cast<int>(rand()) / RAND_MAX; // Random [0, 1]
    }
    for (size_t j = 0; j < input_feat->bytes / sizeof(int); j++) {
      input_feat->data.f[j] = static_cast<int>(rand()) / RAND_MAX; // Random [0, 1]
    }

    // Measure inference time
    int64_t start_time = esp_timer_get_time(); // Verified: no arguments, returns int64_t
    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("Inference %d failed!", i);
      return;
    }
    int64_t end_time = esp_timer_get_time(); // Verified: no arguments, returns int64_t
    total_time_us += (end_time - start_time);

    // Print output for each inference
    TfLiteTensor* output = interpreter->output(0);
    if (output->type == kTfLiteInt8) {
      float value = output->data.f[0];
      MicroPrintf("Inference %d output: %f", i, value);
    } else {
      MicroPrintf("Unsupported output type: %s", TfLiteTypeGetName(output->type));
      return;
    }
  }

  // Calculate and print average inference time
  float avg_time_ms = (total_time_us / static_cast<float>(num_inferences)) / 1000.0f;
  MicroPrintf("Ran %d inferences successfully", num_inferences);
  MicroPrintf("Average inference time: %.3f ms", avg_time_ms);

  // Free tensor arena
  heap_caps_free(tensor_arena); // Verified: takes void* pointer
  tensor_arena = nullptr;
}

extern "C" void setup() {
  tflite::InitializeTarget();
  check_flash_memory();
  check_psram();
  test_model();
}

extern "C" void loop() {
  // Empty
}