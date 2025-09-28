idf_component_register(SRCS "main.cpp" "run_model.cc"
                       INCLUDE_DIRS "."
                       REQUIRES esp-tflite-micro spi_flash esp_partition esp_psram esp_system esp_timer)