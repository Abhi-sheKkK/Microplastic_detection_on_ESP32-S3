
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "run_model.cc"

extern "C" void app_main(void)
{

    setup();
    // while (true)
    // {
    //     loop();

    //     // trigger one inference every 500ms
    //     vTaskDelay(pdMS_TO_TICKS(500));
    // }
}
