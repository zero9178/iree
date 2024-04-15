
#include "iree/base/status.h"
#include "iree/hal/driver_registry.h"

extern "C" iree_status_t iree_hal_quidditch_driver_module_register(
    iree_hal_driver_registry_t*) {
  return iree_ok_status();
}
