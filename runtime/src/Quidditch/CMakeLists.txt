
iree_register_external_hal_driver(
        NAME
        Quidditch
        DRIVER_TARGET
        Quidditch_Quidditch
        REGISTER_FN
        iree_hal_quidditch_driver_module_register
)

iree_cc_library(
  NAME
    Quidditch
  SRCS
    HelloWorld.cpp
  DEPS
    iree::base
  PUBLIC
)
