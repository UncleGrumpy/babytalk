idf_component_register(
    SRCS "vbench.c"
    INCLUDE_DIRS "."
    REQUIRES mmrt esp_timer esp_partition spi_flash esp_psram heap
)
