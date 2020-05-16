#ifndef OUTPUT_FILES_H
#define OUTPUT_FILES_H

#include <fstream>

static std::ofstream          turnover_output_file{"turnover",          std::ios_base::app};
static std::ofstream       turnover_EI_output_file("turnover_EI",       std::ios_base::app);
static std::ofstream    scaling_deltas_output_file("scaling_deltas",    std::ios_base::app);
static std::ofstream scaling_deltas_EI_output_file{"scaling_deltas_EI", std::ios_base::app};
static std::ofstream      spk_register_output_file{"spk_register",      std::ios_base::app};
static std::ofstream   spk_register_EI_output_file("spk_register_EI",   std::ios_base::app);

#endif // OUTPUT_FILES_h