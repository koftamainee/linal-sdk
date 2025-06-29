#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "solver/solver.h"

namespace fs = std::filesystem;

std::string remove_extension(const std::string& filename) {
  size_t lastdot = filename.find_last_of('.');
  if (lastdot == std::string::npos) {
    return filename;
  }
  return filename.substr(0, lastdot);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file>\n";
    return 1;
  }

  std::string input_file = argv[1];
  std::string base_name = remove_extension(input_file);
  std::string tex_file = base_name + "-solved.tex";
  std::string pdf_file = base_name + "-solved.pdf";

  auto s = Solver();

  if (!s.set_output(tex_file)) {
    std::cerr << "Error: Failed to set output file " << tex_file << "\n";
    return 1;
  }

  if (!s.process_file(input_file)) {
    std::cerr << "Error: Failed to process input file " << input_file << "\n";
    return 1;
  }

  s.close();

  std::string compile_command =
      "xelatex -interaction=nonstopmode > /dev/null " + tex_file;
  int compile_result = system(compile_command.c_str());

  if (compile_result != 0) {
    std::cerr << "Error: LaTeX compilation failed\n";
    return 1;
  }

  try {
    // fs::remove(tex_file);
    fs::remove(base_name + "-solved.aux");
    fs::remove(base_name + "-solved.log");
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Warning: Could not clean up temporary files: " << e.what()
              << "\n";
  }

  std::cout << "Successfully generated " << pdf_file << "\n";
  return 0;
}
