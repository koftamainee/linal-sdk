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

  std::cout << "Starting LaTeX compilation via XeLaTeX...\n";

  auto start = std::chrono::steady_clock::now();

  int compile_result = system(compile_command.c_str());

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  if (compile_result == 0) {
    std::cout << "Successfully generated " << pdf_file << " in "
              << elapsed_seconds.count() << " seconds\n";
  } else {
    std::cout << "Compilation failed with error code: " << compile_result
              << "\n";
  }
  try {
    fs::remove(tex_file);
    fs::remove(base_name + "-solved.aux");
    fs::remove(base_name + "-solved.log");
    fs::remove("indent.log");
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Warning: Could not clean up temporary files: " << e.what()
              << "\n";
  }

  return 0;
}
