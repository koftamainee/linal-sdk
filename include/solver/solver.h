#pragma once

#include <string>

#include "writter/latex_writter.h"

class Solver {
 private:
  LatexWriter& writer;

  void solve_dot_product(std::string const& data);
  void solve_3D_cross_product(std::string const& data);
  void solve_7D_cross_product(std::string const& data);
  void solve_3D_triple_product(std::string const& data);
  void solve_7D_triple_product(std::string const& data);
  void solve_vector_add(std::string const& data);
  void solve_vector_sub(std::string const& data);
  void solve_scalar_mul(std::string const& data);
  void solve_norm(std::string const& data);
  void solve_gram_schmidt_process(const std::string& data);

 public:
  Solver();

  bool set_output(const std::string& filename);
  bool process_file(const std::string& input_path);
  void close();
};
