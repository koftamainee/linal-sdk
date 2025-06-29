#pragma once

#include <string>

#include "writer/latex_writer.h"

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

  void solve_matrix_add(std::string const& data);
  void solve_matrix_multiply(std::string const& data);
  void solve_matrix_scalar_multiply(std::string const& data);
  void solve_matrix_determinant(std::string const& data);
  void solve_matrix_inverse(std::string const& data);
  void solve_matrix_gauss(std::string const& data);
  void solve_matrix_gauss_jordan(std::string const& data);
  void solve_matrix_eigenvalues(std::string const& data);
  void solve_matrix_eigenvectors(std::string const& data);
  void solve_matrix_rank(std::string const& data);
  void solve_span_dimension(std::string const& data);
  void solve_vector_in_span(std::string const& data);

  void solve_line2D_equasions(std::string const& data);
  void solve_line2D_intersection(std::string const& data);
  void solve_distance_point_to_line(const std::string& data);
  void solve_distance_between_parallel_lines(const std::string& data);
  void solve_area_triangle_with_axes(const std::string& data);
  void solve_angle_between_lines(const std::string& data);
  void solve_line_segment_intersection(const std::string& data);
  void solve_distance_point_to_segment(const std::string& data);
  void solve_segment_segment_intersection(const std::string& data);

  void solve_distance_point_to_line_nd(const std::string& data);
  void solve_symmetric_point_nd(const std::string& data);
  void solve_line_equations_nd(const std::string& data);
  void solve_points_collinearity_nd(const std::string& data);

 public:
  Solver();

  bool set_output(const std::string& filename);
  bool process_file(const std::string& input_path);
  void close();
};
