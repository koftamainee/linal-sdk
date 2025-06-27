#pragma once

#include <string>
#include <vector>

#include "core/bigfloat.h"
#include "latex_writter.h"

class Matrix {
 private:
  std::vector<std::vector<bigfloat>> data_;
  LatexWriter& writter = LatexWriter::get_instance();
  size_t rows_, cols_;

  void check_same_size(const Matrix& other, const std::string& op) const;
  void check_square(const std::string& op) const;

 public:
  Matrix() = default;
  Matrix(size_t rows, size_t cols);
  Matrix(std::vector<std::vector<bigfloat>> const& data);
  Matrix(std::string const& str);

  size_t rows() const noexcept;
  size_t cols() const noexcept;
  bigfloat& at(size_t row, size_t col);
  const bigfloat& at(size_t row, size_t col) const;

  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);
  Matrix& operator*=(const bigfloat& scalar);
  Matrix& operator*=(const Matrix& other);

  Matrix operator+(const Matrix& other) const;
  Matrix operator-(const Matrix& other) const;
  Matrix operator*(const bigfloat& scalar) const;
  Matrix operator*(const Matrix& other) const;

  bool operator==(const Matrix& other) const;
  bool operator!=(const Matrix& other) const;

  bigfloat determinant() const;
  Matrix inverse() const;

  std::vector<bigfloat> solve_gauss(std::vector<bigfloat> const& b) const;
  std::vector<bigfloat> solve_gauss_jordan(
      std::vector<bigfloat> const& b) const;

  std::vector<bigfloat> eigenvalues() const;
  std::vector<std::vector<bigfloat>> eigenvectors() const;

  size_t rank() const;

  static size_t span_dimension(
      const std::vector<std::vector<bigfloat>>& vectors);
  static bool is_in_span(const std::vector<std::vector<bigfloat>>& basis,
                         const std::vector<bigfloat>& vector);

  std::string to_string() const;
  std::string to_latex() const;
};
