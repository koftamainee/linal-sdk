#include "matrix.h"

#include <exception>
#include <sstream>
#include <stdexcept>

void Matrix::check_same_size(const Matrix& other, const std::string& op) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::runtime_error("Matrix size mismatch in operation: " + op);
  }
}

void Matrix::check_square(const std::string& op) const {
  if (rows_ != cols_) {
    throw std::runtime_error("Matrix must be square for operation: " + op);
  }
}

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows, std::vector<bigfloat>(cols, 0)) {}

Matrix::Matrix(const std::vector<std::vector<bigfloat>>& data)
    : rows_(data.size()),
      cols_(data.empty() ? 0 : data[0].size()),
      data_(data) {
  for (const auto& row : data) {
    if (row.size() != cols_) {
      throw std::runtime_error("Inconsistent row sizes in matrix");
    }
  }
}

Matrix::Matrix(const std::string& str) {
  std::istringstream ss(str);
  std::vector<std::vector<bigfloat>> result;
  std::string token;
  while (std::getline(ss, token, ')')) {
    size_t start = token.find('(');
    if (start == std::string::npos) {
      continue;
    }
    std::istringstream row_stream(token.substr(start + 1));
    std::vector<bigfloat> row;
    bigfloat value;
    while (row_stream >> value) {
      row.push_back(value);
    }
    if (!row.empty()) {
      result.push_back(row);
    }
  }

  if (result.empty()) {
    throw std::runtime_error("Failed to parse matrix from string");
  }

  size_t cols = result[0].size();
  for (const auto& row : result) {
    if (row.size() != cols) {
      throw std::runtime_error("Inconsistent row sizes in parsed matrix");
    }
  }

  rows_ = result.size();
  cols_ = cols;
  data_ = std::move(result);
}

size_t Matrix::rows() const noexcept { return rows_; }
size_t Matrix::cols() const noexcept { return cols_; }

bigfloat& Matrix::at(size_t row, size_t col) { return data_.at(row).at(col); }
const bigfloat& Matrix::at(size_t row, size_t col) const {
  return data_.at(row).at(col);
}

Matrix& Matrix::operator+=(const Matrix& other) {
  check_same_size(other, "+=");
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      data_[i][j] += other.data_[i][j];
    }
  }
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  check_same_size(other, "-=");
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      data_[i][j] -= other.data_[i][j];
    }
  }
  return *this;
}

Matrix& Matrix::operator*=(const bigfloat& scalar) {
  for (auto& row : data_) {
    for (auto& val : row) {
      val *= scalar;
    }
  }
  return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
  *this = *this * other;
  return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
  Matrix result = *this;
  result += other;
  return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  Matrix result = *this;
  result -= other;
  return result;
}

Matrix Matrix::operator*(const bigfloat& scalar) const {
  Matrix result = *this;
  result *= scalar;
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  if (cols_ != other.rows_) {
    throw std::runtime_error("Matrix multiplication dimension mismatch");
  }

  Matrix result(rows_, other.cols_);
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < other.cols_; ++j) {
      for (size_t k = 0; k < cols_; ++k) {
        result.data_[i][j] += data_[i][k] * other.data_[k][j];
      }
    }
  }
  return result;
}

bool Matrix::operator==(const Matrix& other) const {
  return data_ == other.data_;
}

bool Matrix::operator!=(const Matrix& other) const { return !(*this == other); }

std::string Matrix::to_string() const {
  std::string result;
  for (const auto& row : data_) {
    result += "(";
    for (size_t j = 0; j < row.size(); ++j) {
      result += row[j].to_decimal();
      if (j + 1 < row.size()) {
        result += " ";
      }
    }
    result += ") ";
  }
  return result;
}

std::string Matrix::to_latex() const {
  std::string result = "\\begin{pmatrix}";
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result += data_[i][j].to_decimal();
      if (j + 1 < cols_) {
        result += " & ";
      }
    }
    if (i + 1 < rows_) {
      result += " \\\\ ";
    }
  }
  result += "\\end{pmatrix}";
  return result;
}

bigfloat Matrix::determinant() const {
  check_square("determinant");
  size_t n = rows_;
  Matrix temp = *this;
  bigfloat det = 1;
  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && temp.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      return 0;
    }

    if (pivot != i) {
      std::swap(temp.data_[i], temp.data_[pivot]);
      det = -det;
    }

    det *= temp.at(i, i);
    for (size_t j = i + 1; j < n; ++j) {
      bigfloat factor = temp.at(j, i) / temp.at(i, i);
      for (size_t k = i; k < n; ++k) {
        temp.at(j, k) -= factor * temp.at(i, k);
      }
    }
  }
  return det;
}

Matrix Matrix::inverse() const {
  check_square("inverse");
  size_t n = rows_;
  Matrix a = *this;
  Matrix inv(n, n);
  for (size_t i = 0; i < n; ++i) {
    inv.at(i, i) = 1;
  }

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      throw std::runtime_error("Singular matrix");
    }

    if (pivot != i) {
      std::swap(a.data_[i], a.data_[pivot]);
      std::swap(inv.data_[i], inv.data_[pivot]);
    }

    bigfloat div = a.at(i, i);
    for (size_t j = 0; j < n; ++j) {
      a.at(i, j) /= div;
      inv.at(i, j) /= div;
    }

    for (size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      bigfloat factor = a.at(j, i);
      for (size_t k = 0; k < n; ++k) {
        a.at(j, k) -= factor * a.at(i, k);
        inv.at(j, k) -= factor * inv.at(i, k);
      }
    }
  }

  return inv;
}

std::vector<bigfloat> Matrix::solve_gauss(
    std::vector<bigfloat> const& b) const {
  check_square("solve_gauss");
  size_t n = rows_;
  Matrix a = *this;
  std::vector<bigfloat> x = b;

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      throw std::runtime_error("No unique solution");
    }

    std::swap(a.data_[i], a.data_[pivot]);
    std::swap(x[i], x[pivot]);

    for (size_t j = i + 1; j < n; ++j) {
      bigfloat factor = a.at(j, i) / a.at(i, i);
      for (size_t k = i; k < n; ++k) {
        a.at(j, k) -= factor * a.at(i, k);
      }
      x[j] -= factor * x[i];
    }
  }

  std::vector<bigfloat> result(n);
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    result[i] = x[i];
    for (size_t j = i + 1; j < n; ++j) {
      result[i] -= a.at(i, j) * result[j];
    }
    result[i] /= a.at(i, i);
  }

  return result;
}

std::vector<bigfloat> Matrix::solve_gauss_jordan(
    std::vector<bigfloat> const& b) const {
  check_square("solve_gauss_jordan");
  size_t n = rows_;
  Matrix a = *this;
  std::vector<bigfloat> x = b;

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      throw std::runtime_error("No unique solution");
    }

    std::swap(a.data_[i], a.data_[pivot]);
    std::swap(x[i], x[pivot]);

    bigfloat div = a.at(i, i);
    for (size_t j = 0; j < n; ++j) {
      a.at(i, j) /= div;
    }
    x[i] /= div;

    for (size_t j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      bigfloat factor = a.at(j, i);
      for (size_t k = 0; k < n; ++k) {
        a.at(j, k) -= factor * a.at(i, k);
      }
      x[j] -= factor * x[i];
    }
  }

  return x;
}

size_t Matrix::rank() const {
  Matrix temp = *this;
  size_t rank = 0;
  size_t m = rows_;
  size_t n = cols_;

  for (size_t col = 0, row = 0; col < n && row < m; ++col) {
    size_t sel = row;
    for (size_t i = row + 1; i < m; ++i) {
      if ((temp.at(i, col)).abs() > (temp.at(sel, col)).abs()) {
        sel = i;
      }
    }

    if (temp.at(sel, col) == 0) {
      continue;
    }

    std::swap(temp.data_[row], temp.data_[sel]);

    for (size_t i = row + 1; i < m; ++i) {
      bigfloat factor = temp.at(i, col) / temp.at(row, col);
      for (size_t j = col; j < n; ++j) {
        temp.at(i, j) -= factor * temp.at(row, j);
      }
    }
    ++rank;
    ++row;
  }

  return rank;
}

std::vector<bigfloat> Matrix::eigenvalues() const {
  throw std::runtime_error("Eigenvalue computation not implemented");
}

std::vector<std::vector<bigfloat>> Matrix::eigenvectors() const {
  throw std::runtime_error("Eigenvector computation not implemented");
}

size_t Matrix::span_dimension(
    const std::vector<std::vector<bigfloat>>& vectors) {
  Matrix m(vectors.size(), vectors[0].size());
  for (size_t i = 0; i < vectors.size(); ++i) {
    m.data_[i] = vectors[i];
  }
  return m.rank();
}

bool Matrix::is_in_span(const std::vector<std::vector<bigfloat>>& basis,
                        const std::vector<bigfloat>& vector) {
  Matrix m(basis.size(), basis[0].size());
  for (size_t i = 0; i < basis.size(); ++i) {
    m.data_[i] = basis[i];
  }
  try {
    m.solve_gauss(vector);
    return true;
  } catch (std::exception const& e) {
    return false;
  }
}
