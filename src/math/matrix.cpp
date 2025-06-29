#include "matrix.h"

#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

#include "latex_writter.h"
#include "vector.h"

namespace {

std::string join_decimal_latex(const std::vector<bigfloat>& vec) {
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += vec[i].to_decimal();
  }
  return result;
}
}  // namespace

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
  std::string clean_str = str;

  if (!clean_str.empty() && clean_str.front() == '[') {
    clean_str.erase(0, 1);
  }
  if (!clean_str.empty() && clean_str.back() == ']') {
    clean_str.pop_back();
  }

  std::vector<std::vector<bigfloat>> result;
  std::istringstream stream(clean_str);
  std::string token;

  while (std::getline(stream, token, ')')) {
    size_t start = token.find('(');
    if (start == std::string::npos) {
      continue;
    }

    std::string row_str = token.substr(start + 1);
    std::istringstream row_stream(row_str);
    std::vector<bigfloat> row;
    std::string input;
    while (row_stream >> input) {
      row.emplace_back(input);
    }

    if (!row.empty()) {
      result.push_back(std::move(row));
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
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix addition formula",
      R"(\text{For all } i,j: \quad C_{ij} = A_{ij} + B_{ij})");

  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      writter.add_solution_step(
          "Adding elements at position (" + std::to_string(i) + "," +
              std::to_string(j) + ")",
          data_[i][j].to_decimal() + " + " + other.data_[i][j].to_decimal() +
              " = " + (data_[i][j] + other.data_[i][j]).to_decimal());
      data_[i][j] += other.data_[i][j];
    }
  }
  return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix addition operation",
      R"(\text{Computing } A + B \text{ using compound assignment})");

  Matrix result = *this;
  result += other;
  return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  check_same_size(other, "-=");
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix subtraction formula",
      R"(\text{For all } i,j: \quad C_{ij} = A_{ij} - B_{ij})");

  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      writter.add_solution_step(
          "Subtracting elements at position (" + std::to_string(i) + "," +
              std::to_string(j) + ")",
          data_[i][j].to_decimal() + " - " + other.data_[i][j].to_decimal() +
              " = " + (data_[i][j] - other.data_[i][j]).to_decimal());
      data_[i][j] -= other.data_[i][j];
    }
  }
  return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix subtraction operation",
      R"(\text{Computing } A - B \text{ using compound assignment})");

  Matrix result = *this;
  result -= other;
  return result;
}

Matrix& Matrix::operator*=(const bigfloat& scalar) {
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Scalar multiplication formula",
      R"(\text{For all } i,j: \quad C_{ij} = A_{ij} \times \lambda)");

  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      writter.add_solution_step(
          "Multiplying element at position (" + std::to_string(i) + "," +
              std::to_string(j) + ")",
          data_[i][j].to_decimal() + " \\times " + scalar.to_decimal() + " = " +
              (data_[i][j] * scalar).to_decimal());
      data_[i][j] *= scalar;
    }
  }
  return *this;
}

// Scalar multiplication operator* - через *=
Matrix Matrix::operator*(const bigfloat& scalar) const {
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Scalar multiplication operation",
      R"(\text{Computing } A \times \lambda \text{ using compound assignment})");

  Matrix result = *this;
  result *= scalar;
  return result;
}

Matrix& Matrix::operator*=(const Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::runtime_error("Matrix multiplication dimension mismatch");
  }

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix multiplication formula",
      R"(\text{For all } i,j: \quad C_{ij} = \sum_{k=0}^{n-1} A_{ik} \times B_{kj})");

  Matrix result(rows_, other.cols_);

  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < other.cols_; ++j) {
      std::string calculation =
          "C_{" + std::to_string(i) + "," + std::to_string(j) + "} = ";
      std::string terms;

      for (size_t k = 0; k < cols_; ++k) {
        bigfloat product = data_[i][k] * other.data_[k][j];
        result.data_[i][j] += product;

        if (k > 0) {
          terms += " + ";
        }
        terms += "(" + data_[i][k].to_decimal() + " \\times " +
                 other.data_[k][j].to_decimal() + ")";
      }

      writter.add_solution_step(
          "Computing element at position (" + std::to_string(i) + "," +
              std::to_string(j) + ")",
          calculation + terms + " = " + result.data_[i][j].to_decimal());
    }
  }

  *this = std::move(result);
  return *this;
}

Matrix Matrix::operator*(const Matrix& other) const {
  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix multiplication operation",
      R"(\text{Computing } A \times B \text{ using compound assignment})");

  Matrix result = *this;
  result *= other;
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

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix determinant calculation",
      R"(\text{Starting determinant calculation by Gaussian elimination with pivot selection.})");

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && temp.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      writter.add_solution_step(
          "Zero column found",
          R"(\text{All elements in column } )" + std::to_string(i) +
              R"( \text{ below diagonal are zero, determinant is } 0.)");
      return 0;
    }

    if (pivot != i) {
      std::swap(temp.data_[i], temp.data_[pivot]);
      det = -det;
      writter.add_solution_step(
          "Row swap",
          R"(\text{Swapped rows } )" + std::to_string(i) +
              R"(\ \text{ and } )" + std::to_string(pivot) +
              R"( \text{ to select pivot, determinant sign changed.})");
    }

    det *= temp.at(i, i);
    writter.add_solution_step(
        "Pivot element chosen",
        R"(\text{Pivot element } ( )" + std::to_string(i) + R"(,)" +
            std::to_string(i) + R"( ) = )" + temp.at(i, i).to_decimal() +
            R"(,\ \text{current determinant: } )" + det.to_decimal());

    for (size_t j = i + 1; j < n; ++j) {
      bigfloat factor = temp.at(j, i) / temp.at(i, i);
      writter.add_solution_step(
          "Eliminating element",
          R"(\text{Calculating factor for row } )" + std::to_string(j) +
              R"(:\ )" + temp.at(j, i).to_decimal() + R"( \div )" +
              temp.at(i, i).to_decimal() + R"( = )" + factor.to_decimal());

      for (size_t k = i; k < n; ++k) {
        bigfloat old_value = temp.at(j, k);
        temp.at(j, k) -= factor * temp.at(i, k);

        writter.add_solution_step("Updating matrix element",
                                  R"(\text{temp}( )" + std::to_string(j) +
                                      R"(,)" + std::to_string(k) + R"( ): )" +
                                      old_value.to_decimal() + R"( - )" +
                                      factor.to_decimal() + R"( \times )" +
                                      temp.at(i, k).to_decimal() + R"( = )" +
                                      temp.at(j, k).to_decimal());
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

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Matrix inversion",
      R"(\text{Starting matrix inversion using Gauss-Jordan elimination. The identity matrix is augmented and operations are applied to transform the original matrix to the identity, and the identity to the inverse.})");

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }
    if (pivot == n) {
      writter.add_solution_step(
          "Singular matrix",
          R"(\text{Column } )" + std::to_string(i) +
              R"( \text{ has no non-zero pivot, the matrix is singular.})");
      throw std::runtime_error("Singular matrix");
    }

    if (pivot != i) {
      std::swap(a.data_[i], a.data_[pivot]);
      std::swap(inv.data_[i], inv.data_[pivot]);
      writter.add_solution_step(
          "Row swap", R"(\text{Swapped rows } )" + std::to_string(i) +
                          R"( and )" + std::to_string(pivot) +
                          R"( \text{ in both matrices to select pivot.})");
    }

    bigfloat div = a.at(i, i);
    writter.add_solution_step("Normalize pivot row",
                              R"(\text{Dividing row } )" + std::to_string(i) +
                                  R"( \text{ by pivot } )" + div.to_decimal() +
                                  R"( \text{ to make pivot } 1.)");

    for (size_t j = 0; j < n; ++j) {
      a.at(i, j) /= div;
      inv.at(i, j) /= div;
    }

    for (size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }

      bigfloat factor = a.at(j, i);
      writter.add_solution_step(
          "Eliminate column entry",
          R"(\text{Row } )" + std::to_string(j) + R"( \leftarrow \text{Row })" +
              std::to_string(j) + R"( - )" + factor.to_decimal() +
              R"( \times \text{Row } )" + std::to_string(i) + R"(.)");

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

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Solving Ax = b via Gaussian elimination",
      R"(\text{Solving the system of linear equations using Gaussian elimination with back substitution.})");

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }

    if (pivot == n) {
      writter.add_solution_step(
          "No unique solution",
          R"(\text{Column } i = )" + std::to_string(i) +
              R"( \text{ has all zeros below the diagonal, no unique solution exists.})");
      throw std::runtime_error("No unique solution");
    }

    if (pivot != i) {
      std::swap(a.data_[i], a.data_[pivot]);
      std::swap(x[i], x[pivot]);
      writter.add_solution_step(
          "Row swap", R"(\text{Swapped rows } i = )" + std::to_string(i) +
                          R"( \text{ and } pivot = )" + std::to_string(pivot) +
                          R"( \text{ to bring pivot into position.})");
    }

    for (size_t j = i + 1; j < n; ++j) {
      bigfloat factor = a.at(j, i) / a.at(i, i);
      writter.add_solution_step(
          "Eliminating element",
          R"(\text{Row } j = )" + std::to_string(j) +
              R"( \leftarrow \text{Row } j - )" + factor.to_decimal() +
              R"( \times \text{Row } i = )" + std::to_string(i) + R"(.)");

      for (size_t k = i; k < n; ++k) {
        bigfloat old_value = a.at(j, k);
        a.at(j, k) -= factor * a.at(i, k);
        writter.add_solution_step(
            "Updating matrix element",
            R"(a_{)" + std::to_string(j) + R"(,)" + std::to_string(k) +
                R"(} = )" + old_value.to_decimal() + R"( - )" +
                factor.to_decimal() + R"( \times )" + a.at(i, k).to_decimal() +
                R"( = )" + a.at(j, k).to_decimal());
      }

      bigfloat old_rhs = x[j];
      x[j] -= factor * x[i];
      writter.add_solution_step(
          "Updating RHS", R"(b_{)" + std::to_string(j) + R"(} = )" +
                              old_rhs.to_decimal() + R"( - )" +
                              factor.to_decimal() + R"( \times )" +
                              x[i].to_decimal() + R"( = )" + x[j].to_decimal());
    }
  }

  writter.add_solution_step("Back substitution",
                            R"(\text{Starting back substitution.})");

  std::vector<bigfloat> result(n);
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    bigfloat sum = x[i];
    for (size_t j = i + 1; j < n; ++j) {
      sum -= a.at(i, j) * result[j];
    }

    result[i] = sum / a.at(i, i);
    writter.add_solution_step("Back substitution step",
                              R"(x_{)" + bigfloat(i).to_decimal() +
                                  R"(} = \frac{)" + sum.to_decimal() + R"(}{)" +
                                  a.at(i, i).to_decimal() + R"(} = )" +
                                  result[i].to_decimal());
  }

  writter.add_solution_step(
      "Final result",
      R"(\text{System solved, final solution vector obtained.})");
  return result;
}

std::vector<bigfloat> Matrix::solve_gauss_jordan(
    std::vector<bigfloat> const& b) const {
  check_square("solve_gauss_jordan");
  size_t n = rows_;
  Matrix a = *this;
  std::vector<bigfloat> x = b;

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Solving Ax = b via Gauss-Jordan elimination",
      R"(\text{Solving the system of linear equations using the Gauss-Jordan method with full row reduction.})");

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    while (pivot < n && a.at(pivot, i) == 0) {
      ++pivot;
    }

    if (pivot == n) {
      writter.add_solution_step(
          "No unique solution",
          R"(\text{Column } )" + std::to_string(i) +
              R"( \text{ has all zeros, no unique solution exists.})");
      throw std::runtime_error("No unique solution");
    }

    if (pivot != i) {
      std::swap(a.data_[i], a.data_[pivot]);
      std::swap(x[i], x[pivot]);
      writter.add_solution_step(
          "Row swap", R"(\text{Swapped rows } )" + std::to_string(i) +
                          R"( \text{ and } )" + std::to_string(pivot) +
                          R"( \text{ to bring pivot into position.})");
    }

    bigfloat div = a.at(i, i);
    writter.add_solution_step(
        "Normalize pivot row",
        R"(\text{Dividing row } )" + std::to_string(i) +
            R"( \text{ by pivot } )" + div.to_decimal() +
            R"( \text{ to make leading coefficient } 1.)");

    for (size_t j = 0; j < n; ++j) {
      bigfloat old_val = a.at(i, j);
      a.at(i, j) /= div;
      writter.add_solution_step(
          "Normalizing matrix element",
          R"(a( )" + std::to_string(i) + R"(,)" + std::to_string(j) +
              R"( ) = )" + old_val.to_decimal() + R"( \div )" +
              div.to_decimal() + R"( = )" + a.at(i, j).to_decimal());
    }

    bigfloat old_rhs = x[i];
    x[i] /= div;
    writter.add_solution_step(
        "Normalizing RHS", R"(x[)" + std::to_string(i) + R"(] = )" +
                               old_rhs.to_decimal() + R"( \div )" +
                               div.to_decimal() + R"( = )" + x[i].to_decimal());

    for (size_t j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }

      bigfloat factor = a.at(j, i);
      writter.add_solution_step(
          "Eliminate element",
          R"(\text{Row } )" + std::to_string(j) + R"( \leftarrow \text{Row })" +
              std::to_string(j) + R"( - )" + factor.to_decimal() +
              R"( \times \text{Row } )" + std::to_string(i) + R"(.)");

      for (size_t k = 0; k < n; ++k) {
        bigfloat old_value = a.at(j, k);
        a.at(j, k) -= factor * a.at(i, k);
        writter.add_solution_step(
            "Updating matrix element",
            R"(a()" + std::to_string(j) + R"(,)" + std::to_string(k) +
                R"( ) = )" + old_value.to_decimal() + R"( - )" +
                factor.to_decimal() + R"( \times )" + a.at(i, k).to_decimal() +
                R"( = )" + a.at(j, k).to_decimal());
      }

      bigfloat old_rhs_j = x[j];
      x[j] -= factor * x[i];
      writter.add_solution_step(
          "Updating RHS", R"(x[)" + std::to_string(j) + R"(] = )" +
                              old_rhs_j.to_decimal() + R"( - )" +
                              factor.to_decimal() + R"( \times )" +
                              x[i].to_decimal() + R"( = )" + x[j].to_decimal());
    }
  }

  return x;
}

size_t Matrix::rank() const {
  Matrix temp = *this;
  size_t rank = 0;
  size_t m = rows_;
  size_t n = cols_;

  auto& writter = LatexWriter::get_instance();
  writter.add_solution_step(
      "Rank calculation",
      R"(\text{Starting rank calculation by Gaussian elimination.})");

  for (size_t col = 0, row = 0; col < n && row < m; ++col) {
    size_t sel = row;
    for (size_t i = row + 1; i < m; ++i) {
      if ((temp.at(i, col)).abs() > (temp.at(sel, col)).abs()) {
        sel = i;
      }
    }

    if (temp.at(sel, col) == 0) {
      writter.add_solution_step(
          "Skipping column",
          R"(\text{All elements below row } )" + std::to_string(row) +
              R"( \text{ in column } )" + std::to_string(col) +
              R"( \text{ are zero, skipping.})");
      continue;
    }

    if (sel != row) {
      std::swap(temp.data_[row], temp.data_[sel]);
      writter.add_solution_step(
          "Row swap", R"(\text{Swapped rows } )" + std::to_string(row) +
                          R"( \text{ and } )" + std::to_string(sel) +
                          R"( \text{ to bring pivot into position.})");
    }

    writter.add_solution_step(
        "Pivot selected",
        R"(\text{Pivot at position } ( )" + std::to_string(row) + R"(,)" +
            std::to_string(col) + R"( ) = )" + temp.at(row, col).to_decimal());

    for (size_t i = row + 1; i < m; ++i) {
      bigfloat factor = temp.at(i, col) / temp.at(row, col);
      writter.add_solution_step("Eliminating row",
                                R"(\text{Row } )" + std::to_string(i) +
                                    R"(\ \text{ -= } )" + factor.to_decimal() +
                                    R"( \times \text{row } )" +
                                    std::to_string(row));

      for (size_t j = col; j < n; ++j) {
        bigfloat before = temp.at(i, j);
        temp.at(i, j) -= factor * temp.at(row, j);

        writter.add_solution_step("Updating element",
                                  R"(\text{temp}( )" + std::to_string(i) +
                                      R"(,)" + std::to_string(j) + R"( ) = )" +
                                      before.to_decimal() + R"( - )" +
                                      factor.to_decimal() + R"( \times )" +
                                      temp.at(row, j).to_decimal() + R"( = )" +
                                      temp.at(i, j).to_decimal());
      }
    }

    ++rank;
    ++row;

    writter.add_solution_step(
        "Rank incremented", R"(\text{Current rank: } )" + std::to_string(rank));
  }

  return rank;
}

std::vector<bigfloat> Matrix::eigenvalues(bigfloat const& EPS) const {
  check_square("eigenvalues");
  auto& writter = LatexWriter::get_instance();

  const size_t n = rows_;
  Matrix A = *this;
  const size_t MAX_ITER = 100;

  writter.add_solution_step("Eigenvalues",
                            R"(Initial\ matrix:\\ )" + A.to_latex());

  for (size_t iter = 0; iter < MAX_ITER; ++iter) {
    // QR decomposition: A = Q * R
    std::vector<Vector> Q_vectors;
    for (size_t i = 0; i < n; ++i) {
      Vector v(n);
      for (size_t j = 0; j < n; ++j) {
        v[j] = A.at(j, i);
      }
      Q_vectors.push_back(v);
    }

    Q_vectors = Vector::gram_schmidt_process(Q_vectors, EPS);

    Matrix Q(n, n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        Q.at(j, i) = Q_vectors[i][j];
      }
    }

    Matrix R = Q.transpose() * A;
    A = R * Q;

    writter.add_solution_step(
        "Eigenvalues - Iteration " + std::to_string(iter + 1),
        "Q:\\\\ " + Q.to_latex() + R"(\\ R:\\ )" + R.to_latex() +
            R"(\\ A_{next} = R \cdot Q:\\ )" + A.to_latex());

    bigfloat off_diagonal = 0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          off_diagonal += A.at(i, j).abs();
        }
      }
    }

    if (off_diagonal < EPS) {
      writter.add_solution_step("Eigenvalues",
                                R"(Converged\ matrix:\\ )" + A.to_latex());
      break;
    }
  }

  std::vector<bigfloat> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = A.at(i, i);
  }

  return result;
}

std::vector<Vector> Matrix::eigenvectors(bigfloat const& EPS) const {
  std::vector<bigfloat> eigvals = eigenvalues();
  std::vector<Vector> eigvecs;
  const size_t n = rows_;
  const size_t MAX_ITER = 50;

  for (const bigfloat& lambda : eigvals) {
    Matrix shifted = *this;
    for (size_t i = 0; i < n; ++i) {
      shifted.at(i, i) -= lambda;
    }

    Vector x(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = bigfloat(1);
    }

    for (size_t iter = 0; iter < MAX_ITER; ++iter) {
      std::vector<bigfloat> b(n);
      for (size_t i = 0; i < n; ++i) {
        b[i] = x[i];
      }
      b = shifted.solve_gauss_jordan(b);
      Vector new_x(b);

      new_x = new_x.normalize(EPS);

      if ((new_x - x).norm() < EPS) {
        break;
      }
      x = new_x;
    }

    eigvecs.push_back(x.normalize(EPS));
  }

  return eigvecs;
}

size_t Matrix::span_dimension(
    const std::vector<std::vector<bigfloat>>& vectors) {
  auto& writter = LatexWriter::get_instance();

  writter.add_solution_step(
      "Span dimension",
      R"(\text{Constructing a matrix from input vectors to compute the dimension of their span.})");

  Matrix m(vectors.size(), vectors[0].size());
  for (size_t i = 0; i < vectors.size(); ++i) {
    m.data_[i] = vectors[i];
    writter.add_solution_step(
        "Inserting vector", R"(\text{Row } )" + std::to_string(i) +
                                R"( = \left[)" +
                                join_decimal_latex(vectors[i]) + R"(\right])");
  }

  return m.rank();
}

bool Matrix::is_in_span(const std::vector<std::vector<bigfloat>>& basis,
                        const std::vector<bigfloat>& vector) {
  auto& writter = LatexWriter::get_instance();

  writter.add_solution_step(
      "Span membership check",
      R"(\text{Checking if the given vector is in the span of the basis.})");

  Matrix m(basis.size(), basis[0].size());
  for (size_t i = 0; i < basis.size(); ++i) {
    m.data_[i] = basis[i];
    writter.add_solution_step("Inserting basis vector",
                              R"(\text{Row } )" + std::to_string(i) +
                                  R"( = \left[)" +
                                  join_decimal_latex(basis[i]) + R"(\right])");
  }

  writter.add_solution_step("Vector to test",
                            R"(\text{Target vector } = \left[)" +
                                join_decimal_latex(vector) + R"(\right])");

  try {
    m.solve_gauss(vector);

    return true;
  } catch (std::exception const& e) {
    return false;
  }
}

Matrix Matrix::transpose() const {
  Matrix result(cols_, rows_);
  auto& writter = LatexWriter::get_instance();

  writter.add_solution_step("Transpose",
                            R"(Original\ matrix:\\ )" + to_latex());

  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result.at(j, i) = at(i, j);
    }
  }

  writter.add_solution_step("Transpose",
                            R"(Transposed\ matrix:\\ )" + result.to_latex());

  return result;
}
