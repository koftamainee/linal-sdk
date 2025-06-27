#include "vector.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bigfloat.h"

// Private methods
void Vector::check_dimension(size_t expected,
                             const std::string& operation) const {
  if (dimension() != expected) {
    throw std::invalid_argument(
        "Vector::" + operation + " - dimension mismatch: " +
        std::to_string(dimension()) + " != " + std::to_string(expected));
  }
}

void Vector::check_non_zero() const {
  if (is_zero()) {
    throw std::domain_error("Vector operation on zero vector");
  }
}

// Constructors
Vector::Vector(size_t dimension) : components_(dimension) {}

Vector::Vector(const std::vector<bigfloat>& components)
    : components_(components) {}

Vector::Vector(std::initializer_list<bigfloat> init) : components_(init) {}

// Accessors
size_t Vector::dimension() const noexcept { return components_.size(); }

const bigfloat& Vector::operator[](size_t index) const {
  if (index >= dimension()) {
    throw std::out_of_range("Vector index out of range");
  }
  return components_[index];
}

bigfloat& Vector::operator[](size_t index) {
  if (index >= dimension()) {
    throw std::out_of_range("Vector index out of range");
  }
  return components_[index];
}

Vector& Vector::operator+=(const Vector& other) & {
  check_dimension(other.dimension(), "operator+=");

  writter->add_solution_step(
      "Vector addition formula",
      R"((a_1, a_2, \dots, a_n) + (b_1, b_2, \dots, b_n) = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n))");

  for (size_t i = 0; i < dimension(); ++i) {
    std::string expr =
        components_[i].to_decimal() + " + " + other.components_[i].to_decimal();
    components_[i] += other.components_[i];
    expr += " = " + components_[i].to_decimal();
    writter->add_solution_step("Component " + std::to_string(i + 1), expr);
  }

  return *this;
}

Vector& Vector::operator-=(const Vector& other) & {
  check_dimension(other.dimension(), "operator-=");

  writter->add_solution_step(
      "Vector subtraction formula",
      R"((a_1, a_2, \dots, a_n) - (b_1, b_2, \dots, b_n) = (a_1 - b_1, a_2 - b_2, \dots, a_n - b_n))");

  for (size_t i = 0; i < dimension(); ++i) {
    std::string expr =
        components_[i].to_decimal() + " - " + other.components_[i].to_decimal();
    components_[i] -= other.components_[i];
    expr += " = " + components_[i].to_decimal();
    writter->add_solution_step("Component " + std::to_string(i + 1), expr);
  }

  return *this;
}

Vector& Vector::operator*=(const bigfloat& scalar) & {
  writter->add_solution_step(
      "Scalar multiplication formula",
      R"(\lambda \cdot (a_1, a_2, \dots, a_n) = (\lambda a_1, \lambda a_2, \dots, \lambda a_n))");

  std::string scalar_str = scalar.to_decimal();

  for (size_t i = 0; i < dimension(); ++i) {
    std::string expr = scalar_str + " \\cdot " + components_[i].to_decimal();
    components_[i] *= scalar;
    expr += " = " + components_[i].to_decimal();
    writter->add_solution_step("Component " + std::to_string(i + 1), expr);
  }

  return *this;
}

Vector& Vector::operator/=(const bigfloat& scalar) & {
  if (scalar == 0) {
    throw std::domain_error("Division by zero");
  }
  for (auto& component : components_) {
    component /= scalar;
  }
  return *this;
}

Vector Vector::operator+() const { return *this; }

Vector Vector::operator-() const {
  auto copy = *this;
  return copy *= 1;
}

Vector operator+(Vector first, const Vector& second) { return first += second; }

Vector operator-(Vector first, const Vector& second) { return first -= second; }

Vector operator*(Vector vec, const bigfloat& scalar) { return vec *= scalar; }

Vector operator*(const bigfloat& scalar, Vector vec) { return vec *= scalar; }

Vector operator/(Vector vec, const bigfloat& scalar) { return vec /= scalar; }

bool operator==(const Vector& first, const Vector& second) {
  if (first.dimension() != second.dimension()) {
    return false;
  }
  for (size_t i = 0; i < first.dimension(); ++i) {
    if (first[i] != second[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const Vector& first, const Vector& second) {
  return !(first == second);
}

bigfloat Vector::dot(const Vector& other) const {
  check_dimension(other.dimension(), "dot");
  bigfloat result = 0;

  std::string formula;
  for (size_t i = 0; i < dimension(); ++i) {
    formula += components_[i].to_decimal();
    formula += R"( \cdot )";
    formula += other.components_[i].to_decimal();

    if (i + 1 < dimension()) {
      formula += " + ";
    }
  }

  for (size_t i = 0; i < dimension(); ++i) {
    result += components_[i] * other.components_[i];
  }

  writter->add_solution_step("Dot product formula",
                             formula + R"( = )" + result.to_decimal());

  return result;
}

bigfloat Vector::norm() const {
  check_dimension(components_.size(), "norm");

  writter->add_solution_step("Norm formula",
                             R"(\|\vec{v}\| = \sqrt{\vec{v} \cdot \vec{v}})");

  bigfloat dot_res = dot(*this);

  std::string step = R"(\|\vec{v}\| = \sqrt{)" + dot_res.to_decimal() + "}";

  writter->add_solution_step("Norm calculation", step);

  bigfloat result = sqrt(dot_res);

  std::string final_step = step + " = " + result.to_decimal();
  writter->add_solution_step("Norm result", final_step);

  return result;
}

Vector Vector::normalize(const bigfloat& EPS) const {
  check_non_zero();
  const bigfloat n = norm();
  if (n < EPS) {
    throw std::domain_error("Cannot normalize zero vector");
  }
  return *this / n;
}

Vector Vector::cross_3d(const Vector& other) const {
  check_dimension(3, "cross_3d");
  other.check_dimension(3, "cross_3d");

  writter->add_solution_step("Cross product formula",
                             R"(\begin{pmatrix}
a_y b_z - a_z b_y \\ 
a_z b_x - a_x b_z \\ 
a_x b_y - a_y b_x
\end{pmatrix})");

  bigfloat x = components_[1] * other.components_[2] -
               components_[2] * other.components_[1];
  writter->add_solution_step("X component calculation",
                             components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( - )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( = )" +
                                 x.to_decimal());

  bigfloat y = components_[2] * other.components_[0] -
               components_[0] * other.components_[2];
  writter->add_solution_step("Y component calculation",
                             components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( - )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( = )" +
                                 y.to_decimal());

  bigfloat z = components_[0] * other.components_[1] -
               components_[1] * other.components_[0];
  writter->add_solution_step("Z component calculation",
                             components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( - )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( = )" +
                                 z.to_decimal());

  return Vector{x, y, z};
}

Vector Vector::cross_7d(const Vector& other) const {
  check_dimension(7, "cross_7d");
  other.check_dimension(7, "cross_7d");

  writter->add_solution_step("7D Cross Product Formula",
                             R"(\begin{pmatrix}
        a_1b_3 - a_3b_1 + a_2b_6 - a_6b_2 + a_4b_5 - a_5b_4 \\
        a_2b_4 - a_4b_2 + a_3b_0 - a_0b_3 + a_5b_6 - a_6b_5 \\
        a_3b_5 - a_5b_3 + a_4b_1 - a_1b_4 + a_6b_0 - a_0b_6 \\
        a_4b_6 - a_6b_4 + a_5b_2 - a_2b_5 + a_0b_1 - a_1b_0 \\
        a_5b_0 - a_0b_5 + a_6b_3 - a_3b_6 + a_1b_2 - a_2b_1 \\
        a_6b_1 - a_1b_6 + a_0b_4 - a_4b_0 + a_2b_3 - a_3b_2 \\
        a_0b_2 - a_2b_0 + a_1b_5 - a_5b_1 + a_3b_4 - a_4b_3
        \end{pmatrix})");

  std::vector<bigfloat> results(7);

  results[0] = components_[1] * other.components_[3] -
               components_[3] * other.components_[1] +
               components_[2] * other.components_[6] -
               components_[6] * other.components_[2] +
               components_[4] * other.components_[5] -
               components_[5] * other.components_[4];
  writter->add_solution_step("Component 0 Calculation",
                             components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( - )" +
                                 components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( + )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( - )" +
                                 components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( + )" +
                                 components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( - )" +
                                 components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( = )" +
                                 results[0].to_decimal());

  results[1] = components_[2] * other.components_[4] -
               components_[4] * other.components_[2] +
               components_[3] * other.components_[0] -
               components_[0] * other.components_[3] +
               components_[5] * other.components_[6] -
               components_[6] * other.components_[5];
  writter->add_solution_step("Component 1 Calculation",
                             components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( - )" +
                                 components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( + )" +
                                 components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( - )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( + )" +
                                 components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( - )" +
                                 components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( = )" +
                                 results[1].to_decimal());

  results[2] = components_[3] * other.components_[5] -
               components_[5] * other.components_[3] +
               components_[4] * other.components_[1] -
               components_[1] * other.components_[4] +
               components_[6] * other.components_[0] -
               components_[0] * other.components_[6];
  writter->add_solution_step("Component 2 Calculation",
                             components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( - )" +
                                 components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( + )" +
                                 components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( - )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( + )" +
                                 components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( - )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( = )" +
                                 results[2].to_decimal());

  results[3] = components_[4] * other.components_[6] -
               components_[6] * other.components_[4] +
               components_[5] * other.components_[2] -
               components_[2] * other.components_[5] +
               components_[0] * other.components_[1] -
               components_[1] * other.components_[0];
  writter->add_solution_step("Component 3 Calculation",
                             components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( - )" +
                                 components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( + )" +
                                 components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( - )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( + )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( - )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( = )" +
                                 results[3].to_decimal());

  results[4] = components_[5] * other.components_[0] -
               components_[0] * other.components_[5] +
               components_[6] * other.components_[3] -
               components_[3] * other.components_[6] +
               components_[1] * other.components_[2] -
               components_[2] * other.components_[1];
  writter->add_solution_step("Component 4 Calculation",
                             components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( - )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( + )" +
                                 components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( - )" +
                                 components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( + )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( - )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( = )" +
                                 results[4].to_decimal());

  results[5] = components_[6] * other.components_[1] -
               components_[1] * other.components_[6] +
               components_[0] * other.components_[4] -
               components_[4] * other.components_[0] +
               components_[2] * other.components_[3] -
               components_[3] * other.components_[2];
  writter->add_solution_step("Component 5 Calculation",
                             components_[6].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( - )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[6].to_decimal() + R"( + )" +
                                 components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( - )" +
                                 components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( + )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( - )" +
                                 components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( = )" +
                                 results[5].to_decimal());

  results[6] = components_[0] * other.components_[2] -
               components_[2] * other.components_[0] +
               components_[1] * other.components_[5] -
               components_[5] * other.components_[1] +
               components_[3] * other.components_[4] -
               components_[4] * other.components_[3];
  writter->add_solution_step("Component 6 Calculation",
                             components_[0].to_decimal() + R"( \cdot )" +
                                 other.components_[2].to_decimal() + R"( - )" +
                                 components_[2].to_decimal() + R"( \cdot )" +
                                 other.components_[0].to_decimal() + R"( + )" +
                                 components_[1].to_decimal() + R"( \cdot )" +
                                 other.components_[5].to_decimal() + R"( - )" +
                                 components_[5].to_decimal() + R"( \cdot )" +
                                 other.components_[1].to_decimal() + R"( + )" +
                                 components_[3].to_decimal() + R"( \cdot )" +
                                 other.components_[4].to_decimal() + R"( - )" +
                                 components_[4].to_decimal() + R"( \cdot )" +
                                 other.components_[3].to_decimal() + R"( = )" +
                                 results[6].to_decimal());

  return Vector{results[0], results[1], results[2], results[3],
                results[4], results[5], results[6]};
}

// Static methods
bigfloat Vector::triple_product_3d(const Vector& a, const Vector& b,
                                   const Vector& c) {
  auto& wr = LatexWriter::get_instance();
  wr.add_solution_step("Triple product formula",
                       "(\\vec{a} \\times \\vec{b}) \\cdot "
                       "\\vec{c}");
  return a.dot(b.cross_3d(c));
}

bigfloat Vector::triple_product_7d(const Vector& a, const Vector& b,
                                   const Vector& c) {
  auto& wr = LatexWriter::get_instance();
  wr.add_solution_step("Triple product formula (7D)",
                       R"((\vec{a} \cdot (\vec{b} \times \vec{c})))");
  return a.dot(b.cross_7d(c));
}

std::vector<Vector> Vector::gram_schmidt_process(
    const std::vector<Vector>& vectors, const bigfloat& EPS) {
  auto& writter = LatexWriter::get_instance();

  if (vectors.empty()) {
    return {};
  }

  std::vector<Vector> ortho_basis;
  ortho_basis.reserve(vectors.size());

  {
    std::string desc = "Given set of vectors:";
    std::string math = "\\{ ";
    for (size_t i = 0; i < vectors.size(); ++i) {
      math +=
          "\\vec{v}_{" + std::to_string(i + 1) + "} = " + vectors[i].to_latex();
      if (i + 1 < vectors.size()) {
        math += ",\\ ";
      }
    }
    math += " \\}";
    writter.add_solution_step(desc, math);
  }

  for (size_t i = 0; i < vectors.size(); ++i) {
    Vector u = vectors[i];
    writter.add_solution_step(
        "Take vector \\(\\vec{v}_{" + std::to_string(i + 1) + "}\\):",
        u.to_latex());

    for (size_t j = 0; j < ortho_basis.size(); ++j) {
      const Vector& e = ortho_basis[j];
      bigfloat proj_coeff = vectors[i].dot(e) / e.dot(e);

      writter.add_solution_step(
          "Calculate projection coefficient of \\(\\vec{v}_{" +
              std::to_string(i + 1) +
              R"(}\) onto orthonormal vector \(\vec{u}_{)" +
              std::to_string(j + 1) + "}\\):",
          "\\frac{\\vec{v}_{" + std::to_string(i + 1) + "} \\cdot \\vec{u}_{" +
              std::to_string(j + 1) + "}}{\\vec{u}_{" + std::to_string(j + 1) +
              "} \\cdot \\vec{u}_{" + std::to_string(j + 1) +
              "}} = " + proj_coeff.to_decimal());

      Vector proj = e * proj_coeff;
      writter.add_solution_step("Projection vector:", proj.to_latex());

      u -= proj;

      writter.add_solution_step("Subtract projection from \\(\\vec{u}_{" +
                                    std::to_string(i + 1) +
                                    "}\\), resulting in:",
                                u.to_latex());
    }

    if (!u.is_zero()) {
      Vector u_norm = u.normalize(EPS);
      ortho_basis.push_back(u_norm);

      writter.add_solution_step("Normalize \\(\\vec{u}_{" +
                                    std::to_string(i + 1) +
                                    "}\\) to obtain orthonormal vector:",
                                u_norm.to_latex());
    } else {
      writter.add_solution_step(
          "Vector \\(\\vec{v}_{" + std::to_string(i + 1) +
              "}\\) is linearly dependent and is discarded.",
          "");
    }
  }

  return ortho_basis;
}

bool Vector::is_zero() const {
  for (const auto& component : components_) {
    if (component != 0) {
      return false;
    }
  }
  return true;
}

bool Vector::is_orthogonal_to(const Vector& other) const {
  return dot(other) == 0;
}

Vector Vector::zero(size_t dimension) { return Vector(dimension); }

Vector Vector::basis_vector(size_t dimension, size_t index) {
  if (index >= dimension) {
    throw std::out_of_range("Basis vector index out of range");
  }
  Vector result(dimension);
  result[index] = 1;
  return result;
}

bigfloat angle_between(const Vector& a, const Vector& b, const bigfloat& EPS) {
  a.check_dimension(b.dimension(), "angle_between");
  if (a.is_zero() || b.is_zero()) {
    throw std::domain_error("Angle with zero vector is undefined");
  }

  const bigfloat dot_product = a.dot(b);
  const bigfloat norms_product = a.norm() * b.norm();

  if (norms_product < EPS) {
    throw std::domain_error("Vectors too small for angle calculation");
  }

  const bigfloat cos_theta = dot_product / norms_product;
  return arccos(cos_theta, EPS);
}

bool are_orthogonal(const Vector& a, const Vector& b) {
  return a.is_orthogonal_to(b);
}

bool are_collinear(const Vector& a, const Vector& b) {
  if (a.dimension() != b.dimension()) {
    return false;
  }
  if (a.is_zero() || b.is_zero()) {
    return true;
  }

  const bigfloat ratio = a[0] / b[0];
  for (size_t i = 1; i < a.dimension(); ++i) {
    if (a[i] / b[i] != ratio) {
      return false;
    }
  }
  return true;
}

std::string Vector::to_string() const {
  if (components_.empty()) {
    return "[]";
  }

  std::string result = "[";
  for (size_t i = 0; i < components_.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += components_[i].to_decimal();
  }
  result += "]";
  return result;
}

std::string Vector::to_latex() const {
  if (components_.empty()) {
    return "\\begin{pmatrix}\\end{pmatrix}";
  }

  std::string result = "\\begin{pmatrix}";
  for (size_t i = 0; i < components_.size(); ++i) {
    if (i != 0) {
      result += " \\\\ ";
    }
    result += components_[i].to_decimal();
  }
  result += "\\end{pmatrix}";
  return result;
}

Vector::Vector(std::string const& str) {
  std::stringstream in(str);
  while (!in.eof()) {
    std::string num;
    in >> num;
    bigfloat number(bigint(num.c_str()));  // TODO fixme, input may be fraction
    components_.push_back(std::move(number));
  }
}
