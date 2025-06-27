#pragma once

#include <string>
#include <vector>

#include "core/bigfloat.h"
#include "latex_writter.h"

class Vector {
 private:
  std::vector<bigfloat> components_;
  LatexWriter* writter = &LatexWriter::get_instance();

  void check_dimension(size_t expected, const std::string& operation) const;
  void check_non_zero() const;

 public:
  Vector() = default;
  explicit Vector(size_t dimension);
  Vector(const std::vector<bigfloat>& components);
  Vector(std::initializer_list<bigfloat> init);
  Vector(std::string const& str);

  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) noexcept = default;
  ~Vector() = default;

  size_t dimension() const noexcept;
  const bigfloat& operator[](size_t index) const;
  bigfloat& operator[](size_t index);

  Vector& operator+=(const Vector& other) &;
  Vector& operator-=(const Vector& other) &;
  Vector& operator*=(const bigfloat& scalar) &;
  Vector& operator/=(const bigfloat& scalar) &;

  Vector operator+() const;
  Vector operator-() const;

  friend Vector operator+(Vector first, const Vector& second);
  friend Vector operator-(Vector first, const Vector& second);
  friend Vector operator*(Vector vec, const bigfloat& scalar);
  friend Vector operator*(const bigfloat& scalar, Vector vec);
  friend Vector operator/(Vector vec, const bigfloat& scalar);

  friend bool operator==(const Vector& first, const Vector& second);
  friend bool operator!=(const Vector& first, const Vector& second);

  bigfloat dot(const Vector& other) const;
  bigfloat norm() const;

  Vector normalize(const bigfloat& EPS = bigfloat::DEFAULT_EPS) const;

  Vector cross_3d(const Vector& other) const;
  Vector cross_7d(const Vector& other) const;

  static bigfloat triple_product_3d(const Vector& a, const Vector& b,
                                    const Vector& c);
  static bigfloat triple_product_7d(const Vector& a, const Vector& b,
                                    const Vector& c);

  static std::vector<Vector> gram_schmidt_process(
      const std::vector<Vector>& vectors,
      const bigfloat& EPS = bigfloat::DEFAULT_EPS);

  bool is_zero() const;
  bool is_orthogonal_to(const Vector& other) const;

  std::string to_string() const;
  std::string to_latex() const;

  static Vector zero(size_t dimension);
  static Vector basis_vector(size_t dimension, size_t index);

  friend bigfloat angle_between(const Vector& a, const Vector& b,
                                const bigfloat& EPS);
  friend bool are_orthogonal(const Vector& a, const Vector& b);
  friend bool are_collinear(const Vector& a, const Vector& b);
};

bigfloat angle_between(const Vector& a, const Vector& b,
                       const bigfloat& EPS = bigfloat::DEFAULT_EPS);
bool are_orthogonal(const Vector& a, const Vector& b);
bool are_collinear(const Vector& a, const Vector& b);
