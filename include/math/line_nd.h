#pragma once

#include <optional>
#include <string>

#include "bigfloat.h"
#include "latex_writer.h"
#include "matrix.h"
#include "vector.h"

class LineND {
 private:
  Vector point_;
  Vector direction_;
  size_t dimension_;

  LatexWriter* writer_ = &LatexWriter::get_instance();

  void normalize_direction();
  void check_valid_line() const;
  void check_dimension_match(const Vector& vec,
                             const std::string& operation) const;

 public:
  LineND() = default;

  LineND(const Vector& a, const Vector& b, bool is_two_points);

  LineND(const Matrix& A, const Vector& b);

  LineND(const std::string& str);

  LineND(const LineND&) = default;
  LineND(LineND&&) noexcept = default;
  LineND& operator=(const LineND&) = default;
  LineND& operator=(LineND&&) noexcept = default;
  ~LineND() = default;

  const Vector& point() const noexcept;
  const Vector& direction() const noexcept;
  size_t dimension() const noexcept;

  struct ParametricForm {
    Vector point;
    Vector direction;
    std::string to_string() const;
    std::string to_latex() const;
  };

  struct CanonicalForm {
    Vector point;
    Vector direction;
    std::string to_string() const;
    std::string to_latex() const;
  };

  struct SystemForm {
    Matrix A;
    Vector b;
    std::string to_string() const;
    std::string to_latex() const;
  };

  ParametricForm get_parametric_form() const;
  CanonicalForm get_canonical_form() const;
  SystemForm get_system_form() const;

  bool is_parallel(const LineND& other) const;
  bool is_perpendicular(const LineND& other) const;
  bool contains_point(const Vector& point,
                      const bigfloat& EPS = bigfloat::DEFAULT_EPS) const;

  std::optional<Vector> intersect(const LineND& other) const;

  bigfloat distance_to_point(const Vector& point) const;

  bigfloat distance_to_line(const LineND& other) const;

  Vector symmetric_point(const Vector& point) const;

  bigfloat angle_with(const LineND& other,
                      const bigfloat& EPS = bigfloat::DEFAULT_EPS) const;

  Vector project_point(const Vector& point) const;

  bool is_coplanar_with(const LineND& other) const;

  std::optional<LineND> common_perpendicular(const LineND& other) const;

  std::string to_string() const;
  std::string to_latex() const;

  bool operator==(const LineND& other) const;
  bool operator!=(const LineND& other) const;

  static LineND from_two_points(const Vector& p1, const Vector& p2);
  static LineND from_point_and_direction(const Vector& point,
                                         const Vector& direction);
};
