#pragma once

#include <optional>
#include <string>

#include "bigfloat.h"
#include "latex_writter.h"
#include "vector.h"

class Line2D {
 private:
  // r = point + t * direction
  Vector point_;
  Vector direction_;

  LatexWriter* writter = &LatexWriter::get_instance();

  void normalize_direction();
  void check_valid_line() const;

 public:
  Line2D() = default;
  Line2D(const Vector& a, const Vector& b, bool is_two_points);
  Line2D(const bigfloat& A, const bigfloat& B, const bigfloat& C);
  Line2D(const std::string& str);

  Line2D(const Line2D&) = default;
  Line2D(Line2D&&) noexcept = default;
  Line2D& operator=(const Line2D&) = default;
  Line2D& operator=(Line2D&&) noexcept = default;
  ~Line2D() = default;

  const Vector& point() const noexcept;
  const Vector& direction() const noexcept;

  struct GeneralForm {
    bigfloat A, B, C;
    std::string to_string() const;
    std::string to_latex() const;
  };

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

  struct NormalForm {
    bigfloat p;
    bigfloat alpha;
    std::string to_string() const;
    std::string to_latex() const;
  };

  struct SlopeInterceptForm {
    bigfloat k;
    bigfloat b;
    bool is_vertical;
    bigfloat x_const;
    std::string to_string() const;
    std::string to_latex() const;
  };

  GeneralForm get_general_form() const;
  ParametricForm get_parametric_form() const;
  CanonicalForm get_canonical_form() const;
  NormalForm get_normal_form() const;
  SlopeInterceptForm get_slope_intercept_form() const;

  std::optional<Vector> intersect(const Line2D& other) const;
  bool is_parallel(const Line2D& other) const;
  bool is_perpendicular(const Line2D& other) const;
  bool contains_point(const Vector& point,
                      const bigfloat& EPS = bigfloat::DEFAULT_EPS) const;

  bigfloat distance_to_point(const Vector& point) const;

  Vector symmetric_point(const Vector& point) const;

  bigfloat angle_with(const Line2D& other,
                      const bigfloat& EPS = bigfloat::DEFAULT_EPS) const;

  Vector project_point(const Vector& point) const;

  bigfloat distance_to_parallel_line(const Line2D& other) const;

  std::optional<bigfloat> triangle_area_with_axes() const;

  std::optional<Vector> intersect_with_segment(const Vector& a,
                                               const Vector& b) const;

  bigfloat distance_to_segment(const Vector& a, const Vector& b) const;

  static std::optional<Vector> intersect_segments(const Vector& a1,
                                                  const Vector& a2,
                                                  const Vector& b1,
                                                  const Vector& b2);

  std::string to_string() const;
  std::string to_latex() const;

  bool operator==(const Line2D& other) const;
  bool operator!=(const Line2D& other) const;
};
