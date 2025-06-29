#include "line_nd.h"

#include <regex>
#include <sstream>
#include <stdexcept>

LineND::LineND(const Matrix& A, const Vector& b) {
  if (A.rows() != b.dimension()) {
    throw std::invalid_argument("Matrix rows must match vector dimension");
  }

  dimension_ = A.cols();

  std::vector<bigfloat> particular_solution =
      A.solve_gauss(std::vector<bigfloat>(b.dimension()));

  point_ = Vector(particular_solution);

  direction_ = A.nullspace_vector(bigfloat::DEFAULT_EPS);

  check_valid_line();
  normalize_direction();
}

LineND::LineND(const std::string& str) {
  auto* writer = &LatexWriter::get_instance();
  std::string s;
  for (char c : str) {
    if (c != ' ') {
      s += c;
    }
  }

  std::regex coord_regex(
      R"(([a-zA-Z]\d*)=([+\-]?\d*\.?\d*)\+([+\-]?\d*\.?\d*)t)");

  std::sregex_iterator it(s.begin(), s.end(), coord_regex);
  std::sregex_iterator end;

  std::vector<bigfloat> points;
  std::vector<bigfloat> directions;

  while (it != end) {
    const std::smatch& match = *it;
    bigfloat p(match[2].str());
    bigfloat a(match[3].str());

    points.push_back(p);
    directions.push_back(a);

    ++it;
  }

  if (points.empty()) {
    throw std::invalid_argument("Invalid or unsupported line format");
  }

  point_ = Vector(points);
  direction_ = Vector(directions);

  writer->add_solution_step(
      "Parsing parametric line equation",
      "Parsed point and direction vectors from input string");

  normalize_direction();
  check_valid_line();
}

void LineND::normalize_direction() {
  if (!direction_.is_zero()) {
    bigfloat norm = direction_.norm();
    if (norm != bigfloat(0)) {
      direction_ /= norm;
    }
  }
}

void LineND::check_valid_line() const {
  if (direction_.is_zero()) {
    throw std::invalid_argument("Direction vector cannot be zero");
  }
}

void LineND::check_dimension_match(const Vector& vec,
                                   const std::string& operation) const {
  if (vec.dimension() != dimension_) {
    throw std::invalid_argument("Vector dimension mismatch in " + operation);
  }
}

const Vector& LineND::point() const noexcept { return point_; }

const Vector& LineND::direction() const noexcept { return direction_; }

size_t LineND::dimension() const noexcept { return dimension_; }

LineND::ParametricForm LineND::get_parametric_form() const {
  return {.point = point_, .direction = direction_};
}

LineND::CanonicalForm LineND::get_canonical_form() const {
  return {.point = point_, .direction = direction_};
}

LineND::SystemForm LineND::get_system_form() const {
  size_t n = dimension_;
  Matrix A(n - 1, n);
  Vector b(n - 1);

  std::vector<Vector> basis_vectors;

  for (size_t i = 0; i < n; ++i) {
    Vector e_i = Vector::basis_vector(n, i);
    if (!are_collinear(e_i, direction_)) {
      basis_vectors.push_back(e_i);
    }
  }

  for (size_t i = 0; i < basis_vectors.size() && A.rows() > 0; ++i) {
    Vector& v = basis_vectors[i];

    bigfloat proj_coeff = v.dot(direction_);
    v -= proj_coeff * direction_;

    if (!v.is_zero()) {
      v = v.normalize();

      for (size_t j = 0; j < n; ++j) {
        A.at(i, j) = v[j];
      }

      b[i] = v.dot(point_);
    }
  }

  return {.A = A, .b = b};
}

bool LineND::is_parallel(const LineND& other) const {
  return are_collinear(direction_, other.direction_);
}

bool LineND::is_perpendicular(const LineND& other) const {
  return are_orthogonal(direction_, other.direction_);
}

bool LineND::contains_point(const Vector& point, const bigfloat& EPS) const {
  check_dimension_match(point, "contains_point");
  return distance_to_point(point) < EPS;
}

std::optional<Vector> LineND::intersect(const LineND& other) const {
  if (!is_coplanar_with(other)) {
    return std::nullopt;
  }

  if (is_parallel(other)) {
    return std::nullopt;
  }

  Vector delta = other.point_ - point_;

  size_t n = dimension_;
  Matrix A(n, 2);

  for (size_t i = 0; i < n; ++i) {
    A.at(i, 0) = direction_[i];
    A.at(i, 1) = -other.direction_[i];
  }

  std::vector<bigfloat> delta_vec(delta.dimension());
  for (size_t i = 0; i < delta.dimension(); ++i) {
    delta_vec[i] = delta[i];
  }

  try {
    std::vector<bigfloat> params = A.solve_gauss(delta_vec);
    const bigfloat& t1 = params[0];

    Vector intersection = point_ + t1 * direction_;
    return intersection;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bigfloat LineND::distance_to_point(const Vector& point) const {
  check_dimension_match(point, "distance_to_point");

  Vector to_point = point - point_;
  Vector projection = to_point.dot(direction_) * direction_;
  Vector perpendicular = to_point - projection;

  return perpendicular.norm();
}

bigfloat LineND::distance_to_line(const LineND& other) const {
  if (is_coplanar_with(other)) {
    if (is_parallel(other)) {
      return other.distance_to_point(point_);
    }
    return {0};
  }

  Vector delta = other.point_ - point_;

  if (dimension_ == 3) {
    Vector cross = direction_.cross_3d(other.direction_);
    return delta.dot(cross).abs() / cross.norm();
  }
  Matrix M(dimension_, dimension_);

  for (size_t i = 0; i < dimension_; ++i) {
    M.at(i, 0) = direction_[i];
    if (dimension_ > 1) {
      M.at(i, 1) = other.direction_[i];
    }
    if (dimension_ > 2) {
      M.at(i, 2) = delta[i];
    }
  }

  for (size_t j = 3; j < dimension_; ++j) {
    for (size_t i = 0; i < dimension_; ++i) {
      M.at(i, j) = (i == j - 3) ? bigfloat(1) : bigfloat(0);
    }
  }

  bigfloat det = M.determinant().abs();
  Vector cross_analog = direction_.cross_3d(other.direction_);
  return det / cross_analog.norm();
}

Vector LineND::symmetric_point(const Vector& point) const {
  check_dimension_match(point, "symmetric_point");
  Vector projected = project_point(point);
  return bigfloat(2) * projected - point;
}

bigfloat LineND::angle_with(const LineND& other, const bigfloat& EPS) const {
  return angle_between(direction_, other.direction_, EPS);
}

Vector LineND::project_point(const Vector& point) const {
  check_dimension_match(point, "project_point");
  Vector to_point = point - point_;
  bigfloat projection_length = to_point.dot(direction_);
  return point_ + projection_length * direction_;
}

bool LineND::is_coplanar_with(const LineND& other) const {
  if (dimension_ <= 2) {
    return true;
  }

  if (dimension_ == 3) {
    Vector delta = other.point_ - point_;
    Vector cross = direction_.cross_3d(other.direction_);

    bigfloat mixed_product =
        Vector::triple_product_3d(delta, direction_, other.direction_);
    return mixed_product.abs() < bigfloat::DEFAULT_EPS;
  }

  Vector delta = other.point_ - point_;

  Matrix M(dimension_, 3);
  for (size_t i = 0; i < dimension_; ++i) {
    M.at(i, 0) = delta[i];
    M.at(i, 1) = direction_[i];
    M.at(i, 2) = other.direction_[i];
  }

  return M.rank() <= 2;
}

std::optional<LineND> LineND::common_perpendicular(const LineND& other) const {
  if (is_coplanar_with(other)) {
    return std::nullopt;
  }

  if (dimension_ != 3) {
    return std::nullopt;
  }

  Vector perp_direction = direction_.cross_3d(other.direction_);

  if (perp_direction.is_zero()) {
    return std::nullopt;
  }

  Vector delta = other.point_ - point_;

  Matrix A(2, 2);
  Vector b(2);

  A.at(0, 0) = direction_.dot(direction_);
  A.at(0, 1) = -direction_.dot(other.direction_);
  A.at(1, 0) = other.direction_.dot(direction_);
  A.at(1, 1) = -other.direction_.dot(other.direction_);

  b[0] = direction_.dot(delta);
  b[1] = other.direction_.dot(delta);

  try {
    std::vector<bigfloat> params =
        A.solve_gauss(std::vector<bigfloat>{b[0], b[1]});
    const bigfloat& t1 = params[0];

    Vector point_on_perp = point_ + t1 * direction_;
    return from_point_and_direction(point_on_perp, perp_direction);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::string LineND::to_string() const {
  std::ostringstream oss;
  oss << "LineND (dimension " << dimension_ << "):\n";

  ParametricForm param = get_parametric_form();
  oss << "  Parametric: " << param.to_string() << "\n";

  CanonicalForm canonical = get_canonical_form();
  oss << "  Canonical: " << canonical.to_string() << "\n";

  SystemForm system = get_system_form();
  oss << "  System: " << system.to_string();

  return oss.str();
}

std::string LineND::to_latex() const {
  std::ostringstream oss;
  oss << "\\text{LineND (dimension " << dimension_ << "):}\\\\\n";

  ParametricForm param = get_parametric_form();
  oss << "\\text{Parametric: } " << param.to_latex() << "\\\\\n";

  CanonicalForm canonical = get_canonical_form();
  oss << "\\text{Canonical: } " << canonical.to_latex() << "\\\\\n";

  SystemForm system = get_system_form();
  oss << "\\text{System: } " << system.to_latex();

  return oss.str();
}

bool LineND::operator==(const LineND& other) const {
  return dimension_ == other.dimension_ && is_parallel(other) &&
         contains_point(other.point_);
}

bool LineND::operator!=(const LineND& other) const { return !(*this == other); }

LineND LineND::from_point_and_direction(const Vector& point,
                                        const Vector& direction) {
  if (point.dimension() != direction.dimension()) {
    throw std::invalid_argument(
        "Point and direction must have the same dimension");
  }

  LineND line(point, direction, false);
  line.check_valid_line();
  line.normalize_direction();

  return line;
}

LineND LineND::from_two_points(const Vector& p1, const Vector& p2) {
  if (p1.dimension() != p2.dimension()) {
    throw std::invalid_argument("Points must have the same dimension");
  }

  LineND line(p1, p2 - p1, true);
  line.check_valid_line();
  line.normalize_direction();

  return line;
}

std::string LineND::ParametricForm::to_string() const {
  return "r = " + point.to_string() + " + t * " + direction.to_string();
}

std::string LineND::ParametricForm::to_latex() const {
  return "\\vec{r} = " + point.to_latex() + " + t \\cdot " +
         direction.to_latex();
}

std::string LineND::CanonicalForm::to_string() const {
  std::ostringstream oss;
  oss << "(r - " << point.to_string() << ") || " << direction.to_string();
  return oss.str();
}

std::string LineND::CanonicalForm::to_latex() const {
  std::ostringstream oss;
  oss << "(\\vec{r} - " << point.to_latex() << ") \\parallel "
      << direction.to_latex();
  return oss.str();
}

std::string LineND::SystemForm::to_string() const {
  std::ostringstream oss;
  oss << "System: " << A.to_string() << " * r = " << b.to_string();
  return oss.str();
}

std::string LineND::SystemForm::to_latex() const {
  std::ostringstream oss;
  oss << A.to_latex() << " \\vec{r} = " << b.to_latex();
  return oss.str();
}

LineND::LineND(const Vector& a, const Vector& b, bool is_two_points)
    : point_(a),
      direction_(is_two_points ? (b - a) : b),
      dimension_(a.dimension()) {
  if (a.dimension() != b.dimension()) {
    throw std::invalid_argument(
        is_two_points
            ? "Both points must have the same dimension"
            : "Point and direction vector must have the same dimension");
  }
  check_valid_line();
  normalize_direction();
}
