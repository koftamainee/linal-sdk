#include "line_2d.h"

#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

Line2D::Line2D(const Vector& a, const Vector& b, bool is_two_points)
    : point_(a), direction_(is_two_points ? (b - a) : b) {
  if (a.dimension() != 2 || b.dimension() != 2) {
    throw std::invalid_argument(
        is_two_points ? "Both points must be 2-dimensional"
                      : "Point and direction vector must be 2-dimensional");
  }
  check_valid_line();
  normalize_direction();
}

Line2D::Line2D(const std::string& str) {
  std::string s;
  for (char c : str) {
    if (c != ' ') {
      s += c;
    }
  }

  std::regex param_regex(
      R"(x=([+\-]?\d*\.?\d*)\+([+\-]?\d*\.?\d*)t,y=([+\-]?\d*\.?\d*)\+([+\-]?\d*\.?\d*)t)");
  std::smatch match;
  if (std::regex_match(s, match, param_regex)) {
    bigfloat x0(match[1].str());
    bigfloat a(match[2].str());
    bigfloat y0(match[3].str());
    bigfloat b(match[4].str());

    point_ = Vector{x0, y0};
    direction_ = Vector{a, b};

    writter->add_solution_step("Parsing parametric line equation",
                               R"(\text{Given: } x = )" + x0.to_decimal() +
                                   R"( + )" + a.to_decimal() +
                                   R"( t, \quad y = )" + y0.to_decimal() +
                                   R"( + )" + b.to_decimal() + R"( t)");

    writter->add_solution_step(
        "Extracted point and direction vectors",
        R"(\mathbf{point} = \begin{pmatrix})" + x0.to_decimal() + R"( \\ )" +
            y0.to_decimal() + R"(\end{pmatrix}, \quad
           \mathbf{direction} = \begin{pmatrix})" +
            a.to_decimal() + R"( \\ )" + b.to_decimal() + R"(\end{pmatrix})");

    normalize_direction();
    check_valid_line();
    return;
  }

  std::regex general_regex(
      R"(([+\-]?\d*\.?\d*)x([+\-]\d*\.?\d*)y([+\-]\d*\.?\d*)=0)");
  if (std::regex_match(s, match, general_regex)) {
    bigfloat A;
    if (match[1].str().empty() || match[1].str() == "+") {
      A = bigfloat(1);
    } else if (match[1].str() == "-") {
      A = bigfloat(-1);
    } else {
      A = bigfloat(match[1].str());
    }

    bigfloat B(match[2].str());
    bigfloat C(match[3].str());

    point_ = Vector{bigfloat(0), -C / B};
    direction_ = Vector{-B, A};

    writter->add_solution_step("Parsing general line equation",
                               R"(\text{Given: } )" + A.to_decimal() +
                                   R"(x + )" + B.to_decimal() + R"(y + )" +
                                   C.to_decimal() + R"( = 0)");

    writter->add_solution_step("Normal vector and direction vector",
                               R"(\mathbf{n} = \begin{pmatrix})" +
                                   A.to_decimal() + R"( \\ )" + B.to_decimal() +
                                   R"(\end{pmatrix}, \quad
           \mathbf{direction} = \begin{pmatrix})" +
                                   (-B).to_decimal() + R"( \\ )" +
                                   A.to_decimal() + R"(\end{pmatrix})");

    normalize_direction();
    check_valid_line();
    return;
  }

  throw std::invalid_argument("Unsupported line equation format");
}

const Vector& Line2D::point() const noexcept { return point_; }

const Vector& Line2D::direction() const noexcept { return direction_; }

Line2D::GeneralForm Line2D::get_general_form() const {
  bigfloat A = direction_[1];
  bigfloat B = -direction_[0];
  bigfloat C = -(A * point_[0] + B * point_[1]);

  writter->add_solution_step(
      "General form derivation",
      R"(\text{Direction vector: } \vec{d} = \begin{pmatrix})" +
          direction_[0].to_decimal() + R"( \\ )" + direction_[1].to_decimal() +
          R"(\end{pmatrix})");

  writter->add_solution_step(
      "Coefficients A, B, C for Ax + By + C = 0",
      R"(A = )" + A.to_decimal() + R"(,\quad B = )" + B.to_decimal() +
          R"(,\quad C = - (A \cdot x_0 + B \cdot y_0) = )" + C.to_decimal());

  writter->add_solution_step("General form", A.to_decimal() + "x + " +
                                                 B.to_decimal() + "y + " +
                                                 C.to_decimal() + " = 0");

  return {.A = A, .B = B, .C = C};
}

Line2D::ParametricForm Line2D::get_parametric_form() const {
  writter->add_solution_step(
      "Parametric form",
      R"(\vec{r}(t) = \vec{p} + t \cdot \vec{d} = \begin{pmatrix})" +
          point_[0].to_decimal() + R"( \\ )" + point_[1].to_decimal() +
          R"(\end{pmatrix} + t \cdot \begin{pmatrix})" +
          direction_[0].to_decimal() + R"( \\ )" + direction_[1].to_decimal() +
          R"(\end{pmatrix})");

  return {.point = point_, .direction = direction_};
}

Line2D::CanonicalForm Line2D::get_canonical_form() const {
  writter->add_solution_step(
      "Canonical form", R"(\frac{x - )" + point_[0].to_decimal() + R"(}{)" +
                            direction_[0].to_decimal() + R"(} = \frac{y - )" +
                            point_[1].to_decimal() + R"(}{)" +
                            direction_[1].to_decimal() + R"(})");

  return {.point = point_, .direction = direction_};
}

Line2D::NormalForm Line2D::get_normal_form() const {
  GeneralForm general = get_general_form();
  bigfloat norm = sqrt(general.A * general.A + general.B * general.B,
                       bigfloat::DEFAULT_EPS);

  bigfloat p = general.C.abs() / norm;
  bigfloat alpha = arctg(general.B / general.A, bigfloat::DEFAULT_EPS);

  if (general.C < bigfloat(0)) {
    p = -p;
  }

  writter->add_solution_step(
      "Normal form derivation",
      R"(\|\vec{n}\| = \sqrt{A^2 + B^2} = )" + norm.to_decimal() +
          R"(,\quad p = \frac{|C|}{\|\vec{n}\|} = )" + p.to_decimal());

  writter->add_solution_step(
      "Angle α of normal vector",
      R"(\alpha = \arctan{\left(\frac{B}{A}\right)} = )" + alpha.to_decimal());

  writter->add_solution_step(
      "Normal form", R"(x \cos\alpha + y \sin\alpha = )" + p.to_decimal());

  return {.p = p, .alpha = alpha};
}

Line2D::SlopeInterceptForm Line2D::get_slope_intercept_form() const {
  if (direction_[0] == bigfloat(0)) {
    writter->add_solution_step(
        "Slope-intercept form: vertical line",
        R"(\text{Line is vertical, } x = )" + point_[0].to_decimal());
    return {.k = bigfloat(0),
            .b = bigfloat(0),
            .is_vertical = true,
            .x_const = point_[0]};
  }

  bigfloat k = direction_[1] / direction_[0];
  bigfloat b = point_[1] - k * point_[0];

  writter->add_solution_step("Slope k and intercept b",
                             R"(k = \frac{d_y}{d_x} = )" + k.to_decimal() +
                                 R"(,\quad b = y_0 - k \cdot x_0 = )" +
                                 b.to_decimal());

  writter->add_solution_step(
      "Slope-intercept form",
      R"(y = )" + k.to_decimal() + R"(x + )" + b.to_decimal());

  return {.k = k, .b = b, .is_vertical = false, .x_const = bigfloat(0)};
}

std::optional<Vector> Line2D::intersect(const Line2D& other) const {
  GeneralForm form1 = get_general_form();
  GeneralForm form2 = other.get_general_form();

  writter->add_solution_step(
      "General forms of the lines",
      R"(\text{Line 1: } )" + form1.A.to_decimal() + R"(x + )" +
          form1.B.to_decimal() + R"(y + )" + form1.C.to_decimal() + R"( = 0)" +
          "\n" + R"(\text{Line 2: } )" + form2.A.to_decimal() + R"(x + )" +
          form2.B.to_decimal() + R"(y + )" + form2.C.to_decimal() + R"( = 0)");

  bigfloat det = form1.A * form2.B - form1.B * form2.A;

  writter->add_solution_step(
      "Determinant calculation",
      R"(\text{Det} = A_1 B_2 - B_1 A_2 = )" + form1.A.to_decimal() +
          R"( \cdot )" + form2.B.to_decimal() + R"( - )" +
          form1.B.to_decimal() + R"( \cdot )" + form2.A.to_decimal() +
          R"( = )" + det.to_decimal());

  if (det == bigfloat(0)) {
    writter->add_solution_step(
        "Lines are parallel",
        R"(\text{Since } \det = 0, \text{ the lines are parallel or coincident. No unique intersection.})");
    return std::nullopt;
  }

  bigfloat x = (form1.B * form2.C - form1.C * form2.B) / det;
  bigfloat y = (form1.C * form2.A - form1.A * form2.C) / det;

  writter->add_solution_step(
      "Intersection point calculation",
      R"(x = \frac{B_1 C_2 - C_1 B_2}{\det} = \frac{)" + form1.B.to_decimal() +
          R"( \cdot )" + form2.C.to_decimal() + R"( - )" +
          form1.C.to_decimal() + R"( \cdot )" + form2.B.to_decimal() + R"(}{)" +
          det.to_decimal() + R"(} = )" + x.to_decimal() + R"(\\)" +

          R"(y = \frac{C_1 A_2 - A_1 C_2}{\det} = \frac{)" +
          form1.C.to_decimal() + R"( \cdot )" + form2.A.to_decimal() +
          R"( - )" + form1.A.to_decimal() + R"( \cdot )" +
          form2.C.to_decimal() + R"(}{)" + det.to_decimal() + R"(} = )" +
          y.to_decimal());

  writter->add_solution_step("Intersection result",
                             R"(\text{Intersection point: } \begin{pmatrix})" +
                                 x.to_decimal() + R"( \\ )" + y.to_decimal() +
                                 R"(\end{pmatrix})");

  return Vector({x, y});
}

bool Line2D::is_parallel(const Line2D& other) const {
  return are_collinear(direction_, other.direction_);
}

bool Line2D::is_perpendicular(const Line2D& other) const {
  return are_orthogonal(direction_, other.direction_);
}

bool Line2D::contains_point(const Vector& point, const bigfloat& EPS) const {
  return distance_to_point(point) < EPS;
}

bigfloat Line2D::distance_to_point(const Vector& point) const {
  GeneralForm form = get_general_form();

  writter->add_solution_step("General form of the line",
                             R"(\text{Line: } )" + form.A.to_decimal() +
                                 R"(x + )" + form.B.to_decimal() + R"(y + )" +
                                 form.C.to_decimal() + R"( = 0)");

  bigfloat numerator = (form.A * point[0] + form.B * point[1] + form.C).abs();
  writter->add_solution_step(
      "Numerator calculation (absolute value of line equation at point)",
      R"(\left| A x_0 + B y_0 + C \right| = \left| )" + form.A.to_decimal() +
          R"( \cdot )" + point[0].to_decimal() + R"( + )" +
          form.B.to_decimal() + R"( \cdot )" + point[1].to_decimal() +
          R"( + )" + form.C.to_decimal() + R"( \right| = )" +
          numerator.to_decimal());

  bigfloat denominator =
      sqrt(form.A * form.A + form.B * form.B, bigfloat::DEFAULT_EPS);
  writter->add_solution_step(
      "Denominator calculation (norm of vector (A, B))",
      R"(\sqrt{A^2 + B^2} = \sqrt{)" + form.A.to_decimal() + R"(^2 + )" +
          form.B.to_decimal() + R"(^2} = )" + denominator.to_decimal());

  bigfloat distance = numerator / denominator;
  writter->add_solution_step(
      "Distance calculation",
      R"(\text{Distance} = \frac{\text{Numerator}}{\text{Denominator}} = \frac{)" +
          numerator.to_decimal() + R"(}{)" + denominator.to_decimal() +
          R"(} = )" + distance.to_decimal());

  return distance;
}

Vector Line2D::symmetric_point(const Vector& point) const {
  Vector projected = project_point(point);
  return bigfloat(2) * projected - point;
}

bigfloat Line2D::angle_with(const Line2D& other, const bigfloat& EPS) const {
  return angle_between(direction_, other.direction_, EPS);
}

Vector Line2D::project_point(const Vector& point) const {
  Vector to_point = point - point_;
  bigfloat projection_length = to_point.dot(direction_);
  return point_ + projection_length * direction_;
}

std::string Line2D::to_string() const {
  std::ostringstream oss;
  oss << "Line2D:\n";
  oss << "  Parametric: r = " << point_.to_string() << " + t * "
      << direction_.to_string() << "\n";

  GeneralForm general = get_general_form();
  oss << "  General: " << general.to_string() << "\n";

  SlopeInterceptForm slope = get_slope_intercept_form();
  oss << "  Slope-intercept: " << slope.to_string();

  return oss.str();
}

std::string Line2D::to_latex() const {
  std::ostringstream oss;
  oss << "\\text{Line2D:}\\\\\n";

  ParametricForm param = get_parametric_form();
  oss << "\\text{Parametric: } " << param.to_latex() << "\\\\\n";

  GeneralForm general = get_general_form();
  oss << "\\text{General: } " << general.to_latex() << "\\\\\n";

  SlopeInterceptForm slope = get_slope_intercept_form();
  oss << "\\text{Slope-intercept: } " << slope.to_latex();

  return oss.str();
}

bool Line2D::operator==(const Line2D& other) const {
  return is_parallel(other) && contains_point(other.point_);
}

bool Line2D::operator!=(const Line2D& other) const { return !(*this == other); }

std::string Line2D::GeneralForm::to_string() const {
  std::ostringstream oss;
  oss << A.to_decimal() << "*x + " << B.to_decimal() << "*y + "
      << C.to_decimal() << " = 0";
  return oss.str();
}

std::string Line2D::GeneralForm::to_latex() const {
  std::ostringstream oss;
  oss << A.to_decimal() << "x + " << B.to_decimal() << "y + " << C.to_decimal()
      << " = 0";
  return oss.str();
}

std::string Line2D::ParametricForm::to_string() const {
  return "r = " + point.to_string() + " + t * " + direction.to_string();
}

std::string Line2D::ParametricForm::to_latex() const {
  return "\\vec{r} = " + point.to_latex() + " + t \\cdot " +
         direction.to_latex();
}

std::string Line2D::CanonicalForm::to_string() const {
  std::ostringstream oss;
  oss << "(x - " << point[0].to_decimal() << ") / "
      << direction[0].to_decimal();
  oss << " = (y - " << point[1].to_decimal() << ") / "
      << direction[1].to_decimal();
  return oss.str();
}

std::string Line2D::CanonicalForm::to_latex() const {
  std::ostringstream oss;
  oss << "\\frac{x - " << point[0].to_decimal() << "}{"
      << direction[0].to_decimal() << "}";
  oss << " = \\frac{y - " << point[1].to_decimal() << "}{"
      << direction[1].to_decimal() << "}";
  return oss.str();
}

std::string Line2D::NormalForm::to_string() const {
  std::ostringstream oss;
  oss << "x*cos(" << alpha.to_decimal() << ") + y*sin(" << alpha.to_decimal()
      << ") = " << p.to_decimal();
  return oss.str();
}

std::string Line2D::NormalForm::to_latex() const {
  std::ostringstream oss;
  oss << "x\\cos(" << alpha.to_decimal() << ") + y\\sin(" << alpha.to_decimal()
      << ") = " << p.to_decimal();
  return oss.str();
}

std::string Line2D::SlopeInterceptForm::to_string() const {
  if (is_vertical) {
    return "x = " + x_const.to_decimal();
  }
  return "y = " + k.to_decimal() + "*x + " + b.to_decimal();
}

std::string Line2D::SlopeInterceptForm::to_latex() const {
  if (is_vertical) {
    return "x = " + x_const.to_decimal();
  }
  return "y = " + k.to_decimal() + "x + " + b.to_decimal();
}

void Line2D::normalize_direction() {
  bigfloat len = direction_.norm();
  if (len == 0) {
    throw std::invalid_argument("Direction vector cannot be zero");
  }
  direction_ /= len;
}

void Line2D::check_valid_line() const {
  if (point_.dimension() != 2 || direction_.dimension() != 2) {
    throw std::invalid_argument("Line must be 2-dimensional");
  }
}

bool is_point_on_segment(const Vector& pt, const Vector& a, const Vector& b,
                         const bigfloat& EPS) {
  Vector ab = b - a;
  Vector ap = pt - a;
  if (!are_collinear(ab, ap)) {
    return false;
  }
  bigfloat dot1 = ab.dot(ap);
  bigfloat dot2 = ab.dot(ab);
  return dot1 >= -EPS && dot1 <= dot2 + EPS;
}

bigfloat point_to_segment_distance(const Vector& pt, const Vector& a,
                                   const Vector& b) {
  auto* writter = &LatexWriter::get_instance();
  writter->add_solution_step(
      "Input points",
      R"(\text{Point } P = \begin{pmatrix})" + pt[0].to_decimal() + R"( \\ )" +
          pt[1].to_decimal() + R"(\end{pmatrix}, 
      \text{Segment endpoints } A = \begin{pmatrix})" +
          a[0].to_decimal() + R"( \\ )" + a[1].to_decimal() +
          R"(\end{pmatrix}, B = \begin{pmatrix})" + b[0].to_decimal() +
          R"( \\ )" + b[1].to_decimal() + R"(\end{pmatrix})");

  Vector ab = b - a;
  writter->add_solution_step(
      "Vector AB calculation",
      R"(\vec{AB} = \vec{B} - \vec{A} = \begin{pmatrix})" + ab[0].to_decimal() +
          R"( \\ )" + ab[1].to_decimal() + R"(\end{pmatrix})");

  Vector ap = pt - a;
  writter->add_solution_step(
      "Vector AP calculation",
      R"(\vec{AP} = \vec{P} - \vec{A} = \begin{pmatrix})" + ap[0].to_decimal() +
          R"( \\ )" + ap[1].to_decimal() + R"(\end{pmatrix})");

  bigfloat numerator = ap.dot(ab);
  bigfloat denominator = ab.dot(ab);

  writter->add_solution_step(
      "Dot products", R"(\vec{AP} \cdot \vec{AB} = )" + numerator.to_decimal() +
                          R"(, \quad \vec{AB} \cdot \vec{AB} = )" +
                          denominator.to_decimal());

  bigfloat t = numerator / denominator;
  writter->add_solution_step(
      "Initial parameter t calculation",
      R"(t = \frac{\vec{AP} \cdot \vec{AB}}{\vec{AB} \cdot \vec{AB}} = )" +
          t.to_decimal());

  t = std::max(bigfloat(0), std::min(bigfloat(1), t));
  writter->add_solution_step("Clamping t to [0, 1]",
                             R"(t = \max(0, \min(1, t)) = )" + t.to_decimal());

  Vector projection = a + t * ab;
  writter->add_solution_step(
      "Projection point on segment",
      R"(\vec{P}_{proj} = \vec{A} + t \vec{AB} = \begin{pmatrix})" +
          projection[0].to_decimal() + R"( \\ )" + projection[1].to_decimal() +
          R"(\end{pmatrix})");

  bigfloat distance = (pt - projection).norm();

  return distance;
}

bigfloat Line2D::distance_to_parallel_line(const Line2D& other) const {
  if (!is_parallel(other)) {
    return {0};
  }
  return other.distance_to_point(point_);
}

std::optional<bigfloat> Line2D::triangle_area_with_axes() const {
  auto form = get_general_form();
  if (form.A == 0 || form.B == 0) {
    return std::nullopt;
  }
  Vector x_intercept = Vector({-form.C / form.A, 0});
  Vector y_intercept = Vector({0, -form.C / form.B});
  bigfloat area = (x_intercept[0] * y_intercept[1]) / 2;
  return area;
}

std::optional<Vector> Line2D::intersect_with_segment(const Vector& a,
                                                     const Vector& b) const {
  Line2D segment_line(a, b, true);
  auto ipt = intersect(segment_line);
  if (ipt && is_point_on_segment(*ipt, a, b)) {
    return ipt;
  }
  return std::nullopt;
}

bigfloat Line2D::distance_to_segment(const Vector& a, const Vector& b) const {
  if (intersect_with_segment(a, b)) {
    return {0};
  }
  return std::min(distance_to_point(a), distance_to_point(b));
}

std::optional<Vector> Line2D::intersect_segments(const Vector& a1,
                                                 const Vector& a2,
                                                 const Vector& b1,
                                                 const Vector& b2) {
  auto* writter = &LatexWriter::get_instance();
  writter->add_solution_step(
      "Input segments",
      R"(\text{Segment 1: } \begin{pmatrix})" + a1[0].to_decimal() + R"( \\ )" +
          a1[1].to_decimal() + R"( \end{pmatrix} \text{ to } \begin{pmatrix})" +
          a2[0].to_decimal() + R"( \\ )" + a2[1].to_decimal() +
          R"(\end{pmatrix}, 
        \text{Segment 2: } \begin{pmatrix})" +
          b1[0].to_decimal() + R"( \\ )" + b1[1].to_decimal() +
          R"(\end{pmatrix} \text{ to } \begin{pmatrix})" + b2[0].to_decimal() +
          R"( \\ )" + b2[1].to_decimal() + R"(\end{pmatrix})");

  Line2D l1(a1, a2, true);
  Line2D l2(b1, b2, true);

  writter->add_solution_step("Constructing Line2D objects",
                             "Created lines from segments endpoints");

  auto ipt = l1.intersect(l2);

  if (!ipt) {
    return std::nullopt;
  }

  writter->add_solution_step("Intersection point found",
                             R"(\text{Intersection point: } \begin{pmatrix})" +
                                 ipt.value()[0].to_decimal() + R"( \\ )" +
                                 ipt.value()[1].to_decimal() +
                                 R"(\end{pmatrix})");

  bool on_seg1 = is_point_on_segment(*ipt, a1, a2);
  bool on_seg2 = is_point_on_segment(*ipt, b1, b2);

  writter->add_solution_step(
      "Checking if intersection is on segments",
      std::string(R"(\text{On segment 1: } )") + (on_seg1 ? "true" : "false") +
          R"(, \text{On segment 2: } )" + (on_seg2 ? "true" : "false"));

  if (on_seg1 && on_seg2) {
    return ipt;
  }

  return std::nullopt;
}
