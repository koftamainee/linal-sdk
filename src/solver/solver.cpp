#include "solver/solver.h"

#include <unistd.h>

#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "line_2d.h"
#include "line_nd.h"
#include "matrix.h"
#include "vector.h"

namespace {

std::vector<Vector> parse_vectors(const std::string& data) {
  std::vector<Vector> vectors;

  size_t pos = 0;
  while (true) {
    size_t start = data.find('(', pos);
    if (start == std::string::npos) {
      break;
    }

    size_t end = data.find(')', start);
    if (end == std::string::npos) {
      break;
    }

    std::string vec_str = data.substr(start + 1, end - start - 1);

    vectors.emplace_back(vec_str);

    pos = end + 1;
  }

  return vectors;
}
}  // namespace

Solver::Solver() : writer(LatexWriter::get_instance()) {}

bool Solver::set_output(const std::string& filename) {
  return writer.init(filename);
}

bool Solver::process_file(const std::string& input_path) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    return false;
  }
  writer.write_line("\\section*{Solved tasks}");
  size_t solved_tasks = 0;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::string task_type;
    iss >> task_type;
    writer.write_line("\\section*{Task " + std::to_string(++solved_tasks) +
                      "}\n");

    if (task_type == "DOT_PRODUCT") {
      std::cout << "Solving dot product task...\n";
      writer.write_line("\\subsection*{Dot product}");
      solve_dot_product(line);

    } else if (task_type == "3D_CROSS_PRODUCT") {
      std::cout << "Solving 3D cross product task...\n";
      writer.write_line("\\subsection*{3D cross product}");
      solve_3D_cross_product(line);

    } else if (task_type == "7D_CROSS_PRODUCT") {
      std::cout << "Solving 7D cross product task...\n";
      writer.write_line("\\subsection*{7D cross product}");
      solve_7D_cross_product(line);

    } else if (task_type == "3D_TRIPLE_PRODUCT") {
      std::cout << "Solving 3D triple product task...\n";
      writer.write_line("\\subsection*{3D triple product}");
      solve_3D_triple_product(line);

    } else if (task_type == "7D_TRIPLE_PRODUCT") {
      std::cout << "Solving 7D triple product task...\n";
      writer.write_line("\\subsection*{7D triple product}");
      solve_7D_triple_product(line);

    } else if (task_type == "VECTOR_ADD") {
      std::cout << "Solving vector addition task...\n";
      writer.write_line("\\subsection*{Vector addition}");
      solve_vector_add(line);

    } else if (task_type == "VECTOR_SUB") {
      std::cout << "Solving vector subtraction task...\n";
      writer.write_line("\\subsection*{Vector subtraction}");
      solve_vector_sub(line);

    } else if (task_type == "SCALAR_MULT") {
      std::cout << "Solving scalar multiplication task...\n";
      writer.write_line("\\subsection*{Scalar multiplication}");
      solve_scalar_mul(line);

    } else if (task_type == "NORM") {
      std::cout << "Solving norm task...\n";
      writer.write_line("\\subsection*{Norm}");
      solve_norm(line);

    } else if (task_type == "GRAM_SCHMIDT_PROCESS") {
      std::cout << "Solving Gram-Schmidt orthogonalization task...\n";
      writer.write_line("\\subsection*{Gram-Schmidt orthogonalization}");
      solve_gram_schmidt_process(line);

    } else if (task_type == "MATRIX_ADD") {
      std::cout << "Solving matrix addition task...\n";
      writer.write_line("\\subsection*{Matrix addition}");
      solve_matrix_add(line);

    } else if (task_type == "MATRIX_MUL") {
      std::cout << "Solving matrix multiplication task...\n";
      writer.write_line("\\subsection*{Matrix multiplication}");
      solve_matrix_multiply(line);

    } else if (task_type == "MATRIX_SCALAR_MUL") {
      std::cout << "Solving scalar multiplication of matrix task...\n";
      writer.write_line("\\subsection*{Matrix scalar multiplication}");
      solve_matrix_scalar_multiply(line);

    } else if (task_type == "DETERMINANT") {
      std::cout << "Solving determinant task...\n";
      writer.write_line("\\subsection*{Determinant}");
      solve_matrix_determinant(line);

    } else if (task_type == "INVERSE") {
      std::cout << "Solving inverse matrix task...\n";
      writer.write_line("\\subsection*{Inverse matrix}");
      solve_matrix_inverse(line);

    } else if (task_type == "GAUSS_SOLVE") {
      std::cout << "Solving system using Gaussian elimination...\n";
      writer.write_line("\\subsection*{Gaussian elimination}");
      solve_matrix_gauss(line);

    } else if (task_type == "GAUSS_JORDAN_SOLVE") {
      std::cout << "Solving system using Gauss-Jordan elimination...\n";
      writer.write_line("\\subsection*{Gauss-Jordan elimination}");
      solve_matrix_gauss_jordan(line);

    } else if (task_type == "EIGENVALUES") {
      std::cout << "Solving eigenvalue task...\n";
      writer.write_line("\\subsection*{Eigenvalues}");
      solve_matrix_eigenvalues(line);

    } else if (task_type == "EIGENVECTORS") {
      std::cout << "Solving eigenvector task...\n";
      writer.write_line("\\subsection*{Eigenvectors}");
      solve_matrix_eigenvectors(line);

    } else if (task_type == "MATRIX_RANK") {
      std::cout << "Solving matrix rank task...\n";
      writer.write_line("\\subsection*{Matrix rank}");
      solve_matrix_rank(line);

    } else if (task_type == "SPAN_SIZE") {
      std::cout << "Solving span size task...\n";
      writer.write_line("\\subsection*{Span size}");
      solve_span_dimension(line);

    } else if (task_type == "MEMBERSHIP") {
      std::cout << "Checking membership in span...\n";
      writer.write_line("\\subsection*{Membership in span}");
      solve_vector_in_span(line);
    } else if (task_type == "LINE_2D_EQUATIONS") {
      std::cout << "Generating Line2D equations...\n";
      writer.write_line("\\subsection*{All line equations}");
      solve_line2D_equasions(line);

    } else if (task_type == "LINE_2D_INTERSECTION") {
      std::cout << "Solving intersection of two Line2D...\n";
      writer.write_line("\\subsection*{Intersection of two lines}");
      solve_line2D_intersection(line);
    } else if (task_type == "DISTANCE_POINT_TO_LINE") {
      std::cout << "Solving distance from point to line...\n";
      writer.write_line("\\subsection*{Distance from point to line}");
      solve_distance_point_to_line(line);

    } else if (task_type == "DISTANCE_PARALLEL_LINES") {
      std::cout << "Solving distance between two parallel lines...\n";
      writer.write_line("\\subsection*{Distance between two parallel lines}");
      solve_distance_between_parallel_lines(line);

    } else if (task_type == "AREA_TRIANGLE_AXES") {
      std::cout << "Solving area of triangle with coordinate axes...\n";
      writer.write_line("\\subsection*{Triangle area formed with axes}");
      solve_area_triangle_with_axes(line);

    } else if (task_type == "ANGLE_BETWEEN_LINES") {
      std::cout << "Solving angle between two lines...\n";
      writer.write_line("\\subsection*{Angle between two lines}");
      solve_angle_between_lines(line);

    } else if (task_type == "LINE_SEGMENT_INTERSECTION") {
      std::cout << "Solving line-segment intersection...\n";
      writer.write_line("\\subsection*{Line and segment intersection}");
      solve_line_segment_intersection(line);

    } else if (task_type == "DISTANCE_POINT_TO_SEGMENT") {
      std::cout << "Solving distance from point to segment...\n";
      writer.write_line("\\subsection*{Distance from point to segment}");
      solve_distance_point_to_segment(line);

    } else if (task_type == "SEGMENT_SEGMENT_INTERSECTION") {
      std::cout << "Solving segment-segment intersection...\n";
      writer.write_line("\\subsection*{Segment and segment intersection}");
      solve_segment_segment_intersection(line);

    } else if (task_type == "DISTANCE_POINT_TO_LINE_ND") {
      std::cout << "Solving distance from point to line in N-D...\n";
      writer.write_line("\\subsection*{Distance from point to line in N-D}");
      solve_distance_point_to_line_nd(line);

    } else if (task_type == "SYMMETRIC_POINT_ND") {
      std::cout << "Solving symmetric point relative to line in N-D...\n";
      writer.write_line(
          "\\subsection*{Symmetric point relative to line in N-D}");
      solve_symmetric_point_nd(line);

    } else if (task_type == "LINE_EQUATIONS_ND") {
      std::cout << "Solving line equations in N-D...\n";
      writer.write_line("\\subsection*{Line equations in N-D}");
      solve_line_equations_nd(line);

    } else if (task_type == "POINTS_COLLINEARITY_ND") {
      std::cout << "Checking points collinearity in N-D...\n";
      writer.write_line("\\subsection*{Points collinearity in N-D}");
      solve_points_collinearity_nd(line);
    } else {
      throw std::runtime_error("Unknown task type: " + task_type);
    }

    writer.write_sep_line();
  }

  std::cout << "Solved " << solved_tasks << " tasks.\n";

  file.close();
  return true;
}
void Solver::close() { writer.close(); }

void Solver::solve_dot_product(std::string const& data) {
  std::string temp = data.substr(data.find('('));
  auto pos = temp.find(')');
  std::string vec1_s = temp.substr(1, pos - 1);
  std::string vec2_s = temp.substr(pos + 3, temp.length() - pos - 4);
  Vector vec1(vec1_s);
  Vector vec2(vec2_s);
  auto res = vec1.dot(vec2);
  writer.write_line("\\medskip\n");
  std::string step;
  step.append(vec1.to_latex());
  step.append(" \\cdot ");
  step.append(vec2.to_latex());
  step.append(" = ");
  step.append(res.to_decimal());
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_3D_cross_product(std::string const& data) {
  std::string temp = data.substr(data.find('('));
  auto pos = temp.find(')');
  std::string vec1_s = temp.substr(1, pos - 1);
  std::string vec2_s = temp.substr(pos + 3, temp.length() - pos - 4);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);

  auto res = vec1.cross_3d(vec2);
  writer.write_line("\\medskip\n");

  std::string step;
  step.append(vec1.to_latex());
  step.append(" \\times ");
  step.append(vec2.to_latex());
  step.append(" = ");
  step.append(res.to_latex());

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}
void Solver::solve_7D_cross_product(std::string const& data) {
  std::string temp = data.substr(data.find('('));
  auto pos = temp.find(')');
  std::string vec1_s = temp.substr(1, pos - 1);
  std::string vec2_s = temp.substr(pos + 3, temp.length() - pos - 4);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);

  auto res = vec1.cross_7d(vec2);

  writer.write_line("\\medskip\n");
  std::string step;
  step.append(vec1.to_latex());
  step.append(" \\times ");
  step.append(vec2.to_latex());
  step.append(" = ");
  step.append(res.to_latex());

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_3D_triple_product(std::string const& data) {
  std::string temp = data.substr(data.find('('));

  auto pos1 = temp.find(')');
  std::string vec1_s = temp.substr(1, pos1 - 1);

  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec2_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  auto pos3 = temp.find('(', end2);
  auto end3 = temp.find(')', pos3);
  std::string vec3_s = temp.substr(pos3 + 1, end3 - pos3 - 1);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);
  Vector vec3(vec3_s);

  auto res = Vector::triple_product_3d(vec1, vec2, vec3);
  writer.write_line("\\medskip\n");

  std::string step;
  step.append(vec1.to_latex());
  step.append(" \\times ");
  step.append(vec2.to_latex());
  step.append(" \\times ");
  step.append(vec3.to_latex());
  step.append(" = ");
  step.append(res.to_decimal());

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_7D_triple_product(std::string const& data) {
  std::string temp = data.substr(data.find('('));

  auto pos1 = temp.find(')');
  std::string vec1_s = temp.substr(1, pos1 - 1);

  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec2_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  auto pos3 = temp.find('(', end2);
  auto end3 = temp.find(')', pos3);
  std::string vec3_s = temp.substr(pos3 + 1, end3 - pos3 - 1);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);
  Vector vec3(vec3_s);

  auto res = Vector::triple_product_7d(vec1, vec2, vec3);
  writer.write_line("\\medskip\n");

  std::string step;
  step.append(vec1.to_latex());
  step.append(" \\times ");
  step.append(vec2.to_latex());
  step.append(" \\times ");
  step.append(vec3.to_latex());
  step.append(" = ");
  step.append(res.to_decimal());

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_vector_add(std::string const& data) {
  std::string temp = data.substr(data.find('('));

  auto pos1 = temp.find(')');
  std::string vec1_s = temp.substr(1, pos1 - 1);

  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec2_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);
  Vector copy = vec1;

  vec1 += vec2;
  writer.write_line("\\medskip\n");

  std::string step =
      copy.to_latex() + " + " + vec2.to_latex() + " = " + vec1.to_latex();

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_vector_sub(std::string const& data) {
  std::string temp = data.substr(data.find('('));

  auto pos1 = temp.find(')');
  std::string vec1_s = temp.substr(1, pos1 - 1);

  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec2_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  Vector vec1(vec1_s);
  Vector vec2(vec2_s);

  Vector copy = vec1;
  vec1 -= vec2;
  writer.write_line("\\medskip\n");

  std::string step =
      copy.to_latex() + " - " + vec2.to_latex() + " = " + vec1.to_latex();

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_scalar_mul(std::string const& data) {
  std::istringstream iss(data);
  std::string op;
  std::string scalar_str;
  iss >> op >> scalar_str;

  std::ranges::remove_if(scalar_str, isspace);

  bigfloat scalar(scalar_str);

  auto vec_start = data.find('(');
  auto vec_end = data.find(')', vec_start);
  std::string vec_s = data.substr(vec_start + 1, vec_end - vec_start - 1);

  Vector vec(vec_s);

  Vector copy = vec;
  vec *= scalar;
  writer.write_line("\\medskip\n");

  std::string step = copy.to_latex() + " \\cdot " + scalar.to_decimal() +
                     " = " + vec.to_latex();

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_norm(std::string const& data) {
  std::string temp = data.substr(data.find('('));
  auto pos = temp.find(')');
  std::string vec_s = temp.substr(1, pos - 1);

  Vector vec(vec_s);

  bigfloat norm_res = vec.norm();
  writer.write_line("\\medskip\n");

  std::string step =
      vec.to_latex() + R"( = \|\vec{v}\| = )" + norm_res.to_decimal();

  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_gram_schmidt_process(const std::string& data) {
  auto vectors = parse_vectors(data);

  writer.add_solution_step("Input vectors", [&]() {
    std::string s;
    for (const auto& v : vectors) {
      s += v.to_latex() + ", ";
    }
    if (!s.empty()) {
      s.pop_back(), s.pop_back();
    }
    return s;
  }());

  auto ortho_basis = Vector::gram_schmidt_process(vectors);
  writer.write_line("\\medskip\n");

  writer.add_solution_step("Orthogonal basis", [&]() {
    std::string s;
    for (const auto& v : ortho_basis) {
      s += v.to_latex() + ", ";
    }
    if (!s.empty()) {
      s.pop_back(), s.pop_back();
    }
    return s;
  }());

  writer.begin_math();
  for (const auto& v : ortho_basis) {
    writer.write_line(v.to_latex());
  }
  writer.end_math();
}

void Solver::solve_matrix_add(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat1_s = temp.substr(0, pos1 + 1);
  auto pos2 = temp.find('[', pos1);
  auto end2 = temp.find(']', pos2);
  std::string mat2_s = temp.substr(pos2, end2 - pos2 + 1);

  Matrix mat1(mat1_s);
  Matrix mat2(mat2_s);
  Matrix copy = mat1;
  mat1 += mat2;

  writer.write_line("\\medskip\n");
  std::string step =
      copy.to_latex() + " + " + mat2.to_latex() + " = " + mat1.to_latex();
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_matrix_multiply(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat1_s = temp.substr(0, pos1 + 1);
  auto pos2 = temp.find('[', pos1);
  auto end2 = temp.find(']', pos2);
  std::string mat2_s = temp.substr(pos2, end2 - pos2 + 1);

  Matrix mat1(mat1_s);
  Matrix mat2(mat2_s);
  Matrix result = mat1 * mat2;

  writer.write_line("\\medskip\n");
  std::string step = mat1.to_latex() + " \\cdot " + mat2.to_latex() + " = " +
                     result.to_latex();
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_matrix_scalar_multiply(std::string const& data) {
  auto first_space = data.find(' ');

  std::string rest = data.substr(first_space + 1);

  auto second_space = rest.find(' ');
  std::string scalar_s = rest.substr(0, second_space);

  std::string temp = rest.substr(second_space + 1);

  std::string mat_s = temp.substr(0, temp.size() - 1);
  bigfloat scalar(scalar_s);
  Matrix mat(mat_s);
  Matrix result = mat * scalar;

  writer.write_line("\\medskip\n");
  std::string step = scalar.to_decimal() + " \\cdot " + mat.to_latex() + " = " +
                     result.to_latex();
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_matrix_determinant(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);

  Matrix mat(mat_s);
  bigfloat det = mat.determinant();

  writer.write_line("\\medskip\n");
  std::string step = "\\det(" + mat.to_latex() + ") = " + det.to_decimal();
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_matrix_inverse(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);

  Matrix mat(mat_s);
  Matrix inv = mat.inverse();

  writer.write_line("\\medskip\n");
  std::string step = mat.to_latex() + "^{-1} = " + inv.to_latex();
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_matrix_gauss(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);
  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  Matrix mat(mat_s);
  Vector vec(vec_s);

  std::vector<bigfloat> b_vec;
  b_vec.reserve(vec.dimension());
  for (size_t i = 0; i < vec.dimension(); ++i) {
    b_vec.push_back(vec[i]);
  }

  std::vector<bigfloat> solution = mat.solve_gauss(b_vec);
  Vector result(solution);

  writer.write_line("\\medskip\n");
  writer.write_line("Solving the system $" + mat.to_latex() + "\\vec{x} = " +
                    vec.to_latex() + "$ using Gaussian elimination:");
  writer.begin_math();
  writer.write_line("\\vec{x} = " + result.to_latex());
  writer.end_math();
}

void Solver::solve_matrix_gauss_jordan(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);
  auto pos2 = temp.find('(', pos1);
  auto end2 = temp.find(')', pos2);
  std::string vec_s = temp.substr(pos2 + 1, end2 - pos2 - 1);

  Matrix mat(mat_s);
  Vector vec(vec_s);

  std::vector<bigfloat> b_vec;
  b_vec.reserve(vec.dimension());
  for (size_t i = 0; i < vec.dimension(); ++i) {
    b_vec.push_back(vec[i]);
  }

  std::vector<bigfloat> solution = mat.solve_gauss_jordan(b_vec);
  Vector result(solution);

  writer.write_line("\\medskip\n");
  writer.write_line("Solving the system $" + mat.to_latex() + "\\vec{x} = " +
                    vec.to_latex() + "$ using Gauss-Jordan elimination:");
  writer.begin_math();
  writer.write_line("\\vec{x} = " + result.to_latex() + "");
  writer.end_math();
}

void Solver::solve_matrix_eigenvalues(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);

  Matrix mat(mat_s);
  std::vector<bigfloat> eigenvals = mat.eigenvalues();
  Vector result(eigenvals);

  writer.write_line("\\medskip\n");
  writer.write_line("Eigenvalues of matrix " + mat.to_latex() + ":");
  writer.begin_math();
  writer.write_line("\\lambda = " + result.to_latex());
  writer.end_math();
}

void Solver::solve_matrix_eigenvectors(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);

  Matrix mat(mat_s);
  auto eigenvecs = mat.eigenvectors();

  writer.write_line("\\medskip\n");
  writer.write_line("Eigenvectors of matrix " + mat.to_latex() + ":");
  writer.begin_math();

  for (size_t i = 0; i < eigenvecs.size(); ++i) {
    const Vector& eigenvec(eigenvecs[i]);
    writer.write_line("\\vec{v}_{" + std::to_string(i + 1) +
                      "} = " + eigenvec.to_latex());
    if (i < eigenvecs.size() - 1) {
      writer.write_line("\\\\");
    }
  }
  writer.end_math();
}

void Solver::solve_matrix_rank(std::string const& data) {
  std::string temp = data.substr(data.find('['));
  auto pos1 = temp.find(']');
  std::string mat_s = temp.substr(0, pos1 + 1);

  Matrix mat(mat_s);
  size_t matrix_rank = mat.rank();

  writer.write_line("\\medskip\n");
  std::string step =
      "\\text{rank}(" + mat.to_latex() + ") = " + std::to_string(matrix_rank);
  writer.begin_math();
  writer.write_line(step);
  writer.end_math();
}

void Solver::solve_span_dimension(std::string const& data) {
  std::vector<std::vector<bigfloat>> vectors;
  size_t pos = 0;

  while ((pos = data.find('(', pos)) != std::string::npos) {
    auto end_pos = data.find(')', pos);
    if (end_pos == std::string::npos) {
      break;
    }

    std::string vec_s = data.substr(pos + 1, end_pos - pos - 1);
    Vector vec(vec_s);

    std::vector<bigfloat> vec_data;
    vec_data.reserve(vec.dimension());
    for (size_t i = 0; i < vec.dimension(); ++i) {
      vec_data.push_back(vec[i]);
    }
    vectors.push_back(vec_data);
    pos = end_pos + 1;
  }

  size_t span_dim = Matrix::span_dimension(vectors);

  writer.write_line("\\medskip\n");
  writer.add_solution_step("Dimension of the span of the given vectors:",
                           "\\dim(\\text{span}) = " + std::to_string(span_dim));
}

void Solver::solve_vector_in_span(std::string const& data) {
  const std::string prefix = "MEMBERSHIP ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());

  // Разделяем на левую и правую части по '|'
  size_t sep = input.find('|');
  if (sep == std::string::npos) {
    throw std::invalid_argument(
        "Expected '|' separator between test vector and basis");
  }

  std::string test_vec_str = input.substr(0, sep);
  std::string basis_str = input.substr(sep + 1);

  auto trim = [](std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    if (start == std::string::npos || end == std::string::npos) {
      s = "";
    } else {
      s = s.substr(start, end - start + 1);
    }
  };

  trim(test_vec_str);
  trim(basis_str);

  std::regex vec_regex(R"(\(([^()]+)\))");
  std::smatch match;

  if (!std::regex_match(test_vec_str, match, vec_regex) || match.size() != 2) {
    throw std::invalid_argument("Invalid test vector format");
  }
  Vector test_vec(match[1].str());

  std::vector<Vector> basis_vectors;
  size_t pos_basis = 0;
  while ((pos_basis = basis_str.find('(')) != std::string::npos) {
    size_t end_pos = basis_str.find(')', pos_basis);
    if (end_pos == std::string::npos) {
      throw std::invalid_argument("Unmatched '(' in basis vectors");
    }
    std::string vec_content =
        basis_str.substr(pos_basis + 1, end_pos - pos_basis - 1);
    basis_vectors.emplace_back(vec_content);
    basis_str = basis_str.substr(end_pos + 1);
  }

  if (basis_vectors.empty()) {
    throw std::invalid_argument("No basis vectors provided");
  }

  std::vector<std::vector<bigfloat>> basis;
  for (const auto& vec : basis_vectors) {
    std::vector<bigfloat> vec_data;
    vec_data.reserve(vec.dimension());
    for (size_t i = 0; i < vec.dimension(); ++i) {
      vec_data.push_back(vec[i]);
    }
    basis.push_back(vec_data);
  }

  std::vector<bigfloat> test_vector;
  test_vector.reserve(test_vec.dimension());
  for (size_t i = 0; i < test_vec.dimension(); ++i) {
    test_vector.push_back(test_vec[i]);
  }

  bool is_in = Matrix::is_in_span(basis, test_vector);

  writer.write_line("\\medskip\n");
  writer.write_line("Checking whether vector $" + test_vec.to_latex() +
                    "$ belongs to the span:");
  writer.write_line("Result: " + std::string(is_in
                                                 ? "\\text{belongs}"
                                                 : "\\text{does not belong}"));
}

void Solver::solve_line2D_equasions(std::string const& data) {
  const std::string prefix = "LINE_2D_EQUATIONS ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument(
        "Input string missing prefix PLANE_LINE_EQUATIONS");
  }
  std::string equation_str = data.substr(pos + prefix.size());

  Line2D line(equation_str);

  writer.add_solution_step("Input equation", "\\texttt{" + equation_str + "}");

  auto gen = line.get_general_form();
  writer.add_solution_step("General form of the line", gen.to_latex());

  auto param = line.get_parametric_form();
  writer.add_solution_step("Parametric form of the line", param.to_latex());

  auto canon = line.get_canonical_form();
  writer.add_solution_step("Canonical form of the line", canon.to_latex());

  auto norm = line.get_normal_form();
  writer.add_solution_step("Normal form of the line", norm.to_latex());

  auto slope = line.get_slope_intercept_form();
  writer.add_solution_step("Slope-intercept form of the line",
                           slope.to_latex());
}

void Solver::solve_line2D_intersection(std::string const& data) {
  const std::string prefix = "LINE_2D_INTERSECTION ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument(
        "Input string missing prefix LINE_2D_INTERSECTION");
  }

  std::string equations_str = data.substr(pos + prefix.size());
  size_t sep_pos = equations_str.find('|');
  if (sep_pos == std::string::npos) {
    throw std::invalid_argument("Expected two equations separated by '|'");
  }

  std::string eq1_str = equations_str.substr(0, sep_pos);
  std::string eq2_str = equations_str.substr(sep_pos + 1);

  // Trim spaces (basic)
  eq1_str.erase(0, eq1_str.find_first_not_of(" \t"));
  eq1_str.erase(eq1_str.find_last_not_of(" \t") + 1);
  eq2_str.erase(0, eq2_str.find_first_not_of(" \t"));
  eq2_str.erase(eq2_str.find_last_not_of(" \t") + 1);

  writer.add_solution_step(
      "Given equations",
      R"(\texttt{)" + eq1_str + R"(}) \\ \texttt{)" + eq2_str + R"(})");

  Line2D line1(eq1_str);
  Line2D line2(eq2_str);

  writer.add_solution_step("Parsing equations",
                           R"(\text{Converting to internal representation})");

  auto intersection = line1.intersect(line2);

  if (!intersection.has_value()) {
    writer.add_solution_step(
        "No intersection",
        R"(\text{The lines are parallel or coincident. No unique point of intersection.})");
    return;
  }

  const Vector& point = intersection.value();

  writer.add_solution_step(
      "Intersection point",
      R"(\text{The lines intersect at: } \mathbf{p} = \begin{pmatrix})" +
          point[0].to_decimal() + R"( \\ )" + point[1].to_decimal() +
          R"(\end{pmatrix})");
}

void Solver::solve_distance_point_to_line(const std::string& data) {
  const std::string prefix = "DISTANCE_POINT_TO_LINE ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string eq_str = data.substr(pos + prefix.size());
  size_t sep = eq_str.find('|');
  if (sep == std::string::npos)
    throw std::invalid_argument("Expected equation and point separated by '|'");

  std::string line_str = eq_str.substr(0, sep);
  std::string point_str = eq_str.substr(sep + 1);

  writer.add_solution_step("Input",
                           "Equation: " + line_str + ", Point: " + point_str);
  Line2D line(line_str);
  Vector pt(point_str);
  bigfloat dist = line.distance_to_point(pt);

  writer.add_solution_step("Distance", "Distance = " + dist.to_decimal());
}

void Solver::solve_distance_between_parallel_lines(const std::string& data) {
  const std::string prefix = "DISTANCE_PARALLEL_LINES ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());
  size_t sep = input.find('|');
  if (sep == std::string::npos) {
    throw std::invalid_argument("Expected two lines");
  }

  std::string l1 = input.substr(0, sep);
  std::string l2 = input.substr(sep + 1);

  Line2D line1(l1);
  Line2D line2(l2);

  if (!line1.is_parallel(line2)) {
    writer.add_solution_step("Not parallel", "Lines are not parallel");
    return;
  }

  bigfloat dist = line1.distance_to_parallel_line(line2);
  writer.add_solution_step("Distance", "Distance = " + dist.to_decimal());
}

void Solver::solve_area_triangle_with_axes(const std::string& data) {
  const std::string prefix = "AREA_TRIANGLE_AXES ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string line_str = data.substr(pos + prefix.size());
  Line2D line(line_str);
  std::optional<bigfloat> area = line.triangle_area_with_axes();

  if (!area.has_value()) {
    writer.add_solution_step("No triangle",
                             "The line does not intersect both axes");
    return;
  }

  writer.add_solution_step("Triangle area",
                           "Area = " + area.value().to_decimal());
}

void Solver::solve_angle_between_lines(const std::string& data) {
  const std::string prefix = "ANGLE_BETWEEN_LINES ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());
  size_t sep = input.find('|');
  if (sep == std::string::npos) {
    throw std::invalid_argument("Expected two lines");
  }

  std::string l1 = input.substr(0, sep);
  std::string l2 = input.substr(sep + 1);

  Line2D line1(l1);
  Line2D line2(l2);

  bigfloat angle = line1.angle_with(line2);
  writer.add_solution_step("Angle", "Angle = " + angle.to_decimal());
}

void Solver::solve_line_segment_intersection(const std::string& data) {
  const std::string prefix = "LINE_SEGMENT_INTERSECTION ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());
  size_t sep = input.find('|');
  if (sep == std::string::npos) {
    throw std::invalid_argument("Expected line and segment");
  }

  std::string line_str = input.substr(0, sep);
  std::string seg_str = input.substr(sep + 1);

  auto trim = [](std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    if (start == std::string::npos || end == std::string::npos) {
      s = "";
    } else {
      s = s.substr(start, end - start + 1);
    }
  };

  trim(line_str);
  trim(seg_str);

  std::regex segment_regex(R"(\(([^()]+)\)\s*\(([^()]+)\))");
  std::smatch match;
  if (!std::regex_search(seg_str, match, segment_regex) || match.size() != 3) {
    throw std::invalid_argument("Invalid segment format");
  }

  Vector A(match[1].str());
  Vector B(match[2].str());

  Line2D line(line_str);

  auto pt = line.intersect_with_segment(A, B);
  if (!pt.has_value()) {
    writer.add_solution_step("No intersection",
                             "No intersection within the segment");
    return;
  }
  writer.add_solution_step("Intersection", "Point: " + pt.value().to_string());
}

void Solver::solve_distance_point_to_segment(const std::string& data) {
  const std::string prefix = "DISTANCE_POINT_TO_SEGMENT ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());

  size_t sep = input.find('|');
  if (sep == std::string::npos) {
    throw std::invalid_argument(
        "Expected '|' separator between point and segment");
  }

  std::string point_str = input.substr(0, sep);
  std::string segment_str = input.substr(sep + 1);

  auto trim = [](std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    if (start == std::string::npos || end == std::string::npos) {
      s = "";
    } else {
      s = s.substr(start, end - start + 1);
    }
  };

  trim(point_str);
  trim(segment_str);

  std::regex point_regex(R"(\(([^()]+)\))");
  std::smatch match;

  if (!std::regex_match(point_str, match, point_regex) || match.size() != 2) {
    throw std::invalid_argument("Invalid point format");
  }
  Vector P(match[1].str());

  std::regex segment_regex(R"(\(([^()]+)\)\s*\(([^()]+)\))");
  if (!std::regex_match(segment_str, match, segment_regex) ||
      match.size() != 3) {
    throw std::invalid_argument("Invalid segment format");
  }

  Vector A(match[1].str());
  Vector B(match[2].str());

  bigfloat d = point_to_segment_distance(P, A, B);
  writer.add_solution_step("Distance", "Distance = " + d.to_decimal());
}

void Solver::solve_segment_segment_intersection(const std::string& data) {
  const std::string prefix = "SEGMENT_SEGMENT_INTERSECTION ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());

  std::regex segment_regex(
      R"(\(([^()]+)\)\s*\(([^()]+)\)\s*\(([^()]+)\)\s*\(([^()]+)\))");
  std::smatch match;
  if (!std::regex_search(input, match, segment_regex) || match.size() != 5) {
    throw std::invalid_argument(
        "Invalid format: expected four points in (x y) format");
  }

  Vector A(match[1].str());  // (0 0)
  Vector B(match[2].str());  // (2 2)
  Vector C(match[3].str());  // (0 2)
  Vector D(match[4].str());  // (2 0)

  auto pt = Line2D::intersect_segments(A, B, C, D);
  if (!pt.has_value()) {
    writer.add_solution_step("No intersection", "Segments do not intersect");
    return;
  }

  writer.add_solution_step("Intersection", "Point: " + pt.value().to_string());
}

void Solver::solve_distance_point_to_line_nd(const std::string& data) {
  const std::string prefix = "DISTANCE_POINT_TO_LINE_ND ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());
  // Разделяем по '|'
  size_t sep_pos = input.find('|');
  if (sep_pos == std::string::npos) {
    throw std::invalid_argument(
        "Expected '|' separator between line and point");
  }

  std::string line_str = input.substr(0, sep_pos);
  std::string point_str = input.substr(sep_pos + 1);

  LineND line(line_str);
  Vector point(point_str);

  bigfloat dist = line.distance_to_point(point);

  writer.add_solution_step("Distance from point to line",
                           "Distance = " + dist.to_decimal());
}

void Solver::solve_symmetric_point_nd(const std::string& data) {
  const std::string prefix = "SYMMETRIC_POINT_ND ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());

  size_t sep_pos = input.find('|');
  if (sep_pos == std::string::npos) {
    throw std::invalid_argument(
        "Expected '|' separator between line and point");
  }

  std::string line_str = input.substr(0, sep_pos);
  std::string point_str = input.substr(sep_pos + 1);

  LineND line(line_str);
  Vector point(point_str);

  Vector sym_point = line.symmetric_point(point);

  writer.add_solution_step("Symmetric point",
                           "Symmetric point: " + sym_point.to_string());
}

void Solver::solve_line_equations_nd(const std::string& data) {
  const std::string prefix = "LINE_EQUATIONS_ND ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string line_str = data.substr(pos + prefix.size());

  LineND line(line_str);

  auto param = line.get_parametric_form();
  auto canon = line.get_canonical_form();
  auto sys = line.get_system_form();

  writer.add_solution_step("Parametric form", param.to_string());
  writer.add_solution_step("Canonical form", canon.to_string());
  writer.add_solution_step("System form", sys.to_string());
}

void Solver::solve_points_collinearity_nd(const std::string& data) {
  const std::string prefix = "POINTS_COLLINEARITY_ND ";
  size_t pos = data.find(prefix);
  if (pos == std::string::npos) {
    throw std::invalid_argument("Missing prefix");
  }

  std::string input = data.substr(pos + prefix.size());

  size_t sep_pos = input.find('|');
  if (sep_pos != std::string::npos) {
    input = input.substr(0, sep_pos);
  }

  std::regex vec_regex(R"(\(([^()]+)\))");
  std::sregex_iterator iter(input.begin(), input.end(), vec_regex);
  std::sregex_iterator end;

  std::vector<Vector> points;
  for (; iter != end; ++iter) {
    points.emplace_back(iter->str().substr(1, iter->str().size() - 2));
  }

  if (points.size() < 2) {
    throw std::invalid_argument("At least two points are required");
  }

  LineND line = LineND::from_two_points(points[0], points[1]);

  bool all_on_line = true;
  for (size_t i = 2; i < points.size(); ++i) {
    if (!line.contains_point(points[i])) {
      all_on_line = false;
      break;
    }
  }

  writer.add_solution_step("Collinearity check",
                           all_on_line ? "All points lie on the same line"
                                       : "Points do NOT lie on the same line");
}
