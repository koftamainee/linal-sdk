#include "solver/solver.h"

#include <unistd.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    if (line.empty()) {
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

    } else {
      throw std::runtime_error("Unknown task type: " + task_type);
    }

    writer.write_sep_line();
  }

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

  bigfloat scalar(bigint(scalar_str.c_str()));  // TODO FIXME

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
  bigfloat scalar = bigint(scalar_s.c_str());  // TODO FIXME
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
  std::string temp = data.substr(data.find('('));
  auto pos1 = temp.find(')');
  std::string mat_s = temp.substr(1, pos1 - 1);

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
  std::string temp = data.substr(data.find('('));
  auto pos1 = temp.find(')');
  std::string mat_s = temp.substr(1, pos1 - 1);

  Matrix mat(mat_s);
  std::vector<std::vector<bigfloat>> eigenvecs = mat.eigenvectors();

  writer.write_line("\\medskip\n");
  writer.write_line("Eigenvectors of matrix " + mat.to_latex() + ":");
  writer.begin_math();

  for (size_t i = 0; i < eigenvecs.size(); ++i) {
    Vector eigenvec(eigenvecs[i]);
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
  std::vector<std::vector<bigfloat>> basis;
  size_t pos = 0;
  std::vector<bigfloat> test_vector;
  std::vector<Vector> all_vectors;

  while ((pos = data.find('(', pos)) != std::string::npos) {
    auto end_pos = data.find(')', pos);
    if (end_pos == std::string::npos) {
      break;
    }

    std::string vec_s = data.substr(pos + 1, end_pos - pos - 1);
    Vector vec(vec_s);
    all_vectors.push_back(vec);
    pos = end_pos + 1;
  }

  if (all_vectors.empty()) {
    return;
  }

  Vector test_vec = all_vectors.back();
  all_vectors.pop_back();

  for (const auto& vec : all_vectors) {
    std::vector<bigfloat> vec_data;
    vec_data.reserve(vec.dimension());
    for (size_t i = 0; i < vec.dimension(); ++i) {
      vec_data.push_back(vec[i]);
    }
    basis.push_back(vec_data);
  }

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
