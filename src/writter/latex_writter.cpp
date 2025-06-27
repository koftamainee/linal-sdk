#include "writter/latex_writter.h"

LatexWriter* LatexWriter::instance = nullptr;

LatexWriter::~LatexWriter() { close(); }

LatexWriter& LatexWriter::get_instance() {
  if (instance == nullptr) {
    instance = new LatexWriter();
  }
  return *instance;
}

bool LatexWriter::init(const std::string& filename) {
  if (out_file.is_open()) {
    out_file.close();
  }

  out_file.open(filename);
  if (!out_file.is_open()) {
    return false;
  }

  out_file << R"(
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fullpage}
\pagestyle{empty}
\begin{document}
)";
  return true;
}

void LatexWriter::close() {
  if (out_file.is_open()) {
    out_file << "\n\\end{document}\n";
    out_file.close();
  }
}
void LatexWriter::write(const std::string& content) {
  if (out_file.is_open()) {
    out_file << content;
  }
}

void LatexWriter::write_line(const std::string& line) {
  if (out_file.is_open()) {
    out_file << line << "\n";
  }
}

void LatexWriter::begin_math() { write_line("\\["); }

void LatexWriter::end_math() { write_line("\\]"); }

void LatexWriter::begin_align() { write_line("\\begin{align*}"); }

void LatexWriter::end_align() { write_line("\\end{align*}"); }

void LatexWriter::add_solution_step(const std::string& description,
                                    const std::string& math) {
  if (out_file.is_open()) {
    out_file << "\\subsection*{ \\vspace{1em} " << description << "}\n"
             << "\\[\n"
             << math << "\n\\]\n";
  }
}

bool LatexWriter::is_open() const { return out_file.is_open(); }

void LatexWriter::write_sep_line() {
  if (out_file.is_open()) {
    out_file << "\\vspace{1em}\n";
    out_file << "\\hrule\n";
    out_file << "\\vspace{1em}\n";
  }
}
