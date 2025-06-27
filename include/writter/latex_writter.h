#ifndef LATEX_WRITER_H
#define LATEX_WRITER_H

#include <fstream>
#include <string>

class LatexWriter {
 private:
  static LatexWriter* instance;
  std::ofstream out_file;

  LatexWriter() = default;

 public:
  LatexWriter(const LatexWriter&) = delete;
  LatexWriter& operator=(const LatexWriter&) = delete;

  ~LatexWriter();

  static LatexWriter& get_instance();

  bool init(const std::string& filename);

  void write(const std::string& content);
  void write_line(const std::string& line);

  void write_sep_line();

  void begin_math();
  void end_math();
  void begin_align();
  void end_align();

  void add_solution_step(const std::string& description,
                         const std::string& math);

  void close();
  bool is_open() const;
};

#endif
