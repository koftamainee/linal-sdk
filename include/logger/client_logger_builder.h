#ifndef MATH_PRACTICE_AND_OPERATING_SYSTEMS_CLIENT_LOGGER_BUILDER_H
#define MATH_PRACTICE_AND_OPERATING_SYSTEMS_CLIENT_LOGGER_BUILDER_H

#include <map>
#include <set>

#include "logger_builder.h"

class client_logger_builder final : public logger_builder {

private:
  std::map<logger::severity, std::pair<std::set<std::string>, std::string>>
      _streams_info;
  std::string _log_format;

public:
  client_logger_builder();

public:
  logger_builder *set_log_format(std::string const &format);

public:
  logger_builder *add_file_stream(std::string const &stream_file_path,
                                  logger::severity severity) override;

  logger_builder *add_console_stream(logger::severity severity) override;

  logger_builder *
  transform_with_configuration(std::string const &configuration_file_path,
                               std::string const &configuration_path) override;

  logger_builder *clear() override;

  logger *build() const override;

private:
  static std::string convert_to_absolute(std::string const &path);
};

#endif // MATH_PRACTICE_AND_OPERATING_SYSTEMS_CLIENT_LOGGER_BUILDER_H
