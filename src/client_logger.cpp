#include "client_logger.h"

#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

std::map<std::string, std::pair<std::ostream *, size_t>>
    client_logger::_all_streams =
        std::map<std::string, std::pair<std::ostream *, size_t>>();

client_logger::client_logger(
    std::map<logger::severity,
             std::pair<std::set<std::string>, std::string>> const &streams,
    std::string log_format)
    : _log_format(std::move(log_format)) {
  std::set<std::string> registered_paths;

  for (auto const &severity_path : streams) {
    _streams[severity_path.first] =
        std::vector<std::pair<std::ostream *, std::string>>(
            severity_path.second.first.size());
    int i = 0;

    for (auto const &path : severity_path.second.first) {
      auto it = _all_streams.find(path);

      if (it == _all_streams.cend()) {
        _all_streams[path] = std::make_pair(
            path.empty() ? &std::cout : new std::ofstream(path), 1);

        it = _all_streams.find(path);

        registered_paths.insert(path);
      } else if (!registered_paths.contains(it->first)) {
        ++(it->second.second);

        registered_paths.insert(path);
      }

      _streams[severity_path.first][i++] =
          std::make_pair(it->second.first, path);
    }
  }
}

void client_logger::cleanup_streams() {
  std::set<std::string> unregistered_paths;

  for (auto const &severity_stream_path : _streams) {
    for (auto const &stream_path : severity_stream_path.second) {
      if (!unregistered_paths.contains(stream_path.second)) {
        unregistered_paths.insert(stream_path.second);

        auto it = _all_streams.find(stream_path.second);
        if (it != _all_streams.end()) {
          if (--(it->second.second) == 0) {
            if (it->second.first != &std::cout &&
                it->second.first != &std::cerr) {
              it->second.first->flush();
              delete it->second.first;
            }
            _all_streams.erase(it);
          }
        }
      }
    }
  }
}

void client_logger::increment_stream_refcounts() {
  for (auto &severity_streams : _streams) {
    for (auto &stream_pair : severity_streams.second) {
      auto &stream_info = _all_streams[stream_pair.second];
      stream_info.second++;
    }
  }
}

client_logger::~client_logger() { cleanup_streams(); }

client_logger::client_logger(client_logger const &other)
    : _streams(other._streams), _log_format(other._log_format) {
  increment_stream_refcounts();
}

client_logger &client_logger::operator=(client_logger const &other) {
  if (this != &other) {
    cleanup_streams();
    _streams = other._streams;
    _log_format = other._log_format;
    increment_stream_refcounts();
  }
  return *this;
}

client_logger::client_logger(client_logger &&other) noexcept
    : _streams(std::move(other._streams)),
      _log_format(std::move(other._log_format)) {}

client_logger &client_logger::operator=(client_logger &&other) noexcept {
  if (this != &other) {
    cleanup_streams();
    _streams = std::move(other._streams);
    _log_format = std::move(other._log_format);
  }
  return *this;
}

logger const *client_logger::log(std::string const &message,
                                 logger::severity severity) const noexcept {
  auto it = _streams.find(severity);

  if (it == _streams.cend()) {
    return this;
  }

  time_t log_time;
  time(&log_time);
  auto formatted_message = format_log(message, severity, log_time);

  for (auto const &stream_path : it->second) {
    *stream_path.first << formatted_message << std::endl;
  }

  return this;
}

std::string client_logger::format_log(std::string const &message,
                                      logger::severity severity,
                                      time_t current_date_time) const {
  std::string formatted_log = _log_format;
  struct tm *timeinfo = gmtime(&current_date_time);
  if (timeinfo == nullptr) {
    throw std::runtime_error("Failed to fetch time");
  }

  size_t pos = 0;

  while ((pos = formatted_log.find("%d", pos)) != std::string::npos) {
    char date_str[11];
    strftime(date_str, sizeof(date_str), "%Y-%m-%d", timeinfo);
    formatted_log.replace(pos, 2, date_str);
    pos += strlen(date_str);
  }

  pos = 0;
  while ((pos = formatted_log.find("%t", pos)) != std::string::npos) {
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", timeinfo);
    formatted_log.replace(pos, 2, time_str);
    pos += strlen(time_str);
  }

  pos = 0;
  while ((pos = formatted_log.find("%s", pos)) != std::string::npos) {
    auto it = _streams.find(severity);
    std::string severity_str = logger::severity_to_string(severity);
    formatted_log.replace(pos, 2, severity_str);
    pos += severity_str.length();
  }

  pos = 0;
  while ((pos = formatted_log.find("%m", pos)) != std::string::npos) {
    formatted_log.replace(pos, 2, message);
    pos += message.size();
  }

  return formatted_log;
}
