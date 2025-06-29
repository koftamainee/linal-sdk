#include <cstring>
#include <exception>

#include "bigfloat.h"

std::ostream &operator<<(std::ostream &out, bigfloat const &num) noexcept {
  if (num == 0) {
    out << 0;
    return out;
  }
  if (num.denominator_ < 0) {
    out << "-";
  }
  if (num.denominator_ != 1) {
    out << num.numerator_ << "/" << num.denominator_.abs();
  } else {
    out << num.numerator_;
  }
  return out;
}
std::istream &operator>>(std::istream &in, bigfloat &num) {
  std::string input;
  in >> input;

  char *slash_pos = std::strchr(input.data(), '/');
  if (slash_pos == nullptr) {
    in.setstate(std::ios::failbit);
    return in;
  }
  *slash_pos = '\0';
  char *numerator_str = input.data();
  char *denominator_str = slash_pos + 1;

  char sign = '+';
  if (numerator_str[0] == '+' || numerator_str[0] == '-') {
    sign = numerator_str[0];
    ++numerator_str;
  }
  try {
    bigint numerator(numerator_str, 10);
    bigint denominator(denominator_str, 10);
    if (sign == '-') {
      denominator.negate();
    }

    num.denominator_ = std::move(denominator);
    num.numerator_ = std::move(numerator);

  } catch (std::exception const &e) {
    in.setstate(std::ios::failbit);
    return in;
  }
  return in;
}

std::string bigfloat::to_decimal(size_t precision) const {
  bool is_negative = (denominator_ < 0);

  bigint abs_num = numerator_.abs();
  bigint abs_den = denominator_.abs();

  bigint::division_result dr = bigint::division(abs_num, abs_den);
  bigint integer_part = dr.quotient();
  bigint remainder = dr.remainder();

  std::string result;

  if (is_negative) {
    result += "-";
  }

  result += integer_part.to_string();

  if (remainder != 0 && precision > 0) {
    result += ".";

    for (size_t i = 0; i < precision; ++i) {
      remainder *= 10;
      bigint digit = remainder / abs_den;
      remainder %= abs_den;

      result += digit.to_string();

      if (remainder == 0) {
        break;
      }
    }
  }

  return result;
}

bigfloat::bigfloat(std::string const &str) {
  std::string s;
  for (const auto c : str) {
    if (c != ' ') {
      s.push_back(c);
    }
  }

  auto slash_pos = s.find('/');
  if (slash_pos != std::string::npos) {
    std::string num_str = s.substr(0, slash_pos);
    std::string den_str = s.substr(slash_pos + 1);

    if (num_str.empty() || den_str.empty()) {
      throw std::invalid_argument("Invalid fraction format");
    }

    numerator_ = bigint(num_str.c_str());
    denominator_ = bigint(den_str.c_str());
    if (denominator_ == 0) {
      throw std::invalid_argument("Denominator cannot be zero");
    }
    simplify();
    return;
  }

  auto dot_pos = s.find('.');
  if (dot_pos != std::string::npos) {
    std::string int_part = s.substr(0, dot_pos);
    std::string frac_part = s.substr(dot_pos + 1);

    std::string num_str = int_part + frac_part;
    bigint num(num_str.c_str());
    bigint den = bigint(1);
    for (size_t i = 0; i < frac_part.size(); ++i) {
      den *= 10;
    }

    numerator_ = num;
    denominator_ = den;
    simplify();
    return;
  }

  numerator_ = bigint(s.c_str());
  denominator_ = bigint(1);
}
