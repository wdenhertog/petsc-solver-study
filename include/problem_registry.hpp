#pragma once

#include "problem.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

using ProblemRegistry = std::map<std::string, std::function<std::unique_ptr<Problem>()>>;

ProblemRegistry make_registry();

// Throws std::out_of_range with a helpful message if name isn't registered
std::unique_ptr<Problem> make_problem(const std::string& name);
