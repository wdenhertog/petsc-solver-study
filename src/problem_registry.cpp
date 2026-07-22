#include "problem_registry.hpp"

#include "problems/bratu.hpp"
#include "problems/poisson.hpp"

#include <sstream>
#include <stdexcept>

ProblemRegistry make_registry()
{
    ProblemRegistry registry;

    registry["poisson"] = [] { return std::make_unique<PoissonProblem>(); };
    registry["bratu"] = [] { return std::make_unique<BratuProblem>(); };

    return registry;
}

std::unique_ptr<Problem> make_problem(const std::string& name)
{
    static const ProblemRegistry registry = make_registry();

    auto it = registry.find(name);
    if (it == registry.end())
    {
        std::ostringstream oss;
        oss << "Unknown problem '" << name << "'. Available: ";
        for (const auto& [key, _] : registry)
            oss << key << " ";
        throw std::out_of_range(oss.str());
    }
    return it->second();
}
