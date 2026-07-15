#include "problem_registry.hpp"

// #include "problems/elasticity.hpp"
#include "problems/poisson.hpp"
// #include "problems/convection_diffusion.hpp"   // add as they land

#include <sstream>
#include <stdexcept>

ProblemRegistry make_registry()
{
    ProblemRegistry registry;

    registry["poisson"] = [] { return std::make_unique<PoissonProblem>(); };
    // registry["elasticity"] = [] { return std::make_unique<ElasticityProblem>(); };
    // registry["convection_diffusion"] = [] { return
    // std::make_unique<ConvectionDiffusionProblem>(); };

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
