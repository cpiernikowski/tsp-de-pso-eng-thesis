#include "tsp.hpp"
#include "common.hpp"
#include <iostream>
#include <iomanip>

static uint64_t G_cost_func_counter = 0;

struct DE_params {
    //static constexpr unsigned NP = 1000;
    // constexpr double CR = 0;
    static constexpr double F = 0.7;
};

class DE_population final : public Population<Continous_TSP_solution_set, Continous_TSP_solution_set> {
    using _my_base = Population<Continous_TSP_solution_set, Continous_TSP_solution_set>;
    using _my_base::individual_type;
    using _my_base::trial_type;

    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;
    std::uniform_int_distribution<index_t> evolve_distrib_indivi_index; // przeniesc do structa "distrib"
    index_cost_pair current_best_info;
    double CR;
    std::size_t max_iters_2opt;

    static constexpr double CR_default = 0.5;

public:
    DE_population(std::size_t pop_size, const TSP_Graph& graph_ref, std::size_t max_iters_2opt)
    : _my_base(pop_size, graph_ref)
    , evolve_distrib_r(0.0, 1.0)
    , evolve_distrib_chromosome_index(0, graph_ref.n_cities() - 1)
    , evolve_distrib_indivi_index(0, pop_size - 1)
    , CR{CR_default}
    , max_iters_2opt{max_iters_2opt}
    {
        current_best_info.make_invalid();
    }

    auto get_cr() const noexcept { return CR; }

    void evolve(std::mt19937& gen) {
        assert(debug_population_initialized && "best(): population wasn't initialized");

        const auto random_idx_exclusive = [this, &gen](auto... other_idxes) -> index_t {
            index_t out;
            do out = evolve_distrib_indivi_index(gen); while (((out == other_idxes) || ...));
            return out;
        };

        using gene_value_type = individual_type::value_type; // zmienic value_type na gene_type
        const auto normalize = [](gene_value_type val) -> gene_value_type {
            val = std::fmod(val, gene_value_type{1.0});
            if (val < gene_value_type{0.0}) {
                val += gene_value_type{1.0};
            }
            return val;
        };

        const auto n_genes_local = n_genes(); // mozna by było to pominąć, ale jakoś mi się nie podoba wywoływanie tej funkcji tak dużo razy, nawet jeśli jest optymalizowana przez kompilator do zwykłego read'a. przy tym rozwiązaniu mamy teoretycznie mniejszą zależność od grafu

        for (index_t i = 0; i < n; ++i) {
            const auto idx_a = random_idx_exclusive(i);
            const auto idx_b = random_idx_exclusive(i, idx_a);
            const auto idx_c = random_idx_exclusive(i, idx_a, idx_b);

            const auto& a = pop[idx_a];
            const auto& b = pop[idx_b];
            const auto& c = pop[idx_c];

            auto& x = pop[i]; // wektor bazowy

            const auto R = evolve_distrib_chromosome_index(gen);

            for (index_t j = 0; j < n_genes_local; ++j) {
                const auto ri = evolve_distrib_r(gen);

                if (ri < this->CR || j == R) {
                    trial.set(
                        j,
                        normalize(a.at(j) + DE_params::F * (b.at(j) - c.at(j)))
                    );
                } else {
                    trial.set(j, x.at(j));
                }
            }

            auto discrete_trial = trial.discretize<DONT_CACHE_THE_COST>(n_genes_local);

            perform_2opt(discrete_trial, graph, max_iters_2opt);

            if (discrete_trial.total_cost(graph) < x.total_cost(graph)) {
                x.set_from_discrete<DONT_CACHE_THE_COST>(discrete_trial, n_genes_local);
            }

            G_cost_func_counter += 2;

        //const auto best_this_iter = this->best();
        //if (best_this_iter.cost >= current_best_info.cost) {
        //    this->CR = 0.0;
        //}
        //else {
        //    this->CR = CR_default;
        //    current_best_info = best_this_iter;
        //}
        }

        // nowe pomiary
        auto best_current = best();
        std::cout << "Najlepsza trasa dotad: " << get(best_current.index).total_cost(graph) <<'\n';
        std::cout << "Obecna ilosc wywolan funkcji kosztu:" << G_cost_func_counter << '\n';
    }

    const auto& get_best_info() const noexcept {
        return current_best_info;
    }
};

int main(int argc, char** argv) { // dodac argumenty programu - sciezka do problemu, liczba ewolucji, ilosc osobnikow w populacji, max iteracji 2opt
    ProgramArgs pargs;
    pargs.parse_args(argc, argv);

    if (pargs.display_help) {
        pargs.print_help();
        return EXIT_SUCCESS;
    }

    if (pargs.problem_filename.empty()) {
        std::cerr << "Brak pliku problemu (-file)\n";
        pargs.print_help();
        return EXIT_FAILURE;
    }

    const TSP_Graph graph(pargs.problem_filename.data());
    std::random_device rd;
    std::mt19937 mt(rd());
 
    DE_population pop(pargs.pop_size, graph, pargs.max_iters_2opt);
    pop.generate_random(mt);

    auto best = pop.best();

    std::cout << "Początkowa najlepsza droga: " << pop.get(best.index).total_cost(graph) << '\n';

    for (std::size_t i = 0; i < pargs.n_of_evolutions; ++i) { // liczba ewolucji
        pop.evolve(mt);
    }

    best = pop.best();

    std::cout << "Najlepsza znaleziona droga:\n";
    pop.get(best.index)
       .discretize(pop.n_genes())
       .print(std::cout, pop.n_genes(), "\n", true);
    std::cout << "Koszt tej trasy: " << best.cost << '\n';
    std::cout << "Ilosc wywolan funkcji kosztu: " << G_cost_func_counter << '\n';
    std::cout << "Ilosc wywolan funkcji kosztu: " << global_cost_fn_call_counter << '\n';;
    std::cout << "CR: " << pop.get_cr();

    return EXIT_SUCCESS;
}