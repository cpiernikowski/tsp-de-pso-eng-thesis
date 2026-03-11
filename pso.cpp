#include "tsp.hpp"
#include "common.hpp"
#include <iostream> // debug

static uint64_t G_cost_func_counter = 0;

static uint64_t G_best_vec_changed = 0;
static uint64_t G_best_vec_comparison = 0;

struct velocity_vec {
    std::unique_ptr<double[]> v;

    velocity_vec(std::size_t n)
    : v(std::make_unique<double[]>(n))
    {

    }

    void set(index_t i, double d) noexcept {
        v[i] = d;
    }

    double at(index_t i) const noexcept {
        return v[i];
    }

    double& mutable_at(index_t i) noexcept {
        return v[i];
    }
};

template <bool should_cache>
struct t_PSO_individual : public t_Continous_TSP_solution_set<should_cache> { // mozna by bylo trzymac w oddzielnych tablicach, ale wtedy musialby byc obliczany offset
    using base_chromosome = t_Continous_TSP_solution_set<should_cache>;
    velocity_vec v_vec;
    Continous_TSP_solution_set_caching best_known_pos; // w tym ma byc caching

    t_PSO_individual(std::size_t n_indivi)
    : base_chromosome(n_indivi)
    , v_vec(n_indivi)
    , best_known_pos(n_indivi)
    {

    }

   void generate_random(std::mt19937& gen, std::size_t n_indivi) {
        static std::uniform_real_distribution<double> dis_ch(0.0, 1.0);
        const double vmax = 1.0 / static_cast<double>(n_indivi); 
        std::uniform_real_distribution<double> dis_vel(-vmax, vmax);

        for (index_t i = 0; i < n_indivi; ++i) {
            base_chromosome::values[i] = dis_ch(gen);
            v_vec.v[i] = dis_vel(gen);
        }

        best_known_pos.write_copy_of(*this, n_indivi);
        // sprawdz czy nie da sie dac po prostu *this?
        //if constepxr should cache.....
    }
};

using PSO_individual = t_PSO_individual<DONT_CACHE_THE_COST>;
using PSO_individual_caching = t_PSO_individual<CACHE_THE_COST>;
// cacheowanie nie ma sensu - polzoenie zawsze sie zmienia
class PSO_population final : public Population<PSO_individual, PSO_individual> {
    using _my_base = Population<PSO_individual, PSO_individual>;

    const double phi_p;
    const double phi_g;
    const double w;
    Continous_TSP_solution_set best_known_pos; //  w tym moze byc caching
    const std::size_t max_iters_2opt;
    const double V_MAX;
    const double V_MIN;

    static constexpr double X_MIN = 0.0;
    static constexpr double X_MAX = 1.0;

public:
    PSO_population(std::size_t pop_size, const TSP_Graph& graph_ref, std::size_t max_iters_2opt, double phi_p, double phi_g, double w)
    : _my_base(pop_size, graph_ref)
    , phi_p{phi_p}
    , phi_g{phi_g}
    , w{w}
    , best_known_pos(graph_ref.n_cities())
    , max_iters_2opt{max_iters_2opt}
    , V_MAX{1 / static_cast<double>(n_genes())} //{0.05 * 1.0}//
    , V_MIN{-V_MAX}
    {

    }

    void generate_random(std::mt19937& gen) {
        debug_population_initialized = true;

        pop[0].generate_random(gen, n_genes());

        // inicjalizacja najlepszego obecnie osobnika
        index_cost_pair b = {0, pop[0].best_known_pos.total_cost(graph)};

        for (index_t i = 1; i < n; ++i) {
            pop[i].generate_random(gen, n_genes());

            const auto c = pop[i].best_known_pos.total_cost(graph);
            if (b.cost > c) {
                b = {i, c};
            }
        }

        const auto& current_best = pop[b.index];
        best_known_pos.write_copy_of(current_best, n_genes());
        G_cost_func_counter += n;
    }

    const auto& best() const { // override
        assert(debug_population_initialized && "best(): population wasn't initialized");
        //index_cost_pair out = {0, pop[0].best_known_pos.total_cost(graph)};
        //for (index_t i = 1; i < n; ++i) {
        //    const auto c = pop[i].best_known_pos.total_cost(graph);
        //    if (out.cost > c) {
        //        out = {i, c};
        //    }
        //}

        return best_known_pos;
    }

    void reflect_position_velocity(double& x, double& v) {
        if (x < X_MIN) {
            x = X_MIN + (X_MIN - x);
            v = -v;
        }

        else if (x > X_MAX) {
            x = X_MAX - (x - X_MAX); 
            v = -v;
        }

        if (x < X_MIN) {
            x = X_MIN;
        } else if (x > X_MAX) {
            x = X_MAX;
        }
    }

    void evolve(std::mt19937& gen) {
        assert(debug_population_initialized);

        static std::uniform_real_distribution dis1(0.0, 1.0);
        const auto local_n_genes = n_genes();

        for (index_t i = 0; i < n; ++i) {
            auto& particle_i = pop[i];

            for (index_t j = 0; j < local_n_genes; ++j) {
                const double r_p = dis1(gen);
                const double r_g = dis1(gen);
                auto& velocity_ij = particle_i.v_vec.mutable_at(j);
                auto& particle_ij = particle_i.mutable_at(j);

                velocity_ij = w * velocity_ij + phi_p * r_p
                            * (particle_i.best_known_pos.at(j) - particle_ij)
                            + phi_g * r_g * (best_known_pos.at(j) - particle_ij);

                velocity_ij = std::clamp(velocity_ij, V_MIN, V_MAX);
                particle_ij += velocity_ij;
                
                //if (dis1(gen) < 0.05) {   // 5% szansy na szum
                //    particle_ij += std::normal_distribution<double>(0, 0.02)(gen);
                //}
                reflect_position_velocity(particle_ij, velocity_ij);
            }

            auto discrete_ch = particle_i.discretize<DONT_CACHE_THE_COST>(local_n_genes);
            perform_2opt(discrete_ch, graph, max_iters_2opt);

            const auto particle_i_cost = discrete_ch.total_cost(graph);
            
            if (particle_i_cost < particle_i.best_known_pos.total_cost(graph)) {
                particle_i.set_from_discrete(discrete_ch, local_n_genes);
                particle_i.best_known_pos.write_copy_of(particle_i, local_n_genes);

                ++G_cost_func_counter;

                if ((G_cost_func_counter % 200) == 0) {
                    std::cout << "Najlepsza trasa dotad: " << best_known_pos.total_cost(graph) <<'\n';
                    std::cout << "Obecna ilosc wywolan funkcji kosztu:" << G_cost_func_counter << '\n';
                }

                ++G_best_vec_comparison;

                if (particle_i_cost < best_known_pos.total_cost(graph)) {
                    best_known_pos.write_copy_of(particle_i, local_n_genes);
                    ++G_best_vec_changed;
                }
            }
            G_cost_func_counter += 2;
            const auto r = G_cost_func_counter % 200;
            if (r == 0 || r == 1) {
                std::cout << "Najlepsza trasa dotad: " << best_known_pos.total_cost(graph) <<'\n';
                std::cout << "Obecna ilosc wywolan funkcji kosztu:" << G_cost_func_counter << '\n';
            }
        }
    }
};

int main(int argc, char** argv) {
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
 
    PSO_population pop(pargs.pop_size, graph, pargs.max_iters_2opt, 1.5, 1.5, 0.7);
    pop.generate_random(mt);

    const auto& ref_best_current = pop.best();

    std::cout << "Najlepsza trasa dotad: " << ref_best_current.total_cost(graph) <<'\n';
    std::cout << "Obecna ilosc wywolan funkcji kosztu:" << G_cost_func_counter << '\n';

    for (std::size_t i = 0; i < pargs.n_of_evolutions; ++i) { // liczba ewolucji
        pop.evolve(mt);
    }

    auto& best = pop.best();

    std::cout << "Najlepsza znaleziona droga:\n";
    best.discretize(graph.n_cities()).print(std::cout, pop.n_genes(), "\n", true);

    std::cout << "Koszt tej trasy: " << best.total_cost(graph) << '\n';
    std::cout << "Ilosc wywolan funkcji kosztu: " << G_cost_func_counter << '\n';
    std::cout << "Ilosc wywolan funkcji kosztu: " << global_cost_fn_call_counter << '\n';
    std::cout << "Porównania z najelpszym globalnie położeniem: " << G_best_vec_comparison << '\n';
    std::cout << "Ile razy zmnieniło się najlepsze globalnie położenie: " << G_best_vec_changed;

    return EXIT_SUCCESS;
}