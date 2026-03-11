#ifndef TSP_HPP
#define TSP_HPP
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <numeric>
#include <algorithm>
#include <ostream>
#include <iostream>
#include <sstream>

using index_t = std::size_t;
using distance_t = std::int64_t; // nie double! zgodność z TSPLIB aby móc opierać się na wynikach z niej
// moze zmienic na tsp_lib_distance_t?
// trzeba zobaczyć co z problemami z innymi jednostkami (np floating-point)
using point_t = std::pair<double, double>;
using city_index_t = index_t;

static constexpr bool CACHE_THE_COST = true;
static constexpr bool DONT_CACHE_THE_COST = false;

namespace util {
    distance_t euclidean_distance2(const point_t& a, const point_t& b) {
        return static_cast<distance_t>(std::round(std::hypot(b.first - a.first, b.second - a.second))); // kompatybilne z TSPLIB (stąd round)
    }

    distance_t euclidean_distance(const point_t& pa, const point_t& pb) {
        const auto d1 = pb.first - pa.first;
        const auto d2 = pb.second - pa.second;
        const auto a = d1 * d1;
        const auto b = d2 * d2;
        return static_cast<distance_t>(std::sqrt(static_cast<double>(a + b)) + 0.5); // rekomendowana wersja dla kompatybilnosci z tsplib
    }

    template <typename T>
    class square_mat {
        std::unique_ptr<T[]> p;
        std::size_t n;

    public:
        square_mat() {

        }

        square_mat(std::size_t n_rows)
        : p(std::make_unique<T[]>(n_rows * n_rows))
        , n{n_rows} {
        }

        void init(std::size_t n_rows) {
            p = std::make_unique<T[]>(n_rows * n_rows);
            n = n_rows;
        }

        T& at(index_t i, index_t j) noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            return p[idx];
        }

        const T& at(index_t i, index_t j) const noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            return p[idx];
        }

        void set(index_t i, index_t j, const T& t) noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            p[idx] = t;
        }
    };
}

class TSP_Graph { // symetryczny TSP
    std::size_t n;
    util::square_mat<distance_t> distances;

public:
    TSP_Graph(const char* filename) {
        std::ifstream f(filename);

        std::string line;
        while (std::getline(f, line)) { // pominiecie informacji
            if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                break;
            }
        }

        auto pos = f.tellg();
        
        std::size_t count = 0;
        while (std::getline(f, line)) { // zliczenie ilosci miast w pliku
            if (line == "EOF") {
                break;
            }

            if (!line.empty()) {
                ++count;
            }
        }

        n = count;

        f.clear();
        f.seekg(pos); // po liczeniu ilosci miasttrzeba przywtrocic pozycję

        distances.init(n);
        auto coords = std::make_unique<point_t[]>(n);

        int id_dummy;

        for (index_t i = 0; i < n; ++i) {
            auto& coords_i = coords[i];
            f >> id_dummy >> coords_i.first >> coords_i.second;
        }

        f.close();

        for (index_t i = 0; i < n; ++i) {
            for (index_t j = 0; j < n; ++j) {
                // dałoby się oczywiście przechować tylko górną część macierzy, ale to by się wiązało z większą ilością operacji przy obliczaniu indexu
                // ([i*n+j], a [i * n - (i * (i - 1)) / 2 + (j - i))]) oraz zamiana `i` z `j` jeśli j < i, zdecydowałem więc, że wolę poświęcić połowę więcej pamięci dla tablicy
                // szczególnie biorąc pod uwagę, że tablica będzie bardzo często indeksowana przy implementacji PSO i DE
                distances.set(i, j, util::euclidean_distance(coords[i], coords[j]));
            }
        }
    }

    auto distance(index_t i, index_t j) const noexcept {
        return distances.at(i, j);
    }

    auto n_cities() const noexcept {
        return n;
    }

    void print(std::ostream& os, const char* tail = "") const {
        for (index_t i = 0; i < n; ++i) {
            for (index_t j = 0; j < n; ++j) {
                os << distances.at(i, j) << ' ';
            }
            os << '\n';
        }
        os << tail;
    }
};

namespace detail {
    template <bool enable>
    struct cache_t {
        bool up_to_date = false;
        distance_t cost;
    };

    template <>
    struct cache_t<false> {
    };
}

static int global_cost_fn_call_counter = 0;

template <typename T, typename Derived, bool should_cache = CACHE_THE_COST>
class base_TSP_solution_set { // zmienic nazwe na base_chromosome
protected:
    using value_type = T;
    std::unique_ptr<value_type[]> values; // zmienic nazwe na chromosome
    // musi być mutable, bo total_cost powinen być const metodą
    mutable detail::cache_t<should_cache> cache;

    base_TSP_solution_set(std::size_t n_chromosomes) // zmienic nazwe na n_genes
    : values(std::make_unique<value_type[]>(n_chromosomes)) {
    }

    base_TSP_solution_set(base_TSP_solution_set&& other)
    : values(std::move(other.values)) {
    }

    base_TSP_solution_set(const base_TSP_solution_set&) = delete; // uzywac write_copy_of
    base_TSP_solution_set& operator=(const base_TSP_solution_set&) = delete;

    base_TSP_solution_set& operator=(base_TSP_solution_set&& other) {
        if constexpr (should_cache) {
            cache.up_to_date = false;
        }
        values = std::move(other.values);
        return *this;
    }

public:
    distance_t total_cost(const TSP_Graph& graph) const { // klasy pochodne na rozne sposoby moga obliczac koszt. logicznie i praktycznie zachowanie cacheu zawsze jest takie same, dlatego zrobione jest w ponizszy sposob, dzieki temu klasy pochodne nie musza sie martwic o implementacje cacheowania
        static_assert(
            std::is_same_v<
                distance_t,
                decltype(std::declval<const Derived&>()._compute_cost(std::declval<const TSP_Graph&>()))
            >,
            "Derived class must implement: distance_t _compute_cost(const TSP_Graph&) const" // metoda _compute_cost nie powinna być wywoływana nigdzie poza tą metodą
        );

        if constexpr (should_cache) {
            if (!cache.up_to_date) {
                ++global_cost_fn_call_counter;
                cache.cost = static_cast<const Derived&>(*this)._compute_cost(graph);
                cache.up_to_date = true;
            }

            return cache.cost;
        }
        else {
            ++global_cost_fn_call_counter;
            return static_cast<const Derived&>(*this)._compute_cost(graph);
        }
    }

    const T* get_values_raw_ptr() const noexcept {
        return values.get();
    }

    template <typename OtherDerived, bool OtherCache>
    void write_copy_of(const base_TSP_solution_set<T, OtherDerived, OtherCache>& other, std::size_t n_chromosomes) {
        if constexpr (should_cache) {
            cache.up_to_date = false;
        }
        std::copy_n(other.get_values_raw_ptr(), n_chromosomes, values.get());
    }

    const value_type& at(index_t i) const noexcept {
        // niestety nie mozemy sprawdzić czy index jest in-bounds - wywolujacy musi to zapewnić, inaczej UB. tak samo w set()
        return values[i];
    }

    value_type& mutable_at(index_t i) noexcept {
        static_assert(!should_cache);
        return values[i];
    }

    value_type* mutable_get_ptr() noexcept {
        static_assert(!should_cache);
        return values.get();
    }

    void set(index_t i, value_type t) noexcept {
        if constexpr (should_cache) {
            cache.up_to_date = false;
        }
        values[i] = t;
    }

    void print(std::ostream& os, std::size_t n_genes, const char* tail = "", bool print_last = false) const {
        const auto end_idx = n_genes - 1;

        os << '[';

        for (index_t i = 0; i < end_idx; ++i) {
            os << values[i] << ", ";
        }

        os << values[end_idx];

        if (print_last) {
            os << ", " << values[0];
        }

        os << ']' << tail;
    }
};

template <bool should_cache>
class t_TSP_solution_set : public base_TSP_solution_set<city_index_t, t_TSP_solution_set<should_cache>, should_cache> {

    using _my_base = base_TSP_solution_set<city_index_t, t_TSP_solution_set<should_cache>, should_cache>;
    using _my_base::values;

public:
    using _my_base::value_type;

    t_TSP_solution_set(std::size_t n) : _my_base(n) {
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        distance_t out{};
        const auto n = graph.n_cities();

        for (index_t i = 1; i < n; ++i) {
            out += graph.distance(values[i - 1], values[i]);
        }

        out += graph.distance(values[n - 1], values[0]); // powrót do miasta startowego

        return out;
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        const auto values_end = std::next(values.get(), n_chromosomes);

        std::iota(values.get(), values_end, 0);
        std::shuffle(values.get(), values_end, gen);

        if constexpr (should_cache) {
            assert(!_my_base::cache.up_to_date); // ta metoda raczej zawsze będzie używana tuż po inicjalizacji osobnika, więc ta linijka nie powinna być potrzebna, ale jednak dla pewności daję assert'a, bo nie chcę ustawiać tej zmiennej na false za każdym razem
        }
    }
};

using TSP_solution_set = t_TSP_solution_set<DONT_CACHE_THE_COST>;
using TSP_solution_set_caching = t_TSP_solution_set<CACHE_THE_COST>;

template <bool should_cache> // przenies ten template do internal ns?
class t_Continous_TSP_solution_set : public base_TSP_solution_set<double, t_Continous_TSP_solution_set<should_cache>, should_cache> {

    using _my_base = base_TSP_solution_set<double, t_Continous_TSP_solution_set<should_cache>, should_cache>;

public:
    using typename _my_base::value_type;

    t_Continous_TSP_solution_set(std::size_t n_chromosomes) : _my_base(n_chromosomes) {
    }

    template <bool local_should_cache>
    void set_from_discrete(const t_TSP_solution_set<local_should_cache>& discrete, std::size_t n_genes) {
        const double denom = static_cast<double>(n_genes - 1);

        for (index_t rank = 0; rank < n_genes; ++rank) {
            const auto city = discrete.at(rank);
            //this->set(city, static_cast<value_type>(static_cast<double>(rank) / denom));
            _my_base::values[city] = static_cast<value_type>(static_cast<double>(rank) / denom);
        }

        if constexpr (should_cache) {
            _my_base::cache.up_to_date = false;
        }
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        static std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            _my_base::values[i] = dis(gen); // literatura twierdzi, że nie trzeba sprawdzać i zmieniać duplikatów - przy sortowaniu podczas dyskretyzacji duplikaty nie dadzą problemów, szczególnie że prawie nigdy nie wystąpią
        }
        
        if constexpr (should_cache) {
            assert(!_my_base::cache.up_to_date);
        }
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        return discretize<DONT_CACHE_THE_COST>(graph.n_cities()).total_cost(graph); // dodac tu <DONT_CACHE_THE_COST?>
    }

    template <bool local_should_cache = false>
    auto discretize(std::size_t n_genes) const { // zmienic nazwe na make_new_discrete
        t_TSP_solution_set<local_should_cache> out(n_genes);

        using my_dict = std::pair<value_type, index_t>;
        std::unique_ptr<my_dict[]> val_index_map = std::make_unique<my_dict[]>(n_genes); // to nie moze byc static w przyszlosci

        for (index_t i = 0; i < n_genes; ++i) {
            val_index_map[i] = {this->values[i], i};
        }

        const auto my_dict_comparator = [](const my_dict& a, const my_dict& b) {
            return a.first < b.first;
        };

        std::sort(val_index_map.get(), std::next(val_index_map.get(), n_genes), my_dict_comparator); // pomyslec czy jest jakis algorytm sortujący dobrze radzacy sobie z tym typem danych (double od 0 do 1)

        for (index_t i = 0; i < n_genes; ++i) {
            out.set(i, static_cast<city_index_t>(val_index_map[i].second));
        }

        return out; // RVO
    }
};

    using Continous_TSP_solution_set = t_Continous_TSP_solution_set<DONT_CACHE_THE_COST>; // defaultowo cache'ujemy brak cache'owania tylko w typie dla trial vectora (powód wyżej)
    using Continous_TSP_solution_set_caching = t_Continous_TSP_solution_set<CACHE_THE_COST>; // ten typ jest taki sam jak DE_continous_TSP_solution_set, ale bez cache'owania - dla trial vector'a jest to totalnie niepotrzebne, bo za każdym razem gdy obliczany jest koszt, wektor jest inny (cache'owanie nigdy nie wystąpi)

#endif // ifndef TSP_HPP
