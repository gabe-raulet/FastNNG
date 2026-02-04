#ifndef UTILS_H_
#define UTILS_H_

#undef NDEBUG
#include <assert.h>

#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <random>
#include <deque>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <mpi.h>

using Index = int64_t;
using Real = float;

#ifdef MPI_INDEX
#undef MPI_INDEX
#endif

#define MPI_INDEX MPI_INT64_T

#ifdef MPI_REAL
#undef MPI_REAL
#endif

#define MPI_REAL MPI_FLOAT


using IndexVector = std::vector<Index>;
using RealVector = std::vector<Real>;
using IndexSet = std::unordered_set<Index>;
using IndexMap = std::unordered_map<Index, Index>;

using IndexVectorVector = std::vector<IndexVector>;
using RealVectorVector = std::vector<RealVector>;

using Edge = std::tuple<Index, Index, Real>;
using EdgeVector = std::vector<Edge>;

template <class Iter>
std::string container_repr(Iter first, Iter last);

#define CONTAINER_REPR(container) (container_repr((container).begin(), (container).end()).c_str())

void selection_sample(Index range, Index size, IndexVector& sample, int seed);
std::string format_large_number(Index number, int prec);

#define LARGE(number) format_large_number((number), (1)).c_str()

int get_comm_rank(MPI_Comm comm);
int get_comm_size(MPI_Comm comm);

template <class T> MPI_Datatype mpi_type();

#include "utils.hpp"

#endif
