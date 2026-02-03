#ifndef RIPS_H_
#define RIPS_H_

#include "utils.h"
#include "binom.h"
#include "graph.h"
#include "search.h"

static Binom binom;

struct Simplex
{
    Simplex();
    Simplex(Index id);
    Simplex(const IndexVector& verts);

    Index getid() const;
    Index getuid() const;
    Index getdim() const;

    IndexVector getverts(Index n) const;
    void get_facets(std::vector<Simplex>& facets, Index n) const;
    void get_facet_ids(IndexVector& ids, Index n) const;

    Index id;
    Real value;

    friend bool operator<(const Simplex& lhs, const Simplex& rhs) { return (std::tie(lhs.value, lhs.id) < std::tie(rhs.value, rhs.id)); }
    friend bool operator==(const Simplex& lhs, const Simplex& rhs) { return (lhs.id == rhs.id); }
    friend bool operator!=(const Simplex& lhs, const Simplex& rhs) { return (lhs.id != rhs.id); }

    std::string repr(Index n) const;
};

class RipsComplex
{
    public:

        RipsComplex(const Graph& skeleton, Index maxdim);

    private:

        using NeighborList = std::unordered_map<Index, Real>;
        using NeighborListVector = std::vector<NeighborList>;

        Index num_vertices;
        std::vector<Simplex> simplices;

        void bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborListVector& graph, NeighborListVector& weights, Index maxdim);
};

#include "rips.hpp"

#endif
