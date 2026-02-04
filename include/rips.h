#ifndef RIPS_H_
#define RIPS_H_

#include "utils.h"
#include "binom.h"

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
    int interior;

    friend bool operator<(const Simplex& lhs, const Simplex& rhs) { return (std::tie(lhs.value, lhs.id) < std::tie(rhs.value, rhs.id)); }
    friend bool operator==(const Simplex& lhs, const Simplex& rhs) { return (lhs.id == rhs.id); }
    friend bool operator!=(const Simplex& lhs, const Simplex& rhs) { return (lhs.id != rhs.id); }

    std::string repr(Index n) const;

    void reindex(const IndexVector& indices, Index n);
};

#include "rips.hpp"

#endif
