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
    Simplex(Index id, Real value, int interior);

    Index getid() const;
    Index getuid() const;
    Index getdim() const;
    Real getvalue() const;
    int getinterior() const;

    IndexVector getverts(Index n) const;
    void get_facets(std::vector<Simplex>& facets, Index n) const;
    void get_facet_ids(IndexVector& ids, Index n) const;

    Index id;
    Real value;
    int interior;

    friend bool operator<(const Simplex& lhs, const Simplex& rhs);
    friend bool operator==(const Simplex& lhs, const Simplex& rhs);
    friend bool operator!=(const Simplex& lhs, const Simplex& rhs);

    std::string repr(Index n) const;
    std::string fullrepr(Index n) const;

    void reindex(const IndexVector& indices, Index n);
};

#include "rips.hpp"

#endif
