#ifndef POINT_H_
#define POINT_H_

#include "utils.h"

template <class Atom_>
class PointContainer
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;

        PointContainer();
        PointContainer(const AtomVector& atoms, const IndexVector& sizes);
        PointContainer(const AtomVector& atoms, Index size, Index dim);

        Index num_points() const;
        Index num_atoms() const;

        const Atom* mem(Index i) const;
        Index size(Index i) const;

        Index read_fvecs(const char *fname);
        Index read_fvecs(const char *fname, MPI_Comm comm);

        AtomVector& getdata() { return data; }
        IndexVector& getoffsets() { return offsets; }

    protected:

        AtomVector data;
        IndexVector offsets;
};

#include "point.hpp"

#endif
