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

        Index read_seqs(const char *fname);
        Index read_seqs(const char *fname, MPI_Comm comm);

        AtomVector& getdata() { return data; }
        IndexVector& getoffsets() { return offsets; }

        void localgather(const PointContainer& points, const IndexVector& local_indices);
        void allgather(const PointContainer& mypoints, MPI_Comm comm);

    protected:

        AtomVector data;
        IndexVector offsets;
};

template <class Atom_>
class VoronoiDiagram
{
    public:

        using Atom = Atom_;
        using PointContainerType = PointContainer<Atom>;

        template <class Distance>
        VoronoiDiagram(const PointContainerType& points, const PointContainerType& centers, Distance& distance);

    private:

        PointContainerType centers;
        IndexVector cell_indices;
        RealVector dist_to_centers;
};

#include "point.hpp"

#endif
