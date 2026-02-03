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
        PointContainer(const std::vector<const Atom*>& atoms, const IndexVector& sizes);

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
class VoronoiCell : public PointContainer<Atom_>
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using PointContainerType = PointContainer<Atom>;

        VoronoiCell(const PointContainerType& points, const IndexVector& global_indices, const RealVector& dist_to_centers);

    private:

        PointContainerType ghost_points;
        IndexVector global_indices, global_ghost_indices;
        RealVector dist_to_centers;
};

template <class Atom_>
class VoronoiDiagram
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using PointContainerType = PointContainer<Atom>;
        using VoronoiCellType = VoronoiCell<Atom>;

        template <class Distance>
        VoronoiDiagram(const PointContainerType& points, const PointContainerType& centers, const IndexVector& center_ids, Distance& distance);

        void coalesce_cells(const PointContainerType& mypoints, std::vector<VoronoiCellType>& mycells, MPI_Comm comm) const;

    private:

        PointContainerType centers;
        IndexVector center_ids;

        IndexVector cell_indices;
        RealVector dist_to_centers;
};

#include "point.hpp"

#endif
