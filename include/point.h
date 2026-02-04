#ifndef POINT_H_
#define POINT_H_

#include "utils.h"
#include "rips.h"

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
        PointContainer(const PointContainer& lhs, const PointContainer& rhs);

        Index num_points() const;
        Index num_atoms() const;

        const Atom* mem(Index i) const;
        Index size(Index i) const;

        Index read_fvecs(const char *fname);
        Index read_fvecs(const char *fname, MPI_Comm comm);

        Index read_seqs(const char *fname);
        Index read_seqs(const char *fname, MPI_Comm comm);

        AtomVector& getdata();
        IndexVector& getoffsets();

        void localgather(const PointContainer& points, const IndexVector& local_indices);
        void allgather(const PointContainer& mypoints, MPI_Comm comm);

        void push_back(const Atom *point_mem, Index point_size);

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

        Index index(Index i) const;
        Real dist_to_center(Index i) const;

        Index num_ghosts() const;
        Index ghost_index(Index i) const;
        Index ghost_size(Index i) const;
        const Atom* ghost_mem(Index i) const;

        typename IndexVector::const_iterator ids_begin() const;
        typename IndexVector::const_iterator ids_end() const;

        void add_ghost_point(const Atom *point_mem, Index point_size, Index point_index);

        template <class Distance>
        void find_neighbors(Real cover, Index leaf_size, Distance& distance, Real radius, EdgeVector& myedges) const;

        void set_interior(Index i);

        const PointContainerType ghosts() const;
        const std::vector<bool> interiors() const;

    private:

        PointContainerType ghost_points;
        IndexVector global_indices, global_ghost_indices;
        RealVector dist_to_centers;
        std::vector<bool> interior;
};

template <class Atom_>
class VoronoiComplex
{
    public:

        using Atom = Atom_;
        using AtomVector = std::vector<Atom>;
        using PointContainerType = PointContainer<Atom>;
        using VoronoiCellType = VoronoiCell<Atom>;

        VoronoiComplex(const VoronoiCellType& cell, Index universe_point_count);

        template <class Distance>
        void build_filtration(Distance& distance, Real radius, Index maxdim, Real cover, Index leaf_size);

        void write_filtration_file(const char *fname, Index n, bool use_ids) const;

    private:

        using NeighborList = std::unordered_map<Index, Real>;
        using NeighborListVector = std::vector<NeighborList>;

        PointContainerType points;
        IndexVector indices;
        std::vector<bool> interior;
        Index local, total, universe_point_count;

        std::vector<Simplex> simplices;

        void bron_kerbosch(IndexVector& current, const IndexVector& cands, Index excluded, const NeighborListVector& graph, NeighborListVector& weights, Index maxdim);
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

        template <class Distance>
        void add_ghost_points_systolic(std::vector<VoronoiCellType>& mycells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm) const;

        template <class Distance>
        void add_ghost_points_systolic_rips(std::vector<VoronoiCellType>& mycells, Distance& distance, Real radius, Real cover, Index leaf_size, MPI_Comm comm) const;

    private:

        PointContainerType centers;
        IndexVector center_ids;

        IndexVector cell_indices;
        RealVector dist_to_centers;
};

#include "point.hpp"

#endif
