#ifndef REDUCTION_H_
#define REDUCTION_H_

#include "utils.h"
#include "rips.h"

class RipsFiltration
{
    public:

        RipsFiltration();
        RipsFiltration(const std::vector<Simplex>& simplices, Index num_verts);

        void read_file(const char *fname);

        Index size() const { return simplices.size(); }
        Index num_vertices() const { return num_verts; }
        Simplex operator[](Index i) const { return simplices[i]; }

    private:

        Index num_verts;
        std::vector<Simplex> simplices;
};

class BoundaryMatrix
{
    public:

        BoundaryMatrix(const RipsFiltration& filt);

        Index low(Index col) const;
        void add_column(Index left, Index right);
        void reduce();

        Index num_rows() const { return pivots.size(); }
        Index num_cols() const { return pivots.size(); }

        void write_homology_persistence(FILE *f) const;
        void write_cohomology_persistence(FILE *f) const;

        void antitranspose();

    private:

        IndexVector pivots;
        std::vector<IndexVector> columns;

        IndexMap id_to_sorted_id;
        IndexVector sorted_id_to_dim;
        RealVector sorted_id_to_value;
};

#include "reduction.hpp"

#endif
