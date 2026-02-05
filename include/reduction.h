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

        Index num_verts;
        std::vector<Simplex> simplices;
};

struct CoboundaryMatrix
{
    CoboundaryMatrix(const RipsFiltration& filt);

    void reduce();

    Index num_rows() const { return columns.size(); }
    Index num_cols() const { return num_rows(); }

    Index low(Index i) const { return columns[i].empty()? -1 : columns[i].back(); }

    void print_persistence() const;

    Index num_vertices;
    std::vector<IndexVector> columns;
    IndexMap id_to_sorted_id;
    RealVector sorted_id_to_value;
    std::vector<Simplex> simplices;
};

#include "reduction.hpp"

#endif
