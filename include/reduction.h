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

struct UidValue
{
    Real value;
    Index dim;
    Index id;

    UidValue() {}

    UidValue(const Simplex& s) : value(s.getvalue()), dim(s.getdim()), id(s.getid()) {}

    friend std::ostream& operator<<(std::ostream& os, const UidValue& u)
    {
        os << "UidValue(value=" << u.value << ", dim=" << u.dim << ", uid=" << u.id << ")";
        return os;
    }
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
    std::vector<UidValue> col_values;
    std::vector<Simplex> simplices;
};

#include "reduction.hpp"

#endif
