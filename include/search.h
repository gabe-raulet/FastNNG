#ifndef SEARCH_H_
#define SEARCH_H_

#include "utils.h"
#include "point.h"

class CoverTree
{
    public:

        CoverTree();
        CoverTree(Real cover, Index leaf_size);

        template <class Atom, class Distance>
        void build(const PointContainer<Atom>& points, Distance& distance);

        template <class Atom, class Distance, class Functor>
        Index radius_query(const PointContainer<Atom>& points, Distance& distance, const Atom* query, Index dim, Real radius, const Functor& functor) const;

        template <class Atom, class Distance, class Functor>
        Index radius_query_batched(const PointContainer<Atom>& points, Distance& distance, const PointContainer<Atom>& queries, Real radius, const Functor& functor) const;

        template <class Atom, class Distance>
        bool has_radius_neighbor(const PointContainer<Atom>& points, Distance& distance, const Atom* query, Index dim, Real radius) const;

    private:

        Real cover;
        Index leaf_size;

        IndexVector childarr; /* size m-1; children array */
        IndexVector childptrs; /* size m+1; children pointers */
        IndexVector centers; /* size m; vertex centers */
        RealVector radii; /* size m; vertex radii */

        using IndexIter = typename IndexVector::const_iterator;

        IndexIter child_begin(Index vertex) const;
        IndexIter child_end(Index vertex) const;

        void clear_tree();
        void allocate(Index num_verts);
};

#include "search.hpp"

#endif
