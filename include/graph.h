#ifndef GRAPH_H_
#define GRAPH_H_

#include "utils.h"

class Graph
{
    public:

        using Edge = std::tuple<Index, Index, Real>;
        using EdgeVector = std::vector<Edge>;

        Graph(const EdgeVector& myedges, Index num_verts);

        Index my_num_edges() const;
        Index num_vertices() const { return num_verts; }

        void write_file(const char *fname, MPI_Comm comm) const;
        void redistribute_edges(MPI_Comm comm);

        Edge operator[](Index i) const { return myedges[i]; }

    private:

        EdgeVector myedges;
        Index num_verts;
};

#include "graph.hpp"

#endif
