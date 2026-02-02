#ifndef GRAPH_H_
#define GRAPH_H_

#include "utils.h"

class Graph
{
    public:

        using Edge = std::tuple<Index, Index, Real>;
        using EdgeVector = std::vector<Edge>;

        Graph(const EdgeVector& myedges, Index num_vertices);

        Index my_num_edges() const;

        void write_file(const char *fname, MPI_Comm comm) const;
        void redistribute_edges(MPI_Comm comm);

    private:

        EdgeVector myedges;
        Index num_vertices;
};

#include "graph.hpp"

#endif
