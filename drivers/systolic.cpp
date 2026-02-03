#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <algorithm>

#include "utils.h"
#include "point.h"
#include "search.h"
#include "graph.h"

MPI_Comm comm;
int myrank, nprocs;

Real radius = -1;
const char *infile = NULL;
const char *outfile = NULL;
const char *metric = "l2";

Real cover = 1.5;
Index leaf_size = 10;
int verbosity = 1;

template <class Atom>
struct L2Distance
{
    Index distcomps = 0;
    Real operator()(const Atom* p, const Atom* q, Index m, Index n);
};

template <class Atom>
struct EditDistance
{
    Index distcomps = 0;
    Real operator()(const Atom* s, const Atom* t, Index m, Index n);
};

template <class Atom, class Distance>
int main_mpi(int argc, char *argv[]);

void parse_cmdline(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    int err;
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    parse_cmdline(argc, argv);

    if (!strcmp(metric, "edit")) err = main_mpi<char, EditDistance<char>>(argc, argv);
    else if (!strcmp(metric, "l2")) err = main_mpi<float, L2Distance<float>>(argc, argv);

    MPI_Comm_free(&comm);
    MPI_Finalize();
    return err;
}

template <class Atom, class Distance>
int main_mpi(int argc, char *argv[])
{
    using PointContainerType = PointContainer<Atom>;
    using AtomVector = std::vector<Atom>;

    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    double mytime, time;
    double mytottime, tottime;

    Index mydistcomps, distcomps;
    Index mytotdistcomps, totdistcomps;

    Index size, mysize, myoffset;
    PointContainerType mypoints;
    Distance distance;

    EdgeVector myedges;

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    mytottime = -MPI_Wtime();

    if (!strcmp(metric, "edit")) size = mypoints.read_seqs(infile, comm);
    else if (!strcmp(metric, "l2")) size = mypoints.read_fvecs(infile, comm);

    mytime += MPI_Wtime();

    if (verbosity >= 1)
    {
        Index num_atoms, my_num_atoms = mypoints.num_atoms();

        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&my_num_atoms, &num_atoms, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) fprintf(stderr, "[time=%.3f] read input file '%s' [size=%lld,atoms=%s]\n", time, infile, size, LARGE(num_atoms));
        fflush(stderr);
    }

    mytime = -MPI_Wtime();
    mydistcomps = distance.distcomps;

    CoverTree tree(cover, leaf_size);
    tree.build(mypoints, distance);

    mytime += MPI_Wtime();
    mydistcomps = distance.distcomps - mydistcomps;

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) fprintf(stderr, "[time=%.3f] built cover trees [distcomps=%s,avg_distcomps=%s]\n", time, LARGE(distcomps), LARGE(static_cast<Index>((distcomps+0.0)/nprocs)));
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();
    mydistcomps = distance.distcomps;

    mysize = mypoints.num_points();
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    PointContainerType sendbuf = mypoints;
    PointContainerType recvbuf;

    int recvrank = (myrank+1)%nprocs;
    int sendrank = (myrank-1+nprocs)%nprocs;

    int sendcount_buf[3], recvcount_buf[3];

    int sendcount, sendcount_atoms;
    int recvcount, recvcount_atoms;
    Index sendoffset, recvoffset;

    MPI_Request reqs[6];

    sendoffset = myoffset;

    for (int step = 0; step <= nprocs/2; ++step)
    {
        sendcount = sendbuf.num_points();
        sendcount_atoms = sendbuf.num_atoms();

        sendcount_buf[0] = sendcount;
        sendcount_buf[1] = sendcount_atoms;
        sendcount_buf[2] = sendoffset;

        MPI_Irecv(recvcount_buf, 3, MPI_INT, recvrank, myrank,   comm, &reqs[0]);
        MPI_Isend(sendcount_buf, 3, MPI_INT, sendrank, sendrank, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        recvcount = recvcount_buf[0];
        recvcount_atoms = recvcount_buf[1];
        recvoffset = recvcount_buf[2];

        AtomVector& senddata = sendbuf.getdata();
        IndexVector& sendoffsets = sendbuf.getoffsets();

        AtomVector& recvdata = recvbuf.getdata();
        IndexVector& recvoffsets = recvbuf.getoffsets();

        recvdata.resize(recvcount_atoms);
        recvoffsets.resize(recvcount+1);

        MPI_Irecv(recvdata.data(), recvcount_atoms, MPI_ATOM, recvrank, myrank+nprocs, comm, &reqs[0]);
        MPI_Isend(senddata.data(), sendcount_atoms, MPI_ATOM, sendrank, sendrank+nprocs, comm, &reqs[1]);

        MPI_Irecv(recvoffsets.data(), recvcount+1, MPI_INDEX, recvrank, myrank+2*nprocs, comm, &reqs[2]);
        MPI_Isend(sendoffsets.data(), sendcount+1, MPI_INDEX, sendrank, sendrank+2*nprocs, comm, &reqs[3]);

        Index sendsize = sendbuf.num_points();

        auto functor = [&](Index neighbor, Index query, Real weight)
        {
            myedges.emplace_back(neighbor+myoffset, query+sendoffset, weight);
        };

        tree.radius_query_batched(mypoints, distance, sendbuf, radius, functor);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        std::swap(senddata, recvdata);
        std::swap(sendoffsets, recvoffsets);
        std::swap(sendoffset, recvoffset);
    }

    mytime += MPI_Wtime();
    mydistcomps = distance.distcomps - mydistcomps;

    if (verbosity >= 1)
    {
        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) fprintf(stderr, "[time=%.3f] queried neighbors [distcomps=%s,avg_distcomps=%s]\n", time, LARGE(distcomps), LARGE(static_cast<Index>((distcomps+0.0)/nprocs)));
        fflush(stderr);
    }

    MPI_Barrier(comm);
    mytime = -MPI_Wtime();

    Graph graph(myedges, size);
    graph.redistribute_edges(comm);

    mytime += MPI_Wtime();
    mytottime += MPI_Wtime();
    mytotdistcomps = distance.distcomps;

    if (verbosity >= 1)
    {
        Index num_edges;
        Index my_num_edges = graph.my_num_edges();

        MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&my_num_edges, &num_edges, 1, MPI_INDEX, MPI_SUM, 0, comm);

        if (!myrank) fprintf(stderr, "[time=%.3f] redistributed edges [points=%lld,edges=%lld,density=%.3f]\n", time, size, num_edges, (num_edges+0.0)/size);
        fflush(stderr);
    }

    if (outfile)
    {
        MPI_Barrier(comm);
        mytime = -MPI_Wtime();
        graph.write_file(outfile, comm);
        mytime += MPI_Wtime();

        if (verbosity >= 1)
        {
            MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!myrank) fprintf(stderr, "[time=%.3f] wrote edges to file '%s'\n", time, outfile);
            fflush(stderr);
        }
    }

    MPI_Reduce(&mytottime, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&mytotdistcomps, &totdistcomps, 1, MPI_INDEX, MPI_SUM, 0, comm);
    if (!myrank) fprintf(stderr, "[time=%.3f] complete [distcomps=%s,avg_distcomps=%s]\n", tottime, LARGE(totdistcomps), LARGE(static_cast<Index>((totdistcomps+0.0)/nprocs)));
    fflush(stderr);

    return 0;
}

void parse_cmdline(int argc, char *argv[])
{
    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <points> -r <radius>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT cover tree base [%.2f]\n", cover);
            fprintf(stderr, "         -l INT   leaf size [%lld]\n", leaf_size);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -D STR   metric [%s]\n", metric);
            fprintf(stderr, "         -o FILE  output edge file\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "i:r:c:l:v:o:D:h")) >= 0)
    {

        if      (c == 'i') infile = optarg;
        else if (c == 'r') radius = atof(optarg);
        else if (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'D') metric = optarg;
        else if (c == 'o') outfile = optarg;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!infile)
    {
        if (!myrank) fprintf(stderr, "error: missing input file argument! (-i)\n");
        usage(1, myrank == 0);
    }

    if (radius < 0)
    {
        if (!myrank) fprintf(stderr, "error: missing radius argument! (-r)\n");
        usage(1, myrank == 0);
    }

    if (strcmp(metric, "edit") && strcmp(metric, "l2"))
    {
        if (!myrank) fprintf(stderr, "error: invalid metric argument! (-D)\n");
        usage(1, myrank == 0);
    }
}

template <class Atom>
Real L2Distance<Atom>::operator()(const Atom* p, const Atom* q, Index m, Index n)
{
    assert((m == n));

    Real val = 0;
    Real delta;

    for (Index i = 0; i < m; ++i)
    {
        delta = static_cast<Real>(p[i] - q[i]);
        val += delta*delta;
    }

    distcomps++;

    return std::sqrt(val);
}

template <class Atom>
Real EditDistance<Atom>::operator()(const Atom* s, const Atom* t, Index m, Index n)
{
    IndexVector v0(n+1), v1(n+1);

    for (Index i = 0; i <= n; ++i)
        v0[i] = i;

    for (Index i = 0; i < m; ++i)
    {
        v1[0] = i+1;

        for (Index j = 0; j < n; ++j)
        {
            Index del = v0[j+1]+1;
            Index ins = v1[j+0]+1;
            Index sub = (s[i] == t[j])? v0[j] : v0[j]+1;

            v1[j+1] = std::min(del, std::min(ins, sub));
        }

        std::swap(v0, v1);
    }

    distcomps++;

    return static_cast<Real>(v0[n]);
}
