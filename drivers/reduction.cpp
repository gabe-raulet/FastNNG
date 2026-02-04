#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <algorithm>

#include "utils.h"
#include "reduction.h"

MPI_Comm comm;
int myrank, nprocs;

const char *infile = NULL;
const char *outfile = NULL;
int verbosity = 1;

void parse_cmdline(int argc, char *argv[]);
int main_mpi(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    int err;
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);
    parse_cmdline(argc, argv);
    err = main_mpi(argc, argv);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return err;
}

int main_mpi(int argc, char *argv[])
{
    double mytime, time;

    RipsFiltration filtration;
    filtration.read_file(infile);

    Index num_simplices = filtration.size();
    Index num_vertices = filtration.num_vertices();

    BoundaryMatrix bd_matrix(filtration);

    bd_matrix.reduce();

    if (outfile)
    {
        FILE *f = fopen(outfile, "w");
        bd_matrix.write_homology_persistence(f);
        fclose(f);
    }

    return 0;
}

void parse_cmdline(int argc, char *argv[])
{
    auto usage = [&](int err, bool print)
    {
        if (print)
        {
            fprintf(stderr, "Usage: %s [options] -i <filtration>\n", argv[0]);
            fprintf(stderr, "         -v INT   verbosity level [%d]\n", verbosity);
            fprintf(stderr, "         -o FILE  output persistence file\n");
            fprintf(stderr, "         -h       help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "i:v:o:h")) >= 0)
    {
        if      (c == 'i') infile = optarg;
        else if (c == 'v') verbosity = atoi(optarg);
        else if (c == 'o') outfile = optarg;
        else if (c == 'h') usage(0, myrank == 0);
    }

    if (!infile)
    {
        if (!myrank) fprintf(stderr, "error: missing input file argument! (-i)\n");
        usage(1, myrank == 0);
    }
}
