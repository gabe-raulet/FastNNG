template <class Atom_>
PointContainer<Atom_>::PointContainer() : offsets({0}) {}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, const IndexVector& sizes) : data(atoms), offsets(sizes.size()+1)
{
    Index disp = 0;
    Index point_count = sizes.size();

    for (Index i = 0; i < point_count; ++i)
    {
        offsets[i] = disp;
        disp += sizes[i];
    }

    offsets[point_count] = disp;
}

template <class Atom_>
PointContainer<Atom_>::PointContainer(const AtomVector& atoms, Index size, Index dim) : data(atoms), offsets(size+1)
{
    Index disp = 0;

    for (Index i = 0; i <= size; ++i)
    {
        offsets[i] = disp;
        disp += dim;
    }
}

template <class Atom_>
Index PointContainer<Atom_>::num_points() const
{
    return offsets.size()-1;
}

template <class Atom_>
Index PointContainer<Atom_>::num_atoms() const
{
    return data.size();
}

template <class Atom_>
const Atom_* PointContainer<Atom_>::point_mem(Index i) const
{
    return &data[offsets[i]];
}

template <class Atom_>
Index PointContainer<Atom_>::point_dim(Index i) const
{
    return offsets[i+1]-offsets[i];
}

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname)
{
    assert((std::same_as<Atom, float>));

    std::ifstream is;
    size_t filesize, total;
    std::vector<char> p;
    int d;

    is.open(fname, std::ios::binary | std::ios::in);

    is.seekg(0, is.end);
    filesize = is.tellg();
    is.seekg(0, is.beg);

    is.read((char*)&d, sizeof(int));
    is.seekg(0, is.beg);

    size_t point_size = sizeof(Atom)*d;
    size_t record_size = sizeof(int) + point_size;

    assert((filesize % record_size == 0));
    total = filesize / record_size;

    p.resize(record_size);

    data.resize(total*d);
    offsets.resize(total+1);

    Index disp = 0;

    for (size_t i = 0; i < total; ++i)
    {
        is.read(p.data(), record_size);

        const char *ds = p.data();
        const char *ps = p.data() + sizeof(int);

        int dt;
        memcpy(&dt, ds, sizeof(int)); assert((dt == d));

        char *dest = (char*)(&data[i*d]);
        memcpy(dest, ps, point_size);

        offsets[i] = disp;
        disp += d;
    }

    is.close();
    offsets[total] = disp;

    return total;
}

template <class Atom_>
Index PointContainer<Atom_>::read_fvecs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, float>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_File fh;
    MPI_Aint extent;
    MPI_Offset filesize, filedisp;
    Index total, myleft, mysize;
    int dim;

    MPI_Datatype MPI_POINT;
    MPI_Datatype MPI_ATOM = MPI_FLOAT;

    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    if (!myrank)
    {
        MPI_File_get_size(fh, &filesize);
        MPI_File_read(fh, &dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
    MPI_Bcast(&filesize, 1, MPI_OFFSET, 0, comm);

    extent = 4 * (dim + 1);
    total = filesize / extent;

    assert((filesize % extent == 0));

    mysize = total / nprocs;
    myleft = total % nprocs;

    if (myrank < myleft)
        mysize++;

    IndexVector sizes(nprocs);
    sizes[myrank] = mysize;

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INDEX, sizes.data(), 1, MPI_INDEX, comm);

    IndexVector offsets(nprocs);
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), (Index)0);
    Index totsize = offsets.back() + sizes.back();
    Index myoffset = offsets[myrank];

    data.resize(mysize*dim);
    offsets.resize(mysize+1);

    assert((dim >= 1));
    MPI_Type_contiguous(dim, MPI_ATOM, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    MPI_Datatype filetype;
    MPI_Type_create_resized(MPI_POINT, 0, extent, &filetype);
    MPI_Type_commit(&filetype);

    filedisp = myoffset*extent + sizeof(int);
    MPI_File_set_view(fh, filedisp, MPI_POINT, filetype, "native", MPI_INFO_NULL);

    MPI_File_read(fh, data.data(), (int)mysize, MPI_POINT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    MPI_Type_free(&filetype);
    MPI_Type_free(&MPI_POINT);

    Index disp = 0;

    for (Index i = 0; i <= mysize; ++i)
    {
        offsets[i] = disp;
        disp += dim;
    }

    return total;
}
