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
const Atom_* PointContainer<Atom_>::mem(Index i) const
{
    return &data[offsets[i]];
}

template <class Atom_>
Index PointContainer<Atom_>::size(Index i) const
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
Index PointContainer<Atom_>::read_seqs(const char *fname)
{
    assert((std::same_as<Atom, char>));

    std::ifstream is;
    std::string line;

    Index id = 0;

    is.open(fname, std::ios::in);

    offsets.clear();
    data.clear();

    while (std::getline(is, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        offsets.push_back(data.size());
        std::copy(line.begin(), line.end(), std::back_inserter(data));
    }

    offsets.push_back(data.size());
    is.close();

    return offsets.size()-1;
}

template <class Atom_>
Index PointContainer<Atom_>::read_seqs(const char *fname, MPI_Comm comm)
{
    assert((std::same_as<Atom, char>));

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_ATOM = MPI_CHAR;

    AtomVector alldata;
    IndexVector alloffsets;

    Index size, myoffset, mysize, myleft, atoms;

    if (!myrank)
    {
        PointContainer<Atom> allpoints;
        allpoints.read_seqs(fname);

        AtomVector& alldata_ref = allpoints.getdata();
        IndexVector& alloffsets_ref = allpoints.getoffsets();

        alldata.assign(alldata_ref.begin(), alldata_ref.end());
        alloffsets.assign(alloffsets_ref.begin(), alloffsets_ref.end());

        size = allpoints.num_points();
        atoms = allpoints.num_atoms();
    }

    MPI_Bcast(&size, 1, MPI_INDEX, 0, comm);
    MPI_Bcast(&atoms, 1, MPI_INDEX, 0, comm);

    if (myrank != 0)
    {
        alldata.resize(atoms);
        alloffsets.resize(size+1);
    }

    MPI_Bcast(alldata.data(), static_cast<int>(atoms), MPI_ATOM, 0, comm);
    MPI_Bcast(alloffsets.data(), static_cast<int>(size+1), MPI_INDEX, 0, comm);

    mysize = size/nprocs;
    myleft = size%nprocs;

    if (myrank < myleft)
        mysize++;

    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

    data.clear();
    offsets.clear();

    for (Index i = myoffset; i < myoffset+mysize; ++i)
    {
        Index dataoffset = alloffsets[i];
        Index datasize = alloffsets[i+1] - dataoffset;

        auto first = alldata.begin() + dataoffset;
        auto last = first + datasize;

        offsets.push_back(data.size());
        std::copy(first, last, std::back_inserter(data));
    }

    offsets.push_back(data.size());

    return size;
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

    Index totsize;
    Index myoffset;

    MPI_Allreduce(&mysize, &totsize, 1, MPI_INDEX, MPI_SUM, comm);
    MPI_Exscan(&mysize, &myoffset, 1, MPI_INDEX, MPI_SUM, comm);
    if (!myrank) myoffset = 0;

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

template <class Atom_>
void PointContainer<Atom_>::localgather(const PointContainer& points, const IndexVector& local_indices)
{
    Index size = local_indices.size();
    Index atom_count = 0;

    offsets.resize(size+1);

    for (Index i = 0; i < size; ++i)
    {
        offsets[i] = atom_count;
        atom_count += points.size(local_indices[i]);
    }

    offsets[size] = atom_count;

    data.clear();
    data.reserve(atom_count);

    for (Index i = 0; i < size; ++i)
    {
        const Atom *mem = points.mem(local_indices[i]);
        Index dim = points.size(local_indices[i]);
        data.insert(data.end(), mem, mem+dim);
    }
}

template <class Atom_>
void PointContainer<Atom_>::allgather(const PointContainer& mypoints, MPI_Comm comm)
{
    MPI_Datatype MPI_ATOM = mpi_type<Atom>();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    data.clear();
    offsets.clear();

    const AtomVector& sendbuf_atoms = mypoints.data;
    AtomVector& recvbuf_atoms = data;

    Index mysize = mypoints.num_points();

    IndexVector sendbuf_sizes(mysize);
    IndexVector recvbuf_sizes;

    for (Index i = 0; i < mysize; ++i)
    {
        sendbuf_sizes[i] = mypoints.size(i);
    }

    std::vector<int> recvcounts(nprocs), rdispls(nprocs);
    std::vector<int> recvcounts_atoms(nprocs), rdispls_atoms(nprocs);

    recvcounts[myrank] = mysize;
    recvcounts_atoms[myrank] = sendbuf_atoms.size();

    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts_atoms.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);
    std::exclusive_scan(recvcounts_atoms.begin(), recvcounts_atoms.end(), rdispls_atoms.begin(), 0);

    int totrecv = recvcounts.back() + rdispls.back();
    int totrecv_atoms = recvcounts_atoms.back() + rdispls_atoms.back();

    recvbuf_atoms.resize(totrecv_atoms);
    recvbuf_sizes.resize(totrecv);

    MPI_Allgatherv(sendbuf_sizes.data(), recvcounts[myrank], MPI_INDEX, recvbuf_sizes.data(), recvcounts.data(), rdispls.data(), MPI_INDEX, comm);
    MPI_Allgatherv(sendbuf_atoms.data(), recvcounts_atoms[myrank], MPI_ATOM, recvbuf_atoms.data(), recvcounts_atoms.data(), rdispls_atoms.data(), MPI_ATOM, comm);

    offsets.resize(totrecv);

    std::exclusive_scan(recvbuf_sizes.begin(), recvbuf_sizes.end(), offsets.begin(), (Index)0);
    offsets.push_back(offsets.back() + recvbuf_sizes.back());

    assert((offsets.back() == recvbuf_atoms.size()));
}

template <class Atom_>
template <class Distance>
VoronoiDiagram<Atom_>::VoronoiDiagram(const PointContainerType& points, const PointContainerType& centers, Distance& distance) : centers(centers), cell_indices(points.num_points(), 0), dist_to_centers(points.num_points(), std::numeric_limits<Real>::max())
{
    Index size = points.num_points();
    Index num_centers = centers.num_points();

    for (Index i = 0; i < size; ++i)
    {
        for (Index cell_index = 0; cell_index < num_centers; ++cell_index)
        {
            Real dist = distance(centers.mem(cell_index), points.mem(i), centers.size(cell_index), points.size(i));

            if (dist <= dist_to_centers[i])
            {
                dist_to_centers[i] = dist;
                cell_indices[i] = cell_index;
            }
        }
    }
}
