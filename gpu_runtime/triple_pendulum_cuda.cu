#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int StateSize = 6;
constexpr int MaxNewtonIterations = 30;
constexpr int MaxSubdivisions = 10;
constexpr double Gravity = 9.8;
constexpr double Tolerance = 2.0e-10;
constexpr double Pi = 3.14159265358979323846;

struct Options {
    int grid_size = 9;
    double angle_min = -Pi;
    double angle_max = Pi;
    double t_max = 8.0;
    double dt = 0.02;
    int tile_size = 1;
    bool resume = false;
    bool skip_csv = false;
    std::string csv = "triple_pendulum_volume_cuda.csv";
};

__device__ double wrap_angle(double value) {
    return atan2(sin(value), cos(value));
}

__device__ void mass_matrix(const double* q, double* matrix) {
    const double sums[3] = {3.0, 2.0, 1.0};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            matrix[a * 3 + b] =
                sums[a > b ? a : b] * cos(q[a] - q[b]);
        }
    }
}

__device__ bool solve3(const double* input, const double* rhs, double* result) {
    double a[3][4];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            a[row][column] = input[row * 3 + column];
        }
        a[row][3] = rhs[row];
    }
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (fabs(a[row][column]) > fabs(a[pivot][column])) {
                pivot = row;
            }
        }
        if (fabs(a[pivot][column]) < 1.0e-14) return false;
        for (int j = column; j < 4; ++j) {
            double temp = a[column][j];
            a[column][j] = a[pivot][j];
            a[pivot][j] = temp;
        }
        for (int row = column + 1; row < 3; ++row) {
            double factor = a[row][column] / a[column][column];
            for (int j = column; j < 4; ++j) a[row][j] -= factor * a[column][j];
        }
    }
    for (int row = 2; row >= 0; --row) {
        double value = a[row][3];
        for (int column = row + 1; column < 3; ++column) {
            value -= a[row][column] * result[column];
        }
        result[row] = value / a[row][row];
    }
    return true;
}

__device__ double hamiltonian(const double* state) {
    double matrix[9], velocity[3];
    mass_matrix(state, matrix);
    if (!solve3(matrix, state + 3, velocity)) return NAN;
    double kinetic = 0.5 * (state[3] * velocity[0] +
                            state[4] * velocity[1] +
                            state[5] * velocity[2]);
    double potential = Gravity * (3.0 * (1.0 - cos(state[0])) +
                                  2.0 * (1.0 - cos(state[1])) +
                                  1.0 * (1.0 - cos(state[2])));
    return kinetic + potential;
}

__device__ bool gradient(const double* state, double* output) {
    double matrix[9], velocity[3];
    mass_matrix(state, matrix);
    if (!solve3(matrix, state + 3, velocity)) return false;
    const double sums[3] = {3.0, 2.0, 1.0};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        double derivative[9] = {};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                if (coordinate == a && coordinate == b) continue;
                if (coordinate != a && coordinate != b) continue;
                double sign = coordinate == a ? 1.0 : -1.0;
                derivative[a * 3 + b] =
                    -sums[a > b ? a : b] * sin(state[a] - state[b]) * sign;
            }
        }
        double quadratic = 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                quadratic += velocity[a] * derivative[a * 3 + b] * velocity[b];
            }
        }
        output[coordinate] = -0.5 * quadratic +
            Gravity * sums[coordinate] * sin(state[coordinate]);
        output[coordinate + 3] = velocity[coordinate];
    }
    return true;
}

__device__ bool residual(const double* state0, const double* state1,
                         double step, double energy0, double* output) {
    double midpoint[6], gradient_value[6], delta[6];
    for (int i = 0; i < 6; ++i) {
        midpoint[i] = 0.5 * (state0[i] + state1[i]);
        delta[i] = state1[i] - state0[i];
    }
    if (!gradient(midpoint, gradient_value)) return false;
    double norm_squared = 0.0;
    double gradient_delta = 0.0;
    for (int i = 0; i < 6; ++i) {
        norm_squared += delta[i] * delta[i];
        gradient_delta += gradient_value[i] * delta[i];
    }
    double correction = (hamiltonian(state1) - energy0 - gradient_delta) /
                        fmax(norm_squared, 1.0e-28);
    for (int i = 0; i < 6; ++i) gradient_value[i] += correction * delta[i];
    for (int i = 0; i < 3; ++i) {
        output[i] = state1[i] - state0[i] - step * gradient_value[i + 3];
        output[i + 3] = state1[i + 3] - state0[i + 3] + step * gradient_value[i];
    }
    return true;
}

__device__ bool rhs(const double* state, double* output) {
    double gradient_value[6];
    if (!gradient(state, gradient_value)) return false;
    for (int i = 0; i < 3; ++i) {
        output[i] = gradient_value[i + 3];
        output[i + 3] = -gradient_value[i];
    }
    return true;
}

__device__ bool rk4(const double* state, double step, double* output) {
    double k1[6], k2[6], k3[6], k4[6], trial[6];
    if (!rhs(state, k1)) return false;
    for (int i = 0; i < 6; ++i) trial[i] = state[i] + 0.5 * step * k1[i];
    if (!rhs(trial, k2)) return false;
    for (int i = 0; i < 6; ++i) trial[i] = state[i] + 0.5 * step * k2[i];
    if (!rhs(trial, k3)) return false;
    for (int i = 0; i < 6; ++i) trial[i] = state[i] + step * k3[i];
    if (!rhs(trial, k4)) return false;
    for (int i = 0; i < 6; ++i) {
        output[i] = state[i] + step * (k1[i] + 2.0 * k2[i] +
                                       2.0 * k3[i] + k4[i]) / 6.0;
    }
    return true;
}

__device__ bool newton_step(const double* state0, double step, double* state1) {
    if (!rk4(state0, step, state1)) return false;
    double energy0 = hamiltonian(state0), residual_value[6];
    if (!isfinite(energy0)) return false;
    for (int iteration = 0; iteration < MaxNewtonIterations; ++iteration) {
        if (!residual(state0, state1, step, energy0, residual_value)) return false;
        double residual_norm = 0.0;
        for (double value : residual_value) residual_norm = fmax(residual_norm, fabs(value));
        if (residual_norm < Tolerance) return true;
        double jacobian[36], plus[6], minus[6], plus_residual[6], minus_residual[6];
        for (int column = 0; column < 6; ++column) {
            double epsilon = 2.0e-7 * fmax(1.0, fabs(state1[column]));
            for (int i = 0; i < 6; ++i) {
                plus[i] = state1[i]; minus[i] = state1[i];
            }
            plus[column] += epsilon; minus[column] -= epsilon;
            if (!residual(state0, plus, step, energy0, plus_residual) ||
                !residual(state0, minus, step, energy0, minus_residual)) return false;
            for (int row = 0; row < 6; ++row) {
                jacobian[row * 6 + column] =
                    (plus_residual[row] - minus_residual[row]) / (2.0 * epsilon);
            }
        }
        // Fixed-size Gaussian elimination for the 6x6 Newton system.
        double augmented[42];
        for (int row = 0; row < 6; ++row) {
            for (int column = 0; column < 6; ++column)
                augmented[row * 7 + column] = jacobian[row * 6 + column];
            augmented[row * 7 + 6] = residual_value[row];
        }
        for (int column = 0; column < 6; ++column) {
            int pivot = column;
            for (int row = column + 1; row < 6; ++row)
                if (fabs(augmented[row * 7 + column]) >
                    fabs(augmented[pivot * 7 + column])) pivot = row;
            if (fabs(augmented[pivot * 7 + column]) < 1.0e-14) return false;
            for (int j = column; j < 7; ++j) {
                double temp = augmented[column * 7 + j];
                augmented[column * 7 + j] = augmented[pivot * 7 + j];
                augmented[pivot * 7 + j] = temp;
            }
            for (int row = column + 1; row < 6; ++row) {
                double factor = augmented[row * 7 + column] /
                                augmented[column * 7 + column];
                for (int j = column; j < 7; ++j)
                    augmented[row * 7 + j] -= factor * augmented[column * 7 + j];
            }
        }
        double correction[6];
        for (int row = 5; row >= 0; --row) {
            double value = augmented[row * 7 + 6];
            for (int column = row + 1; column < 6; ++column)
                value -= augmented[row * 7 + column] * correction[column];
            correction[row] = value / augmented[row * 7 + row];
        }
        double old_norm = residual_norm, damping = 1.0;
        bool improved = false;
        while (damping >= 1.0 / 128.0) {
            double candidate[6], candidate_residual[6];
            for (int i = 0; i < 6; ++i)
                candidate[i] = state1[i] - damping * correction[i];
            if (residual(state0, candidate, step, energy0, candidate_residual)) {
                double candidate_norm = 0.0;
                for (double value : candidate_residual)
                    candidate_norm = fmax(candidate_norm, fabs(value));
                if (candidate_norm < old_norm) {
                    for (int i = 0; i < 6; ++i) state1[i] = candidate[i];
                    improved = true;
                    break;
                }
            }
            damping *= 0.5;
        }
        if (!improved) return false;
    }
    return false;
}

__device__ bool advance(const double* input, double step, double* output) {
    double current[6], candidate[6];
    for (int i = 0; i < 6; ++i) current[i] = input[i];
    int subdivisions = 1;
    for (int depth = 0; depth <= MaxSubdivisions; ++depth) {
        for (int i = 0; i < 6; ++i) current[i] = input[i];
        double substep = step / subdivisions;
        bool success = true;
        for (int part = 0; part < subdivisions; ++part) {
            if (!newton_step(current, substep, candidate)) {
                success = false; break;
            }
            for (int i = 0; i < 6; ++i) current[i] = candidate[i];
        }
        if (success) {
            for (int i = 0; i < 6; ++i) output[i] = current[i];
            return true;
        }
        subdivisions *= 2;
    }
    return false;
}

__global__ void simulate_kernel(const double* angles, double* finals,
                                unsigned char* failures, std::size_t total,
                                int steps, double dt) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    double state[6] = {angles[index * 3], angles[index * 3 + 1],
                       angles[index * 3 + 2], 0.0, 0.0, 0.0};
    double next[6];
    for (int step = 0; step < steps; ++step) {
        if (!advance(state, dt, next)) {
            failures[index] = 1;
            return;
        }
        for (int i = 0; i < 6; ++i) state[i] = next[i];
    }
    for (int i = 0; i < 3; ++i) finals[index * 3 + i] = state[i];
}

__global__ void chaos_kernel(const double* finals, double* chaos, int grid_size) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(grid_size) * grid_size * grid_size;
    if (index >= total) return;
    int i = index / (grid_size * grid_size);
    int j = (index / grid_size) % grid_size;
    int k = index % grid_size;
    if (i == 0 || j == 0 || k == 0 ||
        i == grid_size - 1 || j == grid_size - 1 || k == grid_size - 1) {
            chaos[index] = NAN;
        return;
    }
    const int neighbors[6] = {
        index - grid_size * grid_size, index + grid_size * grid_size,
        index - grid_size, index + grid_size, index - 1, index + 1};
    double sum = 0.0;
    for (int n = 0; n < 6; ++n) {
        double distance = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            double delta = wrap_angle(finals[neighbors[n] * 3 + axis] -
                                      finals[index * 3 + axis]);
            distance += delta * delta;
        }
        sum += sqrt(distance);
    }
    chaos[index] = sum / 6.0;
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--resume") { options.resume = true; continue; }
        if (key == "--skip-csv") { options.skip_csv = true; continue; }
        if (i + 1 >= argc) break;
        std::string value = argv[++i];
        if (key == "--grid-size") options.grid_size = std::stoi(value);
        else if (key == "--angle-min") options.angle_min = std::stod(value);
        else if (key == "--angle-max") options.angle_max = std::stod(value);
        else if (key == "--t-max") options.t_max = std::stod(value);
        else if (key == "--dt") options.dt = std::stod(value);
        else if (key == "--tile-size") options.tile_size = std::stoi(value);
        else if (key == "--csv") options.csv = value;
    }
    if (options.grid_size < 3 || options.tile_size < 1 ||
        options.tile_size > options.grid_size || options.dt <= 0.0 ||
        options.t_max <= 0.0) {
        throw std::invalid_argument(
            "grid-size must be >= 3, tile-size must be in [1, grid-size], "
            "and times must be positive"
        );
    }
    return options;
}
}  // namespace

namespace {

using Slice = std::vector<double>;

double host_wrap_angle(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

std::size_t slice_point_count(const Options& options) {
    const std::size_t grid = static_cast<std::size_t>(options.grid_size);
    return grid * grid;
}

void simulate_tile(const Options& options, int first_i, int slice_count,
                   int steps, std::ofstream& final_file) {
    const std::size_t grid = static_cast<std::size_t>(options.grid_size);
    const std::size_t slice_points = slice_point_count(options);
    const std::size_t point_count = static_cast<std::size_t>(slice_count) * slice_points;
    const std::size_t value_count = point_count * 3;
    std::vector<double> angles(value_count), finals(value_count);
    std::vector<unsigned char> failures(point_count);
    const double interval = options.angle_max - options.angle_min;

    for (std::size_t local = 0; local < point_count; ++local) {
        const std::size_t in_slice = local % slice_points;
        const int i = first_i + static_cast<int>(local / slice_points);
        const int j = static_cast<int>(in_slice / grid);
        const int k = static_cast<int>(in_slice % grid);
        angles[local * 3] = options.angle_min + interval * i / (options.grid_size - 1);
        angles[local * 3 + 1] = options.angle_min + interval * j / (options.grid_size - 1);
        angles[local * 3 + 2] = options.angle_min + interval * k / (options.grid_size - 1);
    }

    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const std::size_t required_bytes =
        value_count * sizeof(double) * 2 + point_count * sizeof(unsigned char);
    if (required_bytes > free_bytes) {
        throw std::runtime_error(
            "tile-size " + std::to_string(slice_count) + " requires about " +
            std::to_string(required_bytes / (1024.0 * 1024.0 * 1024.0)) +
            " GiB of GPU memory; reduce --tile-size"
        );
    }

    double* device_angles = nullptr;
    double* device_finals = nullptr;
    unsigned char* device_failures = nullptr;
    check_cuda(cudaMalloc(&device_angles, value_count * sizeof(double)), "cudaMalloc angles");
    check_cuda(cudaMalloc(&device_finals, value_count * sizeof(double)), "cudaMalloc finals");
    check_cuda(cudaMalloc(&device_failures, point_count * sizeof(unsigned char)), "cudaMalloc failures");
    check_cuda(cudaMemcpy(device_angles, angles.data(), value_count * sizeof(double),
                          cudaMemcpyHostToDevice), "copy angles");
    check_cuda(cudaMemset(device_failures, 0, point_count * sizeof(unsigned char)), "clear failures");

    const unsigned int blocks = static_cast<unsigned int>((point_count + 127) / 128);
    simulate_kernel<<<blocks, 128>>>(device_angles, device_finals, device_failures,
                                      point_count, steps, options.dt);
    check_cuda(cudaGetLastError(), "launch simulate kernel");
    check_cuda(cudaDeviceSynchronize(), "simulate kernel");
    check_cuda(cudaMemcpy(finals.data(), device_finals, value_count * sizeof(double),
                          cudaMemcpyDeviceToHost), "copy finals");
    check_cuda(cudaMemcpy(failures.data(), device_failures, point_count,
                          cudaMemcpyDeviceToHost), "copy failures");
    cudaFree(device_angles);
    cudaFree(device_finals);
    cudaFree(device_failures);

    for (std::size_t local = 0; local < point_count; ++local) {
        if (failures[local]) {
            throw std::runtime_error(
                "one or more GPU trajectories failed in tile starting at i=" +
                std::to_string(first_i)
            );
        }
    }
    final_file.write(reinterpret_cast<const char*>(finals.data()),
                     static_cast<std::streamsize>(value_count * sizeof(double)));
    // Flush after every tile so an interrupted run loses at most one tile.
    final_file.flush();
    if (!final_file) throw std::runtime_error("failed to write temporary final-state file");
}

std::string metadata_text(const Options& options) {
    std::ostringstream text;
    text << std::setprecision(17)
         << "grid_size=" << options.grid_size << '\n'
         << "angle_min=" << options.angle_min << '\n'
         << "angle_max=" << options.angle_max << '\n'
         << "t_max=" << options.t_max << '\n'
         << "dt=" << options.dt << '\n';
    return text.str();
}

void write_metadata(const std::string& path, const Options& options) {
    std::ofstream file(path, std::ios::trunc);
    file << metadata_text(options);
    if (!file) throw std::runtime_error("failed to write metadata file " + path);
}

// Returns the number of complete i-slices already stored in final_path and
// truncates a partially written trailing slice, if any.
int prepare_resume(const Options& options, const std::string& final_path,
                   const std::string& meta_path) {
    namespace fs = std::filesystem;
    if (!fs::exists(final_path)) {
        std::cout << "[resume] no " << final_path << " found; starting from slice 0\n";
        write_metadata(meta_path, options);
        return 0;
    }
    if (fs::exists(meta_path)) {
        std::ifstream meta(meta_path);
        std::ostringstream saved;
        saved << meta.rdbuf();
        if (saved.str() != metadata_text(options)) {
            throw std::runtime_error(
                "resume parameters do not match " + meta_path +
                "\n--- saved ---\n" + saved.str() +
                "--- current ---\n" + metadata_text(options));
        }
    } else {
        std::cout << "[resume] warning: no metadata file; assuming --grid-size, "
                     "--angle-min/max, --t-max and --dt match the original run\n";
        write_metadata(meta_path, options);
    }
    const std::uintmax_t slice_bytes =
        static_cast<std::uintmax_t>(slice_point_count(options)) * 3 * sizeof(double);
    const std::uintmax_t size = fs::file_size(final_path);
    const std::uintmax_t complete = size / slice_bytes;
    if (complete > static_cast<std::uintmax_t>(options.grid_size)) {
        throw std::runtime_error("temporary final-state file is larger than the grid; "
                                 "wrong --grid-size?");
    }
    if (size != complete * slice_bytes) {
        std::cout << "[resume] discarding partial slice ("
                  << (size - complete * slice_bytes) << " bytes)\n";
        fs::resize_file(final_path, complete * slice_bytes);
    }
    std::cout << "[resume] " << complete << '/' << options.grid_size
              << " slices already simulated\n" << std::flush;
    return static_cast<int>(complete);
}

void read_slice(std::ifstream& file, std::size_t slice_points, int slice_index,
                Slice& values) {
    values.resize(slice_points * 3);
    const std::streamoff offset = static_cast<std::streamoff>(slice_index) *
                                  static_cast<std::streamoff>(slice_points * 3 * sizeof(double));
    file.clear();
    file.seekg(offset);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!file) throw std::runtime_error("failed to read temporary final-state file");
}

double final_value(const Slice& slice, std::size_t point, int component) {
    return slice[point * 3 + component];
}

void compute_chaos(const Options& options, const std::string& final_path,
                   const std::string& chaos_path, double& minimum, double& maximum) {
    const int grid = options.grid_size;
    const std::size_t slice_points = slice_point_count(options);
    std::ifstream final_file(final_path, std::ios::binary);
    std::ofstream chaos_file(chaos_path, std::ios::binary | std::ios::trunc);
    if (!final_file || !chaos_file) throw std::runtime_error("failed to open temporary volume files");

    Slice previous, current, next, chaos(slice_points, NAN);
    read_slice(final_file, slice_points, 0, current);
    if (grid > 1) read_slice(final_file, slice_points, 1, next);
    minimum = INFINITY;
    maximum = -INFINITY;
    for (int i = 0; i < grid; ++i) {
        std::fill(chaos.begin(), chaos.end(), NAN);
        if (i > 0 && i < grid - 1) {
            for (int j = 1; j < grid - 1; ++j) {
                for (int k = 1; k < grid - 1; ++k) {
                    const std::size_t point = static_cast<std::size_t>(j) * grid + k;
                    const std::size_t above = point - grid;
                    const std::size_t below = point + grid;
                    double sum = 0.0;
                    const std::size_t neighbors[4] = {above, below, point - 1, point + 1};
                    for (std::size_t neighbor : neighbors) {
                        double distance = 0.0;
                        for (int axis = 0; axis < 3; ++axis) {
                            const double delta = host_wrap_angle(
                                final_value(current, neighbor, axis) -
                                final_value(current, point, axis));
                            distance += delta * delta;
                        }
                        sum += std::sqrt(distance);
                    }
                    double axial_sum = 0.0;
                    for (int side : {-1, 1}) {
                        const Slice& other = side < 0 ? previous : next;
                        double distance = 0.0;
                        for (int axis = 0; axis < 3; ++axis) {
                            const double delta = host_wrap_angle(
                                final_value(other, point, axis) -
                                final_value(current, point, axis));
                            distance += delta * delta;
                        }
                        axial_sum += std::sqrt(distance);
                    }
                    chaos[point] = (sum + axial_sum) / 6.0;
                    minimum = std::min(minimum, chaos[point]);
                    maximum = std::max(maximum, chaos[point]);
                }
            }
        }
        chaos_file.write(reinterpret_cast<const char*>(chaos.data()),
                         static_cast<std::streamsize>(chaos.size() * sizeof(double)));
        if (!chaos_file) throw std::runtime_error("failed to write temporary chaos file");
        std::cout << "[chaos] " << (i + 1) << '/' << grid << " slices\n" << std::flush;
        if (i + 1 < grid) {
            previous.swap(current);
            current.swap(next);
            if (i + 2 < grid) read_slice(final_file, slice_points, i + 2, next);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        const std::size_t grid = static_cast<std::size_t>(options.grid_size);
        const std::size_t total = grid * grid * grid;
        const std::size_t slice_points = slice_point_count(options);
        const int steps = static_cast<int>(std::ceil(options.t_max / options.dt));
        const std::string final_path = options.csv + ".final.bin";
        const std::string chaos_path = options.csv + ".chaos.bin";
        const std::string meta_path = options.csv + ".meta.txt";
        int start_slice = 0;
        if (options.resume) {
            start_slice = prepare_resume(options, final_path, meta_path);
        } else {
            write_metadata(meta_path, options);
        }
        std::ofstream final_file(final_path, std::ios::binary |
                                 (options.resume ? std::ios::app : std::ios::trunc));
        if (!final_file) throw std::runtime_error("failed to create temporary final-state file");
        for (int first_i = start_slice; first_i < options.grid_size; first_i += options.tile_size) {
            const int count = std::min(options.tile_size, options.grid_size - first_i);
            simulate_tile(options, first_i, count, steps, final_file);
            std::cout << "[simulate] i=" << std::min(first_i + count, options.grid_size)
                      << '/' << options.grid_size << " slices\n" << std::flush;
        }
        final_file.close();

        double minimum = 0.0, maximum = 0.0;
        compute_chaos(options, final_path, chaos_path, minimum, maximum);
        if (options.skip_csv) {
            std::cout << "Wrote " << chaos_path << " (min=" << minimum
                      << ", max=" << maximum << "); skipped CSV (--skip-csv)\n";
            return 0;
        }
        std::ifstream final_input(final_path, std::ios::binary);
        std::ifstream chaos_input(chaos_path, std::ios::binary);
        std::ofstream output(options.csv);
        output << "i,j,k,theta1_initial_rad,theta2_initial_rad,theta3_initial_rad,"
                  "theta1_final_rad,theta2_final_rad,theta3_final_rad,chaos_value,"
                  "chaos_value_normalized,alpha,is_interior\n" << std::setprecision(17);
        const double interval = options.angle_max - options.angle_min;
        Slice finals, chaos(slice_points);
        for (int i = 0; i < options.grid_size; ++i) {
            read_slice(final_input, slice_points, i, finals);
            chaos_input.read(reinterpret_cast<char*>(chaos.data()),
                             static_cast<std::streamsize>(chaos.size() * sizeof(double)));
            for (int j = 0; j < options.grid_size; ++j) {
                for (int k = 0; k < options.grid_size; ++k) {
                    const std::size_t point = static_cast<std::size_t>(j) * options.grid_size + k;
                    const bool interior = std::isfinite(chaos[point]);
                    const double normalized = interior ?
                        (chaos[point] - minimum) / (maximum - minimum + 1.0e-12) : 0.0;
                    const double alpha = interior ? 0.08 + 0.92 * normalized : 0.0;
                    output << i << ',' << j << ',' << k << ','
                           << options.angle_min + interval * i / (options.grid_size - 1) << ','
                           << options.angle_min + interval * j / (options.grid_size - 1) << ','
                           << options.angle_min + interval * k / (options.grid_size - 1) << ','
                           << final_value(finals, point, 0) << ','
                           << final_value(finals, point, 1) << ','
                           << final_value(finals, point, 2) << ',';
                    if (interior) output << chaos[point] << ',' << normalized << ',' << alpha << ",1\n";
                    else output << ",,,0\n";
                }
            }
            std::cout << "[csv] " << (i + 1) << '/' << options.grid_size << " slices\n" << std::flush;
        }
        output.close();
        std::remove(final_path.c_str());
        std::remove(chaos_path.c_str());
        std::remove(meta_path.c_str());
        std::cout << "Wrote " << total << " rows to " << options.csv << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
