# Build

The supported reference build is Linux x86-64 with GCC 11.4 and CMake 3.22.
Use Release without fast-math. The CMake targets retain H2's optimization flags
and the admission sources' original -O2 compilation. The runner fixes
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, and OPENCV_FOR_THREADS_NUM=1.
Do not change solver threading when comparing trajectories.

## Dependencies

| Dependency | Reference version/revision | Use |
|---|---|---|
| Eigen | 3.4.0 | Linear algebra |
| OpenCV | 3.4.20, `1eb1d4c3708f2bd95562cedd58d28461505c2d37` | Images, tracking, ORB, calibration I/O |
| Sophus | `db218a249202fe63ac13248b5f565b0d385f6640` | SO(3)/SE(3) |
| g2o | `1b89597b4e495bef5447c5b8001f5f93ccf32ec4` | Graph optimization and legacy framework support |
| Pangolin | `dd801d244db3a8e27b7fe8020cd751404aa818fd` | Compiled framework viewer support |
| SuiteSparse/CSparse | System development package | Sparse solver support |
| glog | System development package | Logging |
| SQLite3 | 3.37.2 reference | Read-only ROS2 SQLite bag ingestion |
| OpenMP/pthreads | Compiler/system | Existing runtime threading |
| OpenGL/GLU, GLEW, GLUT | System development packages | Pangolin/framework linkage |
| GoogleTest | 1.11 reference; tests only | Synthetic checks |
| Python | 3.8+; standard library only | Runner |
| FBOW | Vendored inference-only subset | Loop vocabulary lookup |

Install dependency development headers and build the pinned OpenCV, Sophus,
g2o, and Pangolin revisions into a separate prefix using their upstream build
instructions. Their source repositories are
[opencv/opencv](https://github.com/opencv/opencv),
[strasdat/Sophus](https://github.com/strasdat/Sophus),
[RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o), and
[stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin).
No dependencies are downloaded by this project's CMake configuration.

On Debian/Ubuntu, the system development packages include `libeigen3-dev`,
`libsuitesparse-dev`, `libgoogle-glog-dev`, `libsqlite3-dev`, `libgtest-dev`,
`libgl1-mesa-dev`, `libglu1-mesa-dev`, `libglew-dev`, and `freeglut3-dev`.
Compiler, CMake, and Python are also needed. OpenCV 4 is deliberately rejected
because it is not the frozen H2 image-processing implementation.

```sh
export DEPENDENCY_PREFIX=/path/to/dependencies/install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$DEPENDENCY_PREFIX" -DG2O_ROOT="$DEPENDENCY_PREFIX"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Build outputs stay in the build directory. Public sources and scripts contain
no machine-specific dependency paths. Standard CMake prefix/package variables
can be used for separately installed dependencies. Native runtime shared
library paths are set by CMake's build RPATH.

## Synthetic H2 checks

The unit target verifies the effective retained-window option, chronological
12-source/11-retained FIFO behavior, connected finite marginalization priors,
timestamp/duplicate protections, global landmark ranking, native camera1 floor,
feasible-floor shortfalls, fixed active budgets, and protected prior IDs.
It contains synthetic data only and performs no benchmark replay or GT access.
The 1.668333-second horizon describes the normal 30 Hz / five-frame cadence;
individual timestamps and skipped frames can change the elapsed span.
