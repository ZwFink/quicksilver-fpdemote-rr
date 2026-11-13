// Prints the QS struct offsets that qs_compare.py needs to find the scalar-flux tally, as
// JSON. Built with the same HIP flags as Quicksilver so the layouts match the recorded binary.
#include <cstddef>
#include <cstdio>
#define private public
#define protected public
#include "MonteCarlo.hh"
#include "Tallies.hh"
#include "BulkStorage.hh"
#include "QS_Vector.hh"
#undef private
#undef protected

#pragma clang diagnostic ignored "-Winvalid-offsetof"
#define OFF(T, m) std::printf("  \"%s::%s\": %zu,\n", #T, #m, offsetof(T, m))
#define SZ(T) std::printf("  \"sizeof(%s)\": %zu,\n", #T, sizeof(T))

int main() {
  std::printf("{\n");
  OFF(MonteCarlo, _tallies);
  OFF(Tallies, _scalarFluxDomain);
  OFF(qs_vector<int>, _data); OFF(qs_vector<int>, _size);
  SZ(ScalarFluxDomain); OFF(ScalarFluxDomain, _task);
  SZ(ScalarFluxTask); OFF(ScalarFluxTask, _scalarFluxCellStorage);
  OFF(BulkStorage<double>, _bulkStorage); OFF(BulkStorage<double>, _size);
  std::printf("  \"sizeof(void*)\": %zu\n}\n", sizeof(void *));
}
