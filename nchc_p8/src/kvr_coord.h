// kvr_coord.h — SRPT coordinator skeleton for restores that share one SSD array (P3-dev).
//
// A POSIX shared-memory table (shm_open NAME) holds one record per active restore: owner pid
// and bytes not yet submitted. The array is granted to the restore with the fewest bytes
// remaining (ties: lowest slot). Reader threads ask coord_turn() before each new read; a
// restore that loses the grant stops submitting (in-flight reads finish) and resumes when it
// is the shortest again — preemptive SRPT at item granularity. Lock-free: every field is a
// std::atomic (address-free on x86-64), so no process-shared mutex can be left locked by a
// crash; a record whose pid is gone is reclaimed.
#pragma once
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>

constexpr int COORD_MAX = 64;

struct CoordRec {
  std::atomic<int> pid;
  std::atomic<long long> remaining;  // bytes not yet submitted; <= 0: nothing left to read
};
struct CoordTable {
  CoordRec r[COORD_MAX];
};

struct Coord {
  CoordTable* t = nullptr;
  int slot = -1;
  double next_live_check = 0;
};

static double coord_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static bool pid_dead(int pid) { return pid > 0 && kill(pid, 0) != 0 && errno == ESRCH; }

// Map (creating if needed) the shared table. A new shm object is zero-filled = all free.
static Coord* coord_open(const char* name) {
  int fd = shm_open(name, O_RDWR | O_CREAT, 0600);
  if (fd < 0) { perror("shm_open"); exit(1); }
  if (ftruncate(fd, sizeof(CoordTable)) != 0) { perror("ftruncate shm"); exit(1); }
  void* p = mmap(nullptr, sizeof(CoordTable), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { perror("mmap shm"); exit(1); }
  Coord* c = new Coord;
  c->t = (CoordTable*)p;
  return c;
}

// Enter the queue with `bytes` to read. Call before the restore starts.
static void coord_register(Coord* c, long long bytes) {
  int me = getpid();
  for (int i = 0;; i = (i + 1) % COORD_MAX) {
    CoordRec& r = c->t->r[i];
    int p = r.pid.load();
    if ((p == 0 || pid_dead(p)) && r.pid.compare_exchange_strong(p, me)) {
      r.remaining.store(bytes);
      c->slot = i;
      return;
    }
    if (i == COORD_MAX - 1) usleep(1000);  // table full: wait for a free slot
  }
}

// Called once per read submitted.
static void coord_consume(Coord* c, long long bytes) {
  c->t->r[c->slot].remaining.fetch_sub(bytes);
}

// True when this restore holds the grant (fewest bytes remaining among records with work).
static bool coord_turn(Coord* c) {
  long long best = LLONG_MAX;
  int win = -1;
  for (int i = 0; i < COORD_MAX; i++) {
    CoordRec& r = c->t->r[i];
    if (r.pid.load(std::memory_order_acquire) == 0) continue;
    long long rem = r.remaining.load(std::memory_order_acquire);
    if (rem > 0 && rem < best) { best = rem; win = i; }
  }
  if (win == c->slot || win < 0) return true;
  // Reclaim the winner's record if its process died (rate-limited: kill() is a syscall).
  double t = coord_now();
  if (t >= c->next_live_check) {
    c->next_live_check = t + 0.01;
    CoordRec& w = c->t->r[win];
    int p = w.pid.load();
    if (pid_dead(p)) {
      w.remaining.store(0);
      w.pid.compare_exchange_strong(p, 0);
    }
  }
  return false;
}

static void coord_done(Coord* c) {
  if (c->slot < 0) return;
  c->t->r[c->slot].remaining.store(0);
  c->t->r[c->slot].pid.store(0, std::memory_order_release);
  c->slot = -1;
}
