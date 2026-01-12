#ifndef WAVEMAP_CORE_UTILS_TIME_STOPWATCH_H_
#define WAVEMAP_CORE_UTILS_TIME_STOPWATCH_H_

#include <chrono>

using Time = std::chrono::steady_clock;
using Timestamp = std::chrono::time_point<Time>;
using Duration = Timestamp::duration;

template <typename T>
T to_seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<T>>(duration).count();
}

class Stopwatch {
public:
  void start();
  void stop();

  double getLastEpisodeDuration() const {
    return to_seconds<double>(last_episode_duration_);
  }
  double getTotalDuration() const {
    return to_seconds<double>(total_duration_);
  }

private:
  bool running = false;

  Timestamp episode_start_time_{};
  Duration last_episode_duration_{};
  Duration total_duration_{};
};

#endif // WAVEMAP_CORE_UTILS_TIME_STOPWATCH_H_
