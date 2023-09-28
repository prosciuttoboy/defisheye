package dev.prosciutto.defisheye;

import java.nio.file.Path;
import java.util.Collection;
import java.util.Objects;

record CameraCalibrationConfiguration(Collection<Path> images, int chessboardWidth, int chessboardHeight) {

  public CameraCalibrationConfiguration {
    Objects.requireNonNull(images);
    if (images.isEmpty()
        || images.stream()
        .anyMatch(Objects::isNull)) {
      throw new IllegalArgumentException();
    }
    if (chessboardWidth <= 0) {
      throw new IllegalArgumentException();
    }
    if (chessboardHeight <= 0) {
      throw new IllegalArgumentException();
    }
  }
}
