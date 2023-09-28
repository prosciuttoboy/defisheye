package dev.prosciutto.defisheye;

import java.nio.file.Path;

interface CameraCalibration {

  void undistortImage(final Path source, final Path destination);

  void undistortVideo(final Path source, final Path destination);
}
